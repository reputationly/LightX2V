import functools
import os
from pathlib import Path
from typing import NamedTuple

import torch
from loguru import logger
from transformers import AutoImageProcessor, Gemma3Processor, PreTrainedModel, ProcessorMixin
from transformers.modeling_outputs import BaseModelOutputWithPast

from lightx2v.models.input_encoders.hf.ltx2.gemma.embeddings_processor import EmbeddingsProcessor
from lightx2v.models.input_encoders.hf.ltx2.gemma.tokenizer import LTXVGemmaTokenizer
from lightx2v.models.input_encoders.hf.ltx2.utils import ModuleOps, find_matching_file
from lightx2v_platform.base.global_var import AI_DEVICE


class GemmaEncoderOutput(NamedTuple):
    video_encoding: torch.Tensor
    audio_encoding: torch.Tensor | None
    attention_mask: torch.Tensor


class GemmaTextEncoder(torch.nn.Module):
    """Unified Gemma text encoder with 3-block pipeline.
    Block 1: Gemma model (runs LLM, gets hidden states)
    Block 2: Feature extractor
    Block 3: Embeddings processor (connector with optional audio)
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        embeddings_processor: EmbeddingsProcessor,
        model: PreTrainedModel | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: ProcessorMixin | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.feature_extractor = feature_extractor.to(dtype=dtype)
        self.embeddings_processor = embeddings_processor.to(dtype=dtype)
        # When True, the Gemma backbone stays resident on CPU and is streamed to the GPU one
        # submodule at a time (see _run_text_model_layerwise_on_gpu), so only the module in
        # flight is ever GPU-resident — largest is embed_tokens at 2.0GB (262208x3840 bf16),
        # a decoder layer is ~0.5GB. Measured end-to-end encode peak 3.96GB and 0.02GB
        # retained afterwards, versus the whole 28.6GB backbone (38.85GB forward peak) that
        # a bulk move needs. That is the only way this bf16 encoder fits a 40GB card and
        # still leaves the DiT room to denoise. LTX2TextEncoder sets it from
        # gemma_cpu_offload; the env var keeps the standalone/offline path working.
        self.pinned_layerwise = os.environ.get("LTX_GEMMA_LAYERWISE_GPU", "") == "1"
        self._embed_scale_checked = False

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (attention_mask.to(torch.int64) - 1).to(dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(dtype).max

    @staticmethod
    def _cpu_home_for(t: torch.Tensor) -> torch.Tensor:
        """The CPU tensor a streamed weight returns to: itself when already on CPU, otherwise
        a copy pulled down from the GPU. Page-locked best-effort so later round trips run at
        ~15-20 GB/s instead of pageable ~2 GB/s; a pin failure just means that one tensor
        transfers slower."""
        home = t if t.device.type == "cpu" else t.to("cpu")
        if home.is_floating_point() and not home.is_pinned():
            try:
                home = home.pin_memory()
            except (RuntimeError, MemoryError):
                pass
        return home

    def _stream_tensor_to_gpu(self, t: torch.Tensor, device: torch.device) -> None:
        """Give one weight a CPU home (once) and put it on the GPU for the coming forward.

        A tensor found already on GPU is *adopted* rather than skipped: it gets a CPU home
        pulled down from the device, so the matching _stream_to_cpu can actually offload it.
        Skipping instead would make streaming a silent no-op on a GPU-resident encoder and
        leave the whole 28.6GB backbone on the card — the very thing this path exists to
        avoid. The config-driven path never hits this (gemma_cpu_offload sets both
        device=cpu at load and pinned_layerwise, so residency and mode cannot disagree), but
        the LTX_GEMMA_LAYERWISE_GPU standalone/offline path can, and the pre-streaming code
        coped with it by unconditionally moving each submodule back to CPU."""
        if getattr(t, "_ltx_cpu_home", None) is None:
            t._ltx_cpu_home = self._cpu_home_for(t.data)
        if t.data.device.type == "cpu":
            t.data = t._ltx_cpu_home.to(device, non_blocking=True)

    def _stream_to_gpu(self, module: torch.nn.Module, device: torch.device) -> None:
        """Move one module's weights onto the GPU for its forward pass, stashing each CPU home
        on the tensor so _stream_to_cpu can restore it."""
        for p in module.parameters(recurse=True):
            self._stream_tensor_to_gpu(p, device)
        for b in module.buffers(recurse=True):
            self._stream_tensor_to_gpu(b, device)

    def _stream_to_cpu(self, module: torch.nn.Module) -> None:
        """Restore one module's weights to their CPU homes and drop the GPU copies. Weights
        are read-only during inference, so we discard the GPU tensor instead of copying it
        back (no D2H) — GPU peak stays at a single streamed module."""
        for p in module.parameters(recurse=True):
            home = getattr(p, "_ltx_cpu_home", None)
            if home is not None:
                p.data = home
                p._ltx_cpu_home = None
        for b in module.buffers(recurse=True):
            home = getattr(b, "_ltx_cpu_home", None)
            if home is not None:
                b.data = home
                b._ltx_cpu_home = None

    def restore_offloaded(self) -> None:
        """Safety net for the pinned-layerwise path: walk the whole encoder and pull any
        tensor still stranded on GPU back to its CPU home. After a clean encode this is a
        no-op (every module already self-restored); after a failure mid-stream it reclaims the
        one module that was in flight (<=2.0GB, embed_tokens being the largest) so a persistent
        worker never poisons itself across requests. Called from encode_text's finally."""
        for module in (self.model, self.feature_extractor, self.embeddings_processor):
            if module is not None:
                self._stream_to_cpu(module)
        # Return the just-dropped GPU blocks to the driver, not just the caching pool. On a
        # mid-stream OOM the reserved arena is left fragmented, and this card is tight enough
        # that the DiT denoise of the very next request can spuriously OOM otherwise. Cheap
        # on the clean path too (only ~1 layer was ever reserved).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _assert_embedding_scaling_once(self, text_model: torch.nn.Module) -> None:
        """Guard the one assumption this hand-written forward makes about transformers.

        Gemma3 applies the sqrt(hidden_size) embedding normalizer (61.97 for the 3840-dim
        12b text model) *inside* Gemma3TextScaledWordEmbedding.forward, not in
        Gemma3TextModel.forward — so calling embed_tokens as a module, exactly as the stock
        forward does, already scales. Verified numerically against the stock forward on
        transformers 4.57.1: amplitude ratio 1.0000 and 0.25% relative diff (bf16 GPU-vs-CPU
        noise), versus the ~62x a missing scale would produce.

        Gemma1/2 put that normalizer in the model forward instead, `transformers` is
        unpinned in requirements, and this layerwise path is now the production default for
        gemma_cpu_offload. If an upgrade moves the scaling back out of the embedding, this
        path would silently emit ~62x-off conditioning — no exception, just quietly wrong
        audio. Fail loudly at the first encode instead, so a smoke test catches it rather
        than customers. Checked once per encoder; costs nothing after that.

        Scope: this catches the realistic regression (scaling leaves the embedding module).
        It cannot detect a refactor that keeps the override *and* adds a second scale in the
        model forward — re-run the A/B parity check when bumping transformers."""
        if self._embed_scale_checked:
            return
        embed = text_model.embed_tokens
        scales_internally = type(embed).forward is not torch.nn.Embedding.forward and hasattr(embed, "embed_scale")
        if not scales_internally:
            import transformers

            raise RuntimeError(
                f"Gemma embedding-scaling invariant broken: {type(embed).__name__} does not apply the "
                f"sqrt(hidden_size) normalizer inside its own forward (embed_scale missing or forward not "
                f"overridden), so _run_text_model_layerwise_on_gpu would drop it and produce silently wrong "
                f"conditioning. transformers=={transformers.__version__} likely moved the normalizer into the "
                f"model forward. Re-verify layerwise-vs-stock parity and patch the layerwise forward before use."
            )
        self._embed_scale_checked = True

    @torch.no_grad()
    def _run_text_model_layerwise_on_gpu(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> BaseModelOutputWithPast:
        """Run Gemma text layers on GPU one at a time while keeping weights on CPU."""
        text_model = self.model.model.language_model
        device = torch.device(AI_DEVICE)
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        self._assert_embedding_scaling_once(text_model)

        # Call embed_tokens as a MODULE (never a raw F.embedding / weight lookup): Gemma3's
        # sqrt(hidden_size) normalizer lives inside Gemma3TextScaledWordEmbedding.forward, so
        # bypassing the module silently drops a ~62x scale. See _assert_embedding_scaling_once.
        self._stream_to_gpu(text_model.embed_tokens, device)
        hidden_states = text_model.embed_tokens(input_ids)
        self._stream_to_cpu(text_model.embed_tokens)

        cache_position = torch.arange(0, hidden_states.shape[1], device=device)
        position_ids = cache_position.unsqueeze(0)
        mask_kwargs = {
            "config": text_model.config,
            "input_embeds": hidden_states,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": None,
            "position_ids": position_ids,
        }

        import transformers.models.gemma3.modeling_gemma3 as modeling_gemma3

        if text_model.config.use_bidirectional_attention:
            mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool, device=device)
            sliding_mask_kwargs = dict(mask_kwargs)
            sliding_mask_kwargs["or_mask_function"] = modeling_gemma3._bidirectional_window_overlay(text_model.config.sliding_window)
        else:
            sliding_mask_kwargs = dict(mask_kwargs)

        causal_mask_mapping = {
            "full_attention": modeling_gemma3.create_causal_mask(**mask_kwargs),
            "sliding_attention": modeling_gemma3.create_sliding_window_causal_mask(**sliding_mask_kwargs),
        }

        self._stream_to_gpu(text_model.rotary_emb, device)
        self._stream_to_gpu(text_model.rotary_emb_local, device)
        position_embeddings_global = text_model.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = text_model.rotary_emb_local(hidden_states, position_ids)
        self._stream_to_cpu(text_model.rotary_emb)
        self._stream_to_cpu(text_model.rotary_emb_local)

        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in text_model.layers[: text_model.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            self._stream_to_gpu(decoder_layer, device)
            layer_outputs = decoder_layer(
                hidden_states,
                position_embeddings_global=position_embeddings_global,
                position_embeddings_local=position_embeddings_local,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
            )
            hidden_states = layer_outputs[0]
            self._stream_to_cpu(decoder_layer)
            torch.cuda.empty_cache()

        self._stream_to_gpu(text_model.norm, device)
        hidden_states = text_model.norm(hidden_states)
        self._stream_to_cpu(text_model.norm)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    @torch.no_grad()
    def precompute(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Blocks 1+2: Gemma model -> feature extraction.
        Used by process_captions.py for offline precomputation.
        Returns (video_features, audio_features | None, attention_mask).

        no_grad is load-bearing, not hygiene: the feature extractor's own parameters have
        requires_grad=True, so without it the returned contexts carry a grad_fn and pin every
        saved intermediate — measured 6.39GB (two 49x3840-wide hidden-state concats) versus
        0.02GB with it. Nothing upstream wraps this: run_text_encoder only has a profiling
        decorator, the v2a call site's no_grad covers just the VAE encode, and run_pipeline has
        none. The runner then holds text_encoder_output for the whole denoise, so that 6.37GB
        would sit on the card for the entire request — 16% of a 40GB A100, spent on a backward
        pass this inference engine never runs.
        """
        # Block 1: Run Gemma
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        # Gemma 4 unified is wrapped in a conditional-generation head.  LTX only
        # consumes hidden states, so call its inner model and avoid materializing
        # the very large vocabulary logits.  Keep the established Gemma 3 path
        # unchanged.
        model_type = getattr(getattr(self.model, "config", None), "model_type", "")
        backbone = self.model.model if model_type == "gemma4_unified" else self.model
        # 逐层流式 offload 只对 Gemma 3 成立:_run_text_model_layerwise_on_gpu 直接走
        # self.model.model.language_model,并 import gemma3 的 mask 构造函数,
        # gemma4_unified 的结构不同,硬套会拿错模块。gemma4 走上游原路,并告警一次,
        # 免得「配了 offload 却没生效」在 40G 卡上变成一次莫名其妙的 OOM。
        if self.pinned_layerwise and model_type == "gemma4_unified":
            if not getattr(self, "_layerwise_gemma4_warned", False):
                logger.warning("pinned_layerwise 对 gemma4_unified 不生效(逐层路径是 Gemma 3 专用),本次按整模型前向运行")
                self._layerwise_gemma4_warned = True
        if self.pinned_layerwise and model_type != "gemma4_unified":
            outputs = self._run_text_model_layerwise_on_gpu(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            attention_mask = attention_mask.to(outputs.last_hidden_state.device)
        else:
            outputs = backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Block 2: Feature extraction
        if self.pinned_layerwise:
            self._stream_to_gpu(self.feature_extractor, torch.device(AI_DEVICE))
            try:
                video_feats, audio_feats = self.feature_extractor(outputs.hidden_states, attention_mask, padding_side)
            finally:
                self._stream_to_cpu(self.feature_extractor)
        else:
            video_feats, audio_feats = self.feature_extractor(outputs.hidden_states, attention_mask, padding_side)
        return video_feats, audio_feats, attention_mask

    @torch.no_grad()
    def forward(self, text: str, padding_side: str = "left") -> GemmaEncoderOutput:
        """Full pipeline: precompute -> embeddings processor.

        Decorated as well as precompute so the connector stage is covered for callers that
        enter here — see the memory rationale on precompute."""
        video_feats, audio_feats, attention_mask = self.precompute(text, padding_side)
        additive_mask = self._convert_to_additive_mask(attention_mask, video_feats.dtype)
        if self.pinned_layerwise:
            self._stream_to_gpu(self.embeddings_processor, torch.device(AI_DEVICE))
            try:
                video_enc, audio_enc, binary_mask = self.embeddings_processor.create_embeddings(video_feats, audio_feats, additive_mask)
            finally:
                self._stream_to_cpu(self.embeddings_processor)
        else:
            video_enc, audio_enc, binary_mask = self.embeddings_processor.create_embeddings(video_feats, audio_feats, additive_mask)
        return GemmaEncoderOutput(video_enc, audio_enc, binary_mask)

    # --- Prompt enhancement methods ---

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        if getattr(getattr(self.model, "config", None), "model_type", "") == "gemma4_unified":
            raise ValueError("The LTX-2.5 Gemma 4 unified encode checkpoint does not support prompt enhancement.")
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")


# --- Standalone utility functions ---


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)
        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    tokenizer_root = str(find_matching_file(gemma_root, "tokenizer.model").parent)
    processor_root = str(find_matching_file(gemma_root, "preprocessor_config.json").parent)

    def load_tokenizer(module: GemmaTextEncoder) -> GemmaTextEncoder:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_root, 1024)
        return module

    def load_processor(module: GemmaTextEncoder) -> GemmaTextEncoder:
        image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        module.processor = Gemma3Processor(image_processor=image_processor, tokenizer=module.tokenizer.tokenizer)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.processor is None,
        mutator=load_processor,
    )
    return (tokenizer_load_ops, processor_load_ops)


def encode_text(text_encoder: GemmaTextEncoder, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Encode a list of prompts using the provided Gemma text encoder.
    Returns:
        List of tuples, each containing (v_context, a_context) tensors for each prompt.
    """
    result = []
    for prompt in prompts:
        v_context, a_context, _ = text_encoder(prompt)
        result.append((v_context, a_context))
    return result
