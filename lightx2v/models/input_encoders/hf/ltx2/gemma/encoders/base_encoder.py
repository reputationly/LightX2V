import functools
import os
from pathlib import Path
from typing import NamedTuple

import torch
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor
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
        model: Gemma3ForConditionalGeneration | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.feature_extractor = feature_extractor.to(dtype=dtype)
        self.embeddings_processor = embeddings_processor.to(dtype=dtype)

    def _convert_to_additive_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return (attention_mask.to(torch.int64) - 1).to(dtype).reshape((attention_mask.shape[0], 1, -1, attention_mask.shape[-1])) * torch.finfo(dtype).max

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

        embed_tokens = text_model.embed_tokens.to(device)
        hidden_states = embed_tokens(input_ids)
        embed_tokens.to("cpu")

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

        text_model.rotary_emb.to(device)
        text_model.rotary_emb_local.to(device)
        position_embeddings_global = text_model.rotary_emb(hidden_states, position_ids)
        position_embeddings_local = text_model.rotary_emb_local(hidden_states, position_ids)
        text_model.rotary_emb.to("cpu")
        text_model.rotary_emb_local.to("cpu")

        all_hidden_states = () if output_hidden_states else None
        for decoder_layer in text_model.layers[: text_model.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            decoder_layer.to(device)
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
            decoder_layer.to("cpu")
            torch.cuda.empty_cache()

        text_model.norm.to(device)
        hidden_states = text_model.norm(hidden_states)
        text_model.norm.to("cpu")

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def precompute(self, text: str, padding_side: str = "left") -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Blocks 1+2: Gemma model -> feature extraction.
        Used by process_captions.py for offline precomputation.
        Returns (video_features, audio_features | None, attention_mask).
        """
        # Block 1: Run Gemma
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        if os.environ.get("LTX_GEMMA_LAYERWISE_GPU", "") == "1":
            outputs = self._run_text_model_layerwise_on_gpu(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            attention_mask = attention_mask.to(outputs.last_hidden_state.device)
        else:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        # Block 2: Feature extraction
        if os.environ.get("LTX_GEMMA_LAYERWISE_GPU", "") == "1":
            self.feature_extractor = self.feature_extractor.to(AI_DEVICE)
        video_feats, audio_feats = self.feature_extractor(outputs.hidden_states, attention_mask, padding_side)
        return video_feats, audio_feats, attention_mask

    def forward(self, text: str, padding_side: str = "left") -> GemmaEncoderOutput:
        """Full pipeline: precompute -> embeddings processor."""
        video_feats, audio_feats, attention_mask = self.precompute(text, padding_side)
        additive_mask = self._convert_to_additive_mask(attention_mask, video_feats.dtype)
        if os.environ.get("LTX_GEMMA_LAYERWISE_GPU", "") == "1":
            self.embeddings_processor = self.embeddings_processor.to(AI_DEVICE)
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
