import torch
import torch.distributed as dist
from torch.nn import functional as F

from lightx2v.utils.envs import *
from lightx2v_platform.base.global_var import AI_DEVICE

from .module_io import GridOutput, WanPreInferModuleOutput
from .utils import guidance_scale_embedding, sinusoidal_embedding_1d


class WanPreInfer:
    def __init__(self, config):
        assert (config["dim"] % config["num_heads"]) == 0 and (config["dim"] // config["num_heads"]) % 2 == 0
        self.config = config
        self.rope = None
        self.clean_cuda_cache = config.get("clean_cuda_cache", False)
        self.task = config["task"]
        self.freq_dim = config["freq_dim"]
        self.dim = config["dim"]
        self.enable_dynamic_cfg = config.get("enable_dynamic_cfg", False)
        self.cfg_scale = config.get("cfg_scale", 4.0)
        self.infer_dtype = GET_DTYPE()
        self.sensitive_layer_dtype = GET_SENSITIVE_DTYPE()

        if self.config["seq_parallel"]:
            self.seq_p_group = self.config.get("device_mesh").get_group(mesh_dim="seq_p")
        else:
            self.seq_p_group = None

        self.cos_sin = None
        self.rope_positions = None
        self.grid_sizes = (0, 0, 0)  # (t, h, w)
        self.head_size = self.config["dim"] // self.config["num_heads"]
        self.freqs = torch.cat(
            [
                self.rope_params(1024, self.head_size - 4 * (self.head_size // 6)),
                self.rope_params(1024, 2 * (self.head_size // 6)),
                self.rope_params(1024, 2 * (self.head_size // 6)),
            ],
            dim=1,
        ).to(torch.device(AI_DEVICE))
        self.rope_t_dim = self.head_size // 2 - 2 * (self.head_size // 6)

    def rope_params(self, max_seq_len, dim, theta=10000):
        assert dim % 2 == 0
        freqs = torch.outer(
            torch.arange(max_seq_len),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float32).div(dim)),
        )
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs

    def src_id_phase(self, source_id, theta=10000.0):
        """Per-source rotary phase for source-id RoPE (bernini transformer_wan.py:282).

        Mirrors `get_1d_rotary_pos_embed(head_dim, pos=[source_id], theta,
        use_real=False)`: complex phase of shape [1, head_dim // 2], broadcast over
        every token of that source. Supports fractional source_id (the linspace
        interpolation for n > max_trained_src_id). Returns a complex tensor.
        """
        dim = self.head_size  # full attention head dim (freqs computed over dim//2)
        pos = torch.tensor([float(source_id)], dtype=torch.float64, device=torch.device(AI_DEVICE))
        freqs = torch.outer(pos, 1.0 / torch.pow(theta, torch.arange(0, dim, 2, device=pos.device).to(torch.float64).div(dim)))
        return torch.polar(torch.ones_like(freqs), freqs)  # [1, dim // 2] complex

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def set_rope(self, rope):
        self.rope = rope
        self.cos_sin = None
        self.rope_positions = None
        self.grid_sizes = (0, 0, 0)

    def prepare_rope_cache(self, freqs):
        if self.rope is None:
            raise RuntimeError("RoPE must be set before preparing the Wan frequency cache.")
        freqs = self.rope.prepare_freqs(freqs, rotary_dim=self.head_size)
        self.rope_positions = self.rope.prepare_positions(freqs)
        return freqs

    def prepare_cos_sin(self, grid_sizes, freqs):
        c = self.head_size // 2
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = grid_sizes
        seq_len = f * h * w
        cos_sin = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        cos_sin = cos_sin.reshape(seq_len, 1, -1)
        if self.seq_p_group is not None:
            world_size = dist.get_world_size(self.seq_p_group)
            cur_rank = dist.get_rank(self.seq_p_group)
            seqlen = cos_sin.shape[0]
            padding_size = (world_size - (seqlen % world_size)) % world_size
            if padding_size > 0:
                cos_sin = F.pad(cos_sin, (0, 0, 0, 0, 0, padding_size))
            cos_sin = torch.chunk(cos_sin, world_size, dim=0)[cur_rank]
        return cos_sin

    def _prepare_cos_sin_full_complex(self, grid_sizes):
        """Full (unsharded) complex cos_sin for one latent grid, torch rope layout.

        Same math as `prepare_cos_sin` (non-flashinfer branch) but WITHOUT the
        seq-parallel chunk — v2v builds the whole context+target sequence first,
        then shards once (bernini: concat before split, transformer_wan.py:458).
        Returns complex [seq_len, 1, head_dim // 2].
        """
        freqs = self.freqs.clone()
        c = self.head_size // 2
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        f, h, w = grid_sizes
        seq_len = f * h * w
        cos_sin = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        return cos_sin.reshape(seq_len, 1, -1)  # complex [seq_len, 1, head_dim // 2]

    def _shard_cos_sin(self, cos_sin):
        """Chunk a full complex cos_sin over the seq-parallel group (matches
        `_seq_parallel_pre_process` padding of x). No-op for seq_p=1."""
        if self.seq_p_group is None:
            return cos_sin
        world_size = dist.get_world_size(self.seq_p_group)
        cur_rank = dist.get_rank(self.seq_p_group)
        seqlen = cos_sin.shape[0]
        padding_size = (world_size - (seqlen % world_size)) % world_size
        if padding_size > 0:
            cos_sin = F.pad(cos_sin, (0, 0, 0, 0, 0, padding_size))
        return torch.chunk(cos_sin, world_size, dim=0)[cur_rank]

    @torch.no_grad()
    def infer(self, weights, inputs, kv_start=0, kv_end=0):
        # bernini v2v: in-context video editing. Concat source (context) tokens
        # BEFORE the noisy target along the sequence dim (target LAST), each source
        # patch-embedded separately with source-id RoPE. See wan_diffusion.py:479.
        if self.task == "v2v" and inputs.get("v2v_context") is not None:
            return self._infer_v2v(weights, inputs)
        x = self.scheduler.latents
        t = self.scheduler.timestep_input

        if self.config["model_cls"] == "wan2.1_mean_flow_distill":
            t_r = self.scheduler.timestep_input_r

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        if self.task in ["i2v", "flf2v", "animate", "s2v", "rs2v"]:
            if self.config.get("use_image_encoder", True):
                clip_fea = inputs["image_encoder_output"]["clip_encoder_out"]

            if self.config.get("changing_resolution", False):
                image_encoder = inputs["image_encoder_output"]["vae_encoder_out"][self.scheduler.changing_resolution_index]
            else:
                image_encoder = inputs["image_encoder_output"]["vae_encoder_out"]

            if image_encoder is not None:
                x = torch.cat([x, image_encoder], dim=0)

        # embeddings
        x = weights.patch_embedding.apply(x.unsqueeze(0))

        if hasattr(self, "after_patch_embedding"):
            x, motion_vec = self.after_patch_embedding(weights, x, inputs["image_encoder_output"]["pose_latents"], inputs["image_encoder_output"]["face_pixel_values"])
        else:
            motion_vec = None

        grid_sizes_t, grid_sizes_h, grid_sizes_w = x.shape[2:]
        x = x.flatten(2).transpose(1, 2).squeeze(0).contiguous()
        # seq_lens = torch.tensor(x.size(1), dtype=torch.int32).unsqueeze(0)

        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        if self.enable_dynamic_cfg:
            s = torch.tensor([self.cfg_scale], dtype=torch.float32, device=x.device)
            cfg_embed = guidance_scale_embedding(s, embedding_dim=256, cfg_range=(1.0, 6.0), target_range=1000.0, dtype=torch.float32).type_as(x)
            cfg_embed = weights.cfg_cond_proj_1.apply(cfg_embed)
            cfg_embed = torch.nn.functional.silu(cfg_embed)
            cfg_embed = weights.cfg_cond_proj_2.apply(cfg_embed)
            embed = embed + cfg_embed
        if self.sensitive_layer_dtype != self.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)

        if self.config["model_cls"] == "wan2.1_mean_flow_distill":
            embed_r = sinusoidal_embedding_1d(self.freq_dim, t_r.flatten())
            if self.sensitive_layer_dtype != self.infer_dtype:
                embed_r = weights.time_embedding_r_0.apply(embed_r.to(self.sensitive_layer_dtype))
            else:
                embed_r = weights.time_embedding_r_0.apply(embed_r)
            embed_r = torch.nn.functional.silu(embed_r)
            embed_r = weights.time_embedding_r_2.apply(embed_r)
            embed0_r = torch.nn.functional.silu(embed_r)
            embed0 = embed0 + embed0_r

        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        # text embeddings
        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)
        if self.clean_cuda_cache:
            del out
            torch.cuda.empty_cache()

        if self.task in ["i2v", "flf2v", "animate"] and self.config.get("use_image_encoder", True):
            if self.task == "flf2v":
                _, n, d = clip_fea.shape
                clip_fea = clip_fea.view(2 * n, d)
                clip_fea = clip_fea + weights.emb_pos.tensor.squeeze()
            context_clip = weights.proj_0.apply(clip_fea)
            if self.clean_cuda_cache:
                del clip_fea
                torch.cuda.empty_cache()
            context_clip = weights.proj_1.apply(context_clip)
            context_clip = torch.nn.functional.gelu(context_clip, approximate="none")
            if self.clean_cuda_cache:
                torch.cuda.empty_cache()
            context_clip = weights.proj_3.apply(context_clip)
            context_clip = weights.proj_4.apply(context_clip)
            context = torch.concat([context_clip, context], dim=0)

        if self.clean_cuda_cache:
            if self.config.get("use_image_encoder", True):
                del context_clip
            torch.cuda.empty_cache()

        grid_sizes = GridOutput(tensor=torch.tensor([[grid_sizes_t, grid_sizes_h, grid_sizes_w]], dtype=torch.int32, device=x.device), tuple=(grid_sizes_t, grid_sizes_h, grid_sizes_w))

        if self.cos_sin is None or self.grid_sizes != grid_sizes.tuple:
            freqs = self.freqs.clone()  # self.freqs init param can not be changed
            self.grid_sizes = grid_sizes.tuple
            self.cos_sin = self.prepare_rope_cache(self.prepare_cos_sin(grid_sizes.tuple, freqs))

        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=x,
            embed0=embed0.squeeze(0),
            context=context,
            cos_sin=self.cos_sin,
            rope_positions=self.rope_positions,
            adapter_args={"motion_vec": motion_vec},
        )

    @torch.no_grad()
    def _infer_v2v(self, weights, inputs):
        """Bernini-R v2v pre-infer: build [context_sources..., noisy_target] token
        sequence (target LAST) with per-source-id RoPE, plus a target-token mask.

        Layout mirrors upstream `_assemble` (wan_diffusion.py:479): each context
        source is patch-embedded separately (patch_vae_latent, transformer_wan.py:446)
        with source_id 1..n; the noisy target keeps source_id 0. cos_sin for a source
        is the base spatial rotary phase multiplied by that source's `freqs_visual_id`
        (transformer_wan.py:289). Tokens + cos_sin concat along the sequence dim
        BEFORE any seq-parallel split (transformer_wan.py:458).
        """
        assert self.config.get("rope_type", "flashinfer") != "flashinfer", (
            "v2v requires rope_type='torch' (complex cos_sin); flashinfer real/imag "
            "layout for source-id RoPE is not implemented yet."
        )

        v2v = inputs["v2v_context"]
        src_latents = v2v["src_latents"]  # list of [C=16, T_lat, H_lat, W_lat]
        src_ids = v2v["src_ids"]          # list[float], context ids 1..n

        noisy = self.scheduler.latents  # [C, T_lat, H_lat, W_lat] (normalized target)
        t = self.scheduler.timestep_input

        if self.scheduler.infer_condition:
            context = inputs["text_encoder_output"]["context"]
        else:
            context = inputs["text_encoder_output"]["context_null"]

        # --- Patch-embed the noisy target (source_id 0 → plain spatial RoPE). ---
        tgt_x = weights.patch_embedding.apply(noisy.unsqueeze(0))  # [1, C, T, H, W]
        grid_t, grid_h, grid_w = tgt_x.shape[2:]
        tgt_tokens = tgt_x.flatten(2).transpose(1, 2).contiguous().squeeze(0)  # [THW, dim]
        base_cos_sin = self._prepare_cos_sin_full_complex((grid_t, grid_h, grid_w))  # [THW,1,d/2] complex

        # --- Patch-embed each context source with its source-id RoPE. ---
        # bernini wan_diffusion.py:430 — context sources concat left-to-right, target last.
        ctx_tokens_list = []
        ctx_cos_sin_list = []
        for z, sid in zip(src_latents, src_ids):
            # NOTE: sources share the target grid (same T_lat/H_lat/W_lat). If a
            # source had a different grid we'd recompute base_cos_sin per source.
            s_x = weights.patch_embedding.apply(z.unsqueeze(0))
            s_tokens = s_x.flatten(2).transpose(1, 2).contiguous().squeeze(0)
            # Cast to the cache dtype: src_id_phase computes in float64 (accuracy)
            # → complex128, but rope_params builds the base cache as complex64;
            # multiplying without the cast would promote the whole (n+1)*THW RoPE
            # cache to complex128 (2x memory + fp64 rope math in every block).
            phase = self.src_id_phase(sid).to(device=base_cos_sin.device, dtype=base_cos_sin.dtype)  # [1, d/2] complex64
            # bernini transformer_wan.py:289: freqs = freqs * freqs_visual_id
            s_cos_sin = base_cos_sin * phase.view(1, 1, -1)
            ctx_tokens_list.append(s_tokens)
            ctx_cos_sin_list.append(s_cos_sin)

        # --- Assemble: context sources first, noisy target LAST. ---
        all_tokens = torch.cat(ctx_tokens_list + [tgt_tokens], dim=0)  # [(n+1)*THW, dim]
        full_cos_sin = torch.cat(ctx_cos_sin_list + [base_cos_sin], dim=0)

        # Target-token mask over the FULL sequence (context False, target True).
        ctx_len = sum(tok.shape[0] for tok in ctx_tokens_list)
        target_mask = torch.zeros(all_tokens.shape[0], dtype=torch.bool, device=all_tokens.device)
        target_mask[ctx_len:] = True

        # --- Time / text embeddings (identical to the base t2v path). ---
        embed = sinusoidal_embedding_1d(self.freq_dim, t.flatten())
        if self.sensitive_layer_dtype != self.infer_dtype:
            embed = weights.time_embedding_0.apply(embed.to(self.sensitive_layer_dtype))
        else:
            embed = weights.time_embedding_0.apply(embed)
        embed = torch.nn.functional.silu(embed)
        embed = weights.time_embedding_2.apply(embed)
        embed0 = torch.nn.functional.silu(embed)
        embed0 = weights.time_projection_1.apply(embed0).unflatten(1, (6, self.dim))

        if self.sensitive_layer_dtype != self.infer_dtype:
            out = weights.text_embedding_0.apply(context.squeeze(0).to(self.sensitive_layer_dtype))
        else:
            out = weights.text_embedding_0.apply(context.squeeze(0))
        out = torch.nn.functional.gelu(out, approximate="tanh")
        context = weights.text_embedding_2.apply(out)

        # grid_sizes stays the TARGET grid: post_infer unpatchifies only the target
        # tokens (sliced out after the blocks), so the grid must match target THW.
        grid_sizes = GridOutput(
            tensor=torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.int32, device=all_tokens.device),
            tuple=(grid_t, grid_h, grid_w),
        )

        # --- Seq-parallel: shard the FULL (context+target) cos_sin like x is sharded
        #     in model._seq_parallel_pre_process (concat-before-split, then gather
        #     reconstructs the full seq before the target slice). ---
        cos_sin = self._shard_cos_sin(full_cos_sin)

        return WanPreInferModuleOutput(
            embed=embed,
            grid_sizes=grid_sizes,
            x=all_tokens,
            embed0=embed0.squeeze(0),
            context=context,
            cos_sin=cos_sin,
            adapter_args={"motion_vec": None},
            v2v_target_mask=target_mask,
        )
