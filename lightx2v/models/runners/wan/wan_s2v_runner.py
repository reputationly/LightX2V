# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import os
import random
import subprocess
import sys

import numpy as np
import torch
from PIL import Image
from loguru import logger
from torchvision import transforms

from lightx2v.models.input_encoders.hf.wan.s2v.audio_encoder import AudioEncoder
from lightx2v.models.networks.wan.s2v_model import WanS2VModel
from lightx2v.models.networks.wan.s2v_utils import get_size_less_than_area
from lightx2v.models.runners.wan.wan_runner import MultiModelStruct, WanRunner
from lightx2v.models.schedulers.wan.s2v.s2v_scheduler import WanS2VScheduler
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process, save_to_video
from lightx2v_platform.base.global_var import AI_DEVICE

try:
    from decord import VideoReader
except ImportError:
    VideoReader = None


def merge_video_audio(video_path: str, audio_path: str):
    tmp_path = video_path + ".tmp.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        tmp_path,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    os.replace(tmp_path, video_path)


@RUNNER_REGISTER("wan2.2_s2v")
class WanS2VRunner(WanRunner):
    def __init__(self, config):
        self.vae_name = "Wan2.1_VAE.pth"
        super().__init__(config)
        assert self.config["task"] == "s2v"
        self.param_dtype = GET_DTYPE()
        self.fps = self.config["target_fps"]
        self.audio_sample_m = 0

    def init_scheduler(self):
        self.scheduler = WanS2VScheduler(self.config)

    def init_modules(self):
        logger.info("Initializing WanS2V runner modules...")
        self.load_model()
        self.model.set_scheduler(self.scheduler)
        self.run_input_encoder = self._run_input_encoder_local_s2v
        self.config.lock()

    @ProfilingContext4DebugL2("Load models")
    def load_model(self):
        self.model = self.load_transformer()
        self.text_encoders = self.load_text_encoder()
        self.vae_encoder, self.vae_decoder = self.load_vae()
        self.audio_encoder = self.load_audio_encoder()

    def load_transformer(self):
        return WanS2VModel(self.config["model_path"], self.config, self.init_device)

    def load_image_encoder(self):
        return None

    def load_audio_encoder(self):
        wav2vec_dir = os.path.join(
            self.config["model_path"],
            self.config.get("wav2vec_subdir", "wav2vec2-large-xlsr-53-english"),
        )
        return AudioEncoder(model_id=wav2vec_dir)

    def _vae_encode(self, video):
        video = video.to(device=AI_DEVICE, dtype=self.vae_encoder.dtype)
        with torch.amp.autocast(str(AI_DEVICE), dtype=self.vae_encoder.dtype):
            return self.vae_encoder.encode(video)

    def _vae_encode_pose_cond(self, video):
        """Align Wan2.2 load_pose_cond: bf16 pixels in, fp32 VAE autocast, float latent out."""
        video = video.to(device=AI_DEVICE, dtype=self.param_dtype)
        with torch.amp.autocast(str(AI_DEVICE), dtype=self.vae_encoder.dtype):
            latents = [self.vae_encoder.model.encode(u.unsqueeze(0), self.vae_encoder.scale).float().squeeze(0) for u in video]
        return torch.stack(latents)

    def build_cond_latents(self, height, width):
        infer_frames = self.config["infer_frames"]
        cond = -torch.ones([1, 3, infer_frames, height, width], device=AI_DEVICE, dtype=self.vae_encoder.dtype)
        cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
        if self.config.get("cpu_offload", False):
            self.vae_encoder.to_cuda()
        cond_lat = self._vae_encode(cond).unsqueeze(0)[:, :, 1:]
        if self.config.get("cpu_offload", False):
            self.vae_encoder.to_cpu()
        return cond_lat * 0

    @staticmethod
    def read_pose_video_frames(video_path, n_frames, target_fps=16, reverse=False):
        if VideoReader is None:
            raise ImportError("decord is required for src_pose_path (pip install decord)")
        vr = VideoReader(video_path)
        original_fps = vr.get_avg_fps()
        total_frames = len(vr)
        interval = max(1, round(original_fps / target_fps))
        required_span = (n_frames - 1) * interval
        start_frame = 0 if reverse else max(0, total_frames - required_span - 1)
        sampled_indices = []
        for i in range(n_frames):
            idx = start_frame + i * interval
            if idx >= total_frames:
                break
            sampled_indices.append(idx)
        return vr.get_batch(sampled_indices).asnumpy()

    def load_pose_cond(self, pose_video, num_repeat, infer_frames, height, width):
        """Align with Wan2.2 WanS2V.load_pose_cond: VAE-encoded pose latents per clip."""
        offload = self.config.get("cpu_offload", False)
        fps = self.config["target_fps"]
        resize_op = transforms.Resize(min(height, width))
        crop_op = transforms.CenterCrop((height, width))

        if pose_video:
            pose_seq = self.read_pose_video_frames(
                pose_video,
                n_frames=infer_frames * num_repeat,
                target_fps=fps,
                reverse=True,
            )
            cond_tensor = torch.from_numpy(pose_seq).permute(0, 3, 1, 2) / 255.0 * 2 - 1.0
            cond_tensor = crop_op(resize_op(cond_tensor)).permute(1, 0, 2, 3).unsqueeze(0)

            padding_frame_num = num_repeat * infer_frames - cond_tensor.shape[2]
            if padding_frame_num > 0:
                pad = -torch.ones([1, 3, padding_frame_num, height, width], dtype=cond_tensor.dtype)
                cond_tensor = torch.cat([cond_tensor, pad], dim=2)
            cond_tensors = torch.chunk(cond_tensor, num_repeat, dim=2)
        else:
            cond_tensors = [-torch.ones([1, 3, infer_frames, height, width])]

        cond_list = []
        for cond in cond_tensors:
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            if offload:
                self.vae_encoder.to_cuda()
            # Match Wan2.2: fp32 latent on CPU for mem save; cast to param_dtype per clip.
            cond_lat = self._vae_encode_pose_cond(cond)[:, :, 1:].cpu()
            if offload:
                self.vae_encoder.to_cpu()
            cond_list.append(cond_lat)
        return cond_list

    def encode_audio(self, audio_path, infer_frames):
        z = self.audio_encoder.extract_audio_feat(audio_path, return_all_layers=True)
        audio_embed_bucket, num_repeat = self.audio_encoder.get_audio_embed_bucket_fps(z, fps=self.fps, batch_frames=infer_frames, m=self.audio_sample_m)
        audio_embed_bucket = audio_embed_bucket.to(AI_DEVICE, self.param_dtype)
        audio_embed_bucket = audio_embed_bucket.unsqueeze(0)
        if len(audio_embed_bucket.shape) == 3:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 1)
        elif len(audio_embed_bucket.shape) == 4:
            audio_embed_bucket = audio_embed_bucket.permute(0, 2, 3, 1)
        return audio_embed_bucket, num_repeat

    @ProfilingContext4DebugL2("Run Encoders")
    def _run_input_encoder_local_s2v(self):
        ref_image = np.array(Image.open(self.input_info.image_path).convert("RGB"))
        height, width = get_size_less_than_area(*ref_image.shape[:2], self.config["max_area"])

        resize_op = transforms.Resize(min(height, width))
        crop_op = transforms.CenterCrop((height, width))
        tensor_trans = transforms.ToTensor()

        with ProfilingContext4DebugL1("Encode audio"):
            audio_emb, audio_num_repeat = self.encode_audio(self.input_info.audio_path, infer_frames=self.config["infer_frames"])

        cfg_repeat = self.config.get("num_repeat")
        if cfg_repeat is None or cfg_repeat > audio_num_repeat:
            num_repeat = audio_num_repeat
        else:
            num_repeat = int(cfg_repeat)
            if num_repeat < audio_num_repeat:
                # The muxed track is cut with -shortest, so a capped num_repeat silently
                # returns a video shorter than the audio the caller sent. Say so.
                clip_seconds = self.config["infer_frames"] / self.config["target_fps"]
                logger.warning(
                    f"num_repeat={num_repeat} caps this request at {num_repeat * clip_seconds:.1f}s; "
                    f"the audio needs {audio_num_repeat} clips ({audio_num_repeat * clip_seconds:.1f}s) "
                    f"and will be truncated. Set num_repeat to null to follow the audio."
                )

        model_pic = crop_op(resize_op(Image.fromarray(ref_image)))

        ref_pixel_values = tensor_trans(model_pic).unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixel_values = ref_pixel_values.to(dtype=self.vae_encoder.dtype, device=self.vae_encoder.device)

        motion_frames = self.config["motion_frames"]
        motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=self.param_dtype, device=AI_DEVICE)

        neg_prompt = self.input_info.negative_prompt or self.config.get("sample_neg_prompt", "")
        with ProfilingContext4DebugL1(
            "Run Text Encoder",
            recorder_mode=GET_RECORDER_MODE(),
            metrics_func=monitor_cli.lightx2v_run_text_encode_duration,
            metrics_labels=["WanS2VRunner"],
        ):
            text_encoder_output = self.run_text_encoder(self.input_info)
        context = text_encoder_output["context"]
        context_null = text_encoder_output.get("context_null")

        if context_null is None:
            with ProfilingContext4DebugL1("Run Text Encoder (negative)"):
                t5_offload = self.config.get("t5_cpu_offload", self.config.get("cpu_offload", False))
                text_encoder = self.text_encoders[0]
                if not t5_offload:
                    text_encoder.model.to(AI_DEVICE)
                context_null = text_encoder.infer([neg_prompt])
                if t5_offload:
                    text_encoder.model.cpu()

        return {
            "ref_pixel_values": ref_pixel_values,  # check
            "motion_latents": motion_latents,  # todo in pose
            "audio_emb": audio_emb,  # check diff: grade_fn
            "num_repeat": num_repeat,  # check
            "context": context,  # check
            "context_null": context_null,  # check
            "height": height,  # check
            "width": width,  # check
            "seed": self.input_info.seed,
        }

    @ProfilingContext4DebugL1("Run VAE Decoder")
    def run_vae_decoder(self, latents):
        if self.config.get("cpu_offload", False):
            self.vae_decoder.to_cuda()
        images = self.vae_decoder.decode(latents.to(self.param_dtype))
        if self.config.get("cpu_offload", False):
            self.vae_decoder.to_cpu()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return images

    def stabilize_first_frame(self, video):
        """Replace the VAE boundary frame with its immediate successor.

        Causal Wan VAE decoding can leave a one-frame bright/ghosted transition
        where the prepended reference latent meets the generated latent stream.
        This opt-in correction changes only 1/fps seconds and keeps the full
        output duration and audio timeline intact.

        The seam only exists on the first clip of a drop_first_motion run, where
        the ref latent is what gets prepended before decoding. Without
        drop_first_motion the first clip decodes [motion_latents, latents] and
        frame 0 is an ordinary frame, so leave it alone.
        """
        if not self.config.get("s2v_stabilize_first_frame", False) or video.shape[2] < 2:
            return video
        if not self.config["drop_first_motion"]:
            return video
        video = video.clone()
        video[:, :, 0] = video[:, :, 1]
        return video

    def run_dit_clip(self, dit_inputs):
        infer_steps = self.scheduler.infer_steps
        for step_index in range(infer_steps):
            logger.info(f"==> step_index: {step_index + 1} / {infer_steps}")
            with ProfilingContext4DebugL1("step_pre"):
                self.scheduler.step_pre(step_index)
            with ProfilingContext4DebugL1("🚀 infer_main"):
                self.model.infer(dit_inputs)
            with ProfilingContext4DebugL1("step_post"):
                self.scheduler.step_post()
        return self.scheduler.latents

    @ProfilingContext4DebugL2("Run DiT + decode")
    def run_main(self):
        inputs = self.inputs
        height, width = inputs["height"], inputs["width"]
        offload = self.config.get("cpu_offload", False)
        motion_frames = self.config["motion_frames"]
        infer_frames = self.config["infer_frames"]
        drop_first_motion = self.config["drop_first_motion"]
        lat_motion_frames = (motion_frames + 3) // 4

        with ProfilingContext4DebugL1("Run VAE Encoder (ref + motion)"):
            if offload:
                self.vae_encoder.to_cuda()

            with torch.amp.autocast(str(AI_DEVICE), dtype=self.vae_encoder.dtype):
                # cpu_offload: inputs were staged on the VAE's original (cpu) device but the
                # VAE itself was just moved to cuda -- mixed-device conv3d dispatches to
                # slow_conv3d_forward which has no CUDA kernel.
                ref_latents = self.vae_encoder.encode(inputs["ref_pixel_values"].to(AI_DEVICE)).unsqueeze(0)
                motion_latents = self.vae_encoder.encode(inputs["motion_latents"].to(AI_DEVICE)).unsqueeze(0)
            if offload:
                self.vae_encoder.to_cpu()

        videos_last_frames = inputs["motion_latents"].detach()
        out_clips = []
        seed = inputs["seed"] if inputs["seed"] >= 0 else random.randint(0, sys.maxsize)

        num_repeat = inputs["num_repeat"]
        src_pose_path = getattr(self.input_info, "src_pose_path", None) or ""
        if src_pose_path and os.path.isfile(src_pose_path):
            with ProfilingContext4DebugL1("Load pose cond"):
                pose_conds = self.load_pose_cond(src_pose_path, num_repeat, infer_frames, height, width)
            logger.info(f"Loaded pose cond from {src_pose_path} ({len(pose_conds)} clips)")
        else:
            pose_conds = None
            if src_pose_path:
                logger.warning(f"src_pose_path not found, ignoring: {src_pose_path}")

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=self.param_dtype):
            for r in range(num_repeat):
                with ProfilingContext4DebugL1(
                    f"clip end2end {r + 1}/{num_repeat}",
                    recorder_mode=GET_RECORDER_MODE(),
                    metrics_func=monitor_cli.lightx2v_run_segments_end2end_duration,
                    metrics_labels=["WanS2VRunner"],
                ):
                    logger.info(f"start clip {r + 1}/{num_repeat}")
                    clip_seed = seed + r
                    lat_target_frames = (infer_frames + 3 + motion_frames) // 4 - lat_motion_frames
                    latent_shape = (16, lat_target_frames, height // 8, width // 8)

                    with ProfilingContext4DebugL1("prepare_clip"):
                        self.scheduler.prepare_clip(clip_seed, latent_shape, self.param_dtype)
                        if pose_conds is not None:
                            cond_latents = pose_conds[r].to(dtype=self.param_dtype, device=AI_DEVICE)
                        else:
                            cond_latents = self.build_cond_latents(height, width)
                    left_idx = r * infer_frames
                    right_idx = r * infer_frames + infer_frames
                    audio_input = inputs["audio_emb"][..., left_idx:right_idx]

                    dit_inputs = {
                        "text_encoder_output": {
                            "context": inputs["context"],
                            "context_null": inputs["context_null"],
                        },
                        "s2v": {
                            "ref_latents": ref_latents,
                            "motion_latents": motion_latents.clone(),
                            "cond_latents": cond_latents,
                            "audio_input": audio_input,
                            "motion_frames": (motion_frames, lat_motion_frames),
                            "drop_motion_frames": drop_first_motion and r == 0,
                            "add_last_motion": 2,
                        },
                    }
                    if self.config.get("s2v_context_latents", False):
                        dit_inputs["s2v"]["context_latents"] = [ref_latents]

                    with ProfilingContext4DebugL2("Run DiT"):
                        latents = self.run_dit_clip(dit_inputs)

                    if drop_first_motion and r == 0:
                        decode_latents = torch.cat([ref_latents, latents.unsqueeze(0)], dim=2)
                    else:
                        decode_latents = torch.cat([motion_latents, latents.unsqueeze(0)], dim=2)

                    image = self.run_vae_decoder(decode_latents.squeeze(0))

                    image = image[:, :, -infer_frames:]
                    if drop_first_motion and r == 0:
                        image = image[:, :, 3:]
                    if r == 0:
                        image = self.stabilize_first_frame(image)

                    overlap = min(motion_frames, image.shape[2])
                    image = image.to(AI_DEVICE)
                    videos_last_frames = videos_last_frames.to(AI_DEVICE)
                    videos_last_frames = torch.cat(
                        [
                            videos_last_frames[:, :, overlap:],
                            image[:, :, -overlap:],
                        ],
                        dim=2,
                    )
                    with ProfilingContext4DebugL1("Run VAE Encoder (motion update)"):
                        if offload:
                            self.vae_encoder.to_cuda()
                        with torch.amp.autocast(str(AI_DEVICE), dtype=self.vae_encoder.dtype):
                            motion_latents = self.vae_encoder.encode(videos_last_frames).unsqueeze(0)
                        if offload:
                            self.vae_encoder.to_cpu()
                    out_clips.append(image.cpu())

        self.gen_video_final = torch.cat(out_clips, dim=2)[0]
        if offload:
            gc.collect()
            torch.cuda.synchronize()
        return self.process_images_after_vae_decoder()

    @ProfilingContext4DebugL1("Process after vae decoder")
    def process_images_after_vae_decoder(self):
        video = self.gen_video_final
        if video.dim() == 4 and video.shape[0] == 3:
            video = video.permute(1, 2, 3, 0)
        self.gen_video_final = ((video.float().clamp(-1, 1) + 1.0) * 0.5).cpu()

        if self.input_info.return_result_tensor:
            return {"video": self.gen_video_final}
        if self.input_info.save_result_path is not None and is_main_process():
            out_path = self.input_info.save_result_path
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            save_to_video(self.gen_video_final, out_path, fps=self.config["target_fps"], method="ffmpeg")
            audio_path = getattr(self.input_info, "audio_path", None)
            if audio_path and os.path.isfile(audio_path):
                try:
                    merge_video_audio(out_path, audio_path)
                    logger.info(f"Muxed audio from {audio_path}")
                except Exception as exc:
                    logger.warning(f"Audio mux failed: {exc}")
            logger.info(f"Video saved to {out_path}")
        return {"video": None}

    def end_run(self):
        return

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["WanS2VRunner"],
    )
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        self.inputs = self.run_input_encoder()
        result = self.run_main()
        self.end_run()
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        return result


class S2VMultiModelStruct(MultiModelStruct):
    """Lazy single-resident high/low expert router for Wan2.2 S2V."""

    def __init__(self, config, boundary, num_train_timesteps, paths, init_device):
        super().__init__([None, None], config, boundary, num_train_timesteps)
        self.paths = paths
        self.init_device = init_device

    @property
    def device(self):
        model = self.model[self.cur_model_index] if self.cur_model_index >= 0 else None
        return model.device if model is not None else self.init_device

    def _build(self, index):
        tag = "HIGH" if index == 0 else "LOW"
        logger.info(f"[s2v-moe] loading {tag}-noise expert from {self.paths[index]}")
        model = WanS2VModel(self.paths[index], self.config, self.init_device)
        model.set_scheduler(self.scheduler)
        return model

    def _free(self, index):
        if self.model[index] is not None:
            logger.info(f"[s2v-moe] releasing {'HIGH' if index == 0 else 'LOW'}-noise expert")
            self.model[index] = None
            gc.collect()
            torch.cuda.empty_cache()

    def get_current_model_index(self):
        # Deliberately does NOT reproduce the parent's per-expert
        # `scheduler.sample_guide_scale = config["sample_guide_scale"][index]`:
        # WanS2VScheduler reads that key as a scalar (float(...)), so a per-expert
        # list would fail at scheduler construction. WanS2VMoeRunner rejects a list
        # up front; both experts therefore share one guidance scale.
        index = 0 if self.scheduler.timesteps[self.scheduler.step_index] >= self.boundary_timestep else 1
        self.cur_model_index = index
        return index

    def infer(self, inputs):
        index = self.get_current_model_index()
        if self.model[index] is None:
            # Single-resident: release the other expert before building this one, so peak
            # footprint stays at one expert. Cost is one full expert load from disk at each
            # high->low crossing, i.e. two loads per clip. See docs/bernini_s2v.md.
            self._free(1 - index)
            self.model[index] = self._build(index)
        self.model[index].infer(inputs)

    # Residency is handled by build/free above, so the parent's model-level moves are
    # never needed here -- and would hit None for the non-resident expert. Keep them
    # harmless rather than letting a stray caller raise AttributeError.
    def offload_cpu(self, model_index):
        if self.model[model_index] is not None:
            self.model[model_index].to_cpu()

    def to_cuda(self, model_index):
        if self.model[model_index] is not None:
            self.model[model_index].to_cuda()


@RUNNER_REGISTER("wan2.2_s2v_moe")
class WanS2VMoeRunner(WanS2VRunner):
    """Bernini-R-S2V dual-expert runner with explicit asset validation."""

    def __init__(self, config):
        # Before super(): DefaultRunner.__init__ builds WanS2VScheduler, which does
        # float(config["sample_guide_scale"]) and would raise an opaque TypeError first.
        if isinstance(config.get("sample_guide_scale"), (list, tuple)):
            raise ValueError("wan2.2_s2v_moe shares one guidance scale across both experts; sample_guide_scale must be a scalar, not a per-expert list.")
        # S2VMultiModelStruct overrides get_current_model_index and so never runs the
        # parent's model-level to_cuda/offload_cpu dance. Under cpu_offload the experts
        # are built on cpu (DefaultRunner.set_init_device) and would silently stay there.
        # Block granularity is fine -- it offloads inside the resident expert.
        if config.get("cpu_offload", False) and config.get("offload_granularity", "block") == "model":
            raise ValueError("wan2.2_s2v_moe manages expert residency itself; use offload_granularity 'block', not 'model'.")
        super().__init__(config)
        self.high_noise_model_path = config.get("high_noise_original_ckpt") or os.path.join(config["model_path"], "high_noise_model")
        self.low_noise_model_path = config.get("low_noise_original_ckpt") or os.path.join(config["model_path"], "low_noise_model")
        self._validate_expert(self.high_noise_model_path, "high")
        self._validate_expert(self.low_noise_model_path, "low")

    @staticmethod
    def _validate_expert(path, label):
        required = ["config.json", "diffusion_pytorch_model.safetensors.index.json", "non_block.safetensors"]
        required.extend(f"block_{index}.safetensors" for index in range(40))
        missing = [name for name in required if not os.path.exists(os.path.join(path, name))]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"{label}-noise S2V expert is incomplete at {path}: {preview}")

    def load_transformer(self):
        boundary = self.config.get("boundary", 0.875)
        num_train_timesteps = self.config.get("num_train_timesteps", 1000)
        paths = {0: self.high_noise_model_path, 1: self.low_noise_model_path}
        logger.info(f"[s2v-moe] high={paths[0]} low={paths[1]} boundary={boundary}")
        return S2VMultiModelStruct(self.config, boundary, num_train_timesteps, paths, self.init_device)
