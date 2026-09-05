import contextlib
import os
import shutil
import subprocess
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import imageio
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from decord import VideoReader
from loguru import logger
from safetensors import safe_open

from lightx2v.models.networks.swiftvr import (
    AntiphaseBlender,
    RestorationAutoencoder,
    SwiftVRModel,
    SwiftVRRestorer,
    build_video_chunks,
    normalize_swiftvr_config,
    padded_frame_count,
)
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import mux_audio_from_video, save_to_image


def mark_stage(device: torch.device):
    if device.type == "cuda":
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event
    return time.perf_counter()


def measure_stage_durations(marks, device: torch.device):
    if device.type == "cuda":
        marks[-1].synchronize()
        return [start.elapsed_time(end) / 1000 for start, end in zip(marks, marks[1:])]
    return [end - start for start, end in zip(marks, marks[1:])]


@dataclass
class PendingVideoWrite:
    chunk_index: int
    frame_count: int
    read_seconds: float
    reader_wait_seconds: float
    future: Future


@RUNNER_REGISTER("swiftvr")
class SwiftVRRunner(DefaultRunner):
    """Native LightX2V runner for SwiftVR image and video restoration."""

    # Two spatial shapes trigger dynamic compilation before serving requests.
    WARMUP_RESOLUTIONS = ((720, 1280), (2048, 1536))

    def __init__(self, config):
        if config["task"] != "sr":
            raise ValueError("SwiftVR only supports the `sr` task.")
        parallel = config.get("parallel") or {}
        if parallel:
            # 上游原本一律拒绝多卡。这里只放开「段级数据并行」这一种:整块视频按 chunk
            # 连续切给各 rank,块之间不需要跨卡通信,末尾按块号 concat。
            # 序列/张量并行仍然拒绝 —— SwiftVR 的注意力是窗口内 SDPA,切 head 或切 seq
            # 都要动 MFSWA 的窗口布局,不是配置层面的事。
            if not isinstance(parallel, dict) or set(parallel) - {"seg_p_size"}:
                raise ValueError(f"SwiftVR only supports segment parallel via parallel.seg_p_size, got {parallel!r}")
        if config.get("cpu_offload"):
            raise NotImplementedError("SwiftVR does not support CPU offload yet.")
        normalize_swiftvr_config(config)
        super().__init__(config)
        self.copy_stream = torch.cuda.Stream(device=self.init_device) if self.init_device.type == "cuda" else None
        self._seg_gloo_group = None

    def init_modules(self):
        logger.info(f"Loading native SwiftVR weights from {self.config['model_path']}")
        self.model = SwiftVRModel(self.config["model_path"], self.config, self.init_device)
        autoencoder = RestorationAutoencoder.from_pretrained(
            self.config["model_path"],
            self.init_device,
            GET_DTYPE(),
        )
        with safe_open(
            os.path.join(self.config["model_path"], "prompt_embedding.safetensors"),
            framework="pt",
            device="cpu",
        ) as weights:
            prompt_embedding = weights.get_tensor("prompt_emb").to(self.init_device, GET_DTYPE())

        def build_restorer(strength: float):
            return SwiftVRRestorer(
                autoencoder,
                self.model,
                prompt_embedding,
                overlap=self.config.get("dit_overlap", 0),
                reae_frame_batch_size=self.config.get("reae_frame_batch_size", 1),
                strength=strength,
            )

        # 两个数,两件事,别混：
        #
        # restoration_strength 是**用户旋钮**。1.0 = 上游原行为;调低用于抑制「把源本身的
        # 缺陷(水波纹/噪点/压缩块)当成细节一起放大」——SwiftVR 在重退化数据上训过,喂它
        # 轻退化素材(AIGC 直出、干净拍摄)会过度生成。
        #
        # antiphase_strength_gain 是**引擎补偿**。反相平均是逐像素取两条流的均值,这个平均
        # 本身要削掉约三成高频(实测锐度 lap 2905 -> 1969),所以双跑路径得把残差按比例放大
        # 才能回到基线画质(1.40 实测补平:2876~2921)。它不代表「用户想要更强的恢复」。
        base_strength = float(self.config.get("restoration_strength", 1.0))
        antiphase = bool(self.config.get("antiphase_dual_pass"))
        gain = float(self.config.get("antiphase_strength_gain", 1.4)) if antiphase else 1.0

        self.restorer = build_restorer(base_strength * gain)
        # 反相双跑的第二条流。SwiftVRRestorer 只持有流式状态(encoder_state/decoder_state/
        # temporal_offset),权重在 autoencoder 与 self.model 里,由 run_causal_layers 把
        # 状态穿进穿出 —— 所以再建一个实例是**共享权重、独立状态**,不吃第二份显存。
        self.restorer_shifted = build_restorer(base_strength * gain) if antiphase else None
        # 图像路径**不能**吃补偿系数:单张图没有时间轴,反相平均无从谈起,补偿加上去就是纯
        # 过锐(实测 s1.40 不做反相 lap 4349,是基线 2905 的 1.5 倍)。而 run_pipeline 是按
        # 输入类型**运行时**分流的,同一个部署会同时收到图和视频,靠约定隔离必然静默出错。
        # 反相关闭时它与 self.restorer 等价,不必单独建 —— restore_frames 的
        # `restorer or self.restorer` 会自然兜住 None。
        self.restorer_image = build_restorer(base_strength) if antiphase else None
        self.config.lock()

    def _reset_restorers(self):
        for restorer in (self.restorer, self.restorer_shifted, self.restorer_image):
            if restorer is not None:
                restorer.reset()

    def _seed_temporal_offset(self, latent_offset: int):
        self.restorer.seed_temporal_offset(latent_offset)
        if self.restorer_shifted is not None:
            self.restorer_shifted.seed_temporal_offset(latent_offset)

    @ProfilingContext4DebugL1("Warmup")
    @torch.inference_mode()
    def run_warmup(self):
        clip_length = self.config.get("clip_len", 24)
        clip_latents = clip_length // 4
        # One first, middle, and one-frame last chunk cover the 7- and 6-latent DiT paths.
        chunks = build_video_chunks(2 * clip_length + 5, clip_length)

        for height, width in self.WARMUP_RESOLUTIONS:
            padded_height = height + (-height) % 32
            padded_width = width + (-width) % 32
            logger.info(f"Warmup: {height}x{width}")
            try:
                for chunk in chunks:
                    video = torch.zeros(
                        1,
                        chunk.frame_count,
                        3,
                        padded_height,
                        padded_width,
                        dtype=GET_DTYPE(),
                        device=self.init_device,
                    )
                    restored = self.restorer.restore_chunk(video, chunk, clip_latents)
                    del video, restored
            finally:
                # 移位流走同一批权重、同一组形状,编译缓存与它共用,不必单独预热一遍;
                # 但它的因果状态必须干净地进正式请求。
                self._reset_restorers()

        logger.info("[Warmup] Warmup completed")
        self._maybe_freeze_gc()

    def _oriented_target(self, source_height: int, source_width: int) -> tuple[int, int]:
        """配置里的标称输出档位,按源的朝向配对长短边。

        与 SeedVR2 的 `_oriented_target`(seedvr_runner.py:220)同语义:config 写的
        1080×1920 表示的是「1080P 这一档」,不是死的高×宽 —— 竖版源要出 1080×1920、
        横版出 1920×1080、方形塌到短边(横竖都救不了方的)。

        返回 (0, 0) 表示没配档位,调用方退回纯 sr_ratio 放大。
        """
        target_height = int(self.config.get("target_height", 1080) or 0)
        target_width = int(self.config.get("target_width", 1920) or 0)
        if target_height <= 0 or target_width <= 0:
            return 0, 0
        long_side, short_side = max(target_height, target_width), min(target_height, target_width)
        if source_height > source_width:
            return long_side, short_side
        if source_width > source_height:
            return short_side, long_side
        return short_side, short_side

    def _clamp_to_ceiling(self, height: int, width: int) -> tuple[int, int]:
        """按面积把输出压到显存安全上限内,保持请求的宽高比。

        这条是兜底护栏,拦的是**显式 target_shape**:标称档位那条路已经被
        `_oriented_target` 封住了,但请求可以直接点名尺寸,而 SwiftVR 的显存随像素数
        线性涨(实测单卡 1080p 15.8G / 2K 19.8G / 4K 32.5G,40G 卡上 4K 已是上限)。
        没有这道闸,一个 8K 的 target_shape 就能把整台机器的卡打爆。

        按**面积**而不是逐边压:显存跟像素数走,而且等比缩放不会改变请求的画幅比例。
        配 0 或负数 = 关闭上限(自建/离线跑大图时用)。
        """
        max_height = int(self.config.get("max_target_height", 2160) or 0)
        max_width = int(self.config.get("max_target_width", 3840) or 0)
        if max_height <= 0 or max_width <= 0:
            return height, width
        budget = max_height * max_width
        if height * width <= budget:
            return height, width
        scale = (budget / (height * width)) ** 0.5
        clamped_height, clamped_width = max(1, round(height * scale)), max(1, round(width * scale))
        logger.warning(f"SwiftVR: requested {width}x{height} exceeds the {max_width}x{max_height} ceiling, clamped to {clamped_width}x{clamped_height}")
        return clamped_height, clamped_width

    def resolve_output_size(
        self,
        input_info,
        source_height: int,
        source_width: int,
        *,
        require_even: bool = False,
    ) -> tuple[int, int]:
        """定输出尺寸。两条路各有各的封顶,别把它们混起来看。

        ⚠️ sr_ratio 这条路**必须**封顶,这是现网 SR 的既有契约,不是防呆:网关前端
        固定下发 sr_ratio=4.0(new-api `VIDEO_SR_RATIO_UNCAPPED`),注释原话是"一个够不着
        的上限",指望引擎拿 config 档位去 min 掉它 —— SeedVR2 靠 seedvr_runner.py:241
        那个 min 兜住。SwiftVR 早先没有这层,1344×768 的源乘 4 会去出 5376×3072(≈5.3K),
        远超 4K/33G 的实测上限,必 OOM,而且不报参数错、跑起来才炸。

        target_shape 这条路是 SwiftVR 独有的能力(SeedVR2 压根不读 target_shape),
        一份配置串行吃 1080p/2K/4K 靠的就是它,所以**不能**用档位去封 —— 那会把 2K/4K
        一起封死。它只过 `_clamp_to_ceiling` 的显存护栏。

        target_short_edge 是给网关用的第三条路:只给一个目标短边,输出按**源的真实画幅**
        等比放大。它存在的理由是「只有引擎知道源有多大」—— 网关手里只有用户选的比例
        标签,而标签和实际画幅并不相等(H3 的 768P/16:9 实际出 1344x768 = 1.75,不是
        1.778;wan 的 720P 是 1280x720 = 1.778),照标签算 target_shape 会带约 1.6% 的
        横向拉伸,且档位越高越明显。按短边对齐则画幅零形变、短边精确命中目标档,
        1080P/2K/4K 共用同一个机制,网关只需下发一个数字。

        三条路的优先级:target_shape(精确尺寸,调用方完全指定) > target_short_edge
        (指定档位、画幅随源) > sr_ratio(倍率,按部署档位封顶)。
        """
        if input_info.target_shape:
            if len(input_info.target_shape) != 2:
                raise ValueError(f"SwiftVR target_shape must be [height, width], got {input_info.target_shape}")
            height, width = input_info.target_shape
        elif int(getattr(input_info, "target_short_edge", 0) or 0) > 0:
            edge = int(input_info.target_short_edge)
            source_edge = min(source_height, source_width)
            if source_edge <= 0:
                raise ValueError(f"SwiftVR source size must be positive, got {source_width}x{source_height}")
            scale = edge / source_edge
            height, width = round(source_height * scale), round(source_width * scale)
        else:
            ratio = input_info.sr_ratio
            height, width = round(source_height * ratio), round(source_width * ratio)
            target_height, target_width = self._oriented_target(source_height, source_width)
            if target_height and height * width > target_height * target_width:
                # 落到档位上就出**精确**档位尺寸,不是等比缩到档位面积:界面标着 1080P
                # 就得真是 1920×1080。SeedVR2 是靠 fixed_shape 中心裁到精确档位达成的
                # (new-api videoPlayground.constants.js:512 记了这个决定),SwiftVR 这边
                # preprocess_frames 直接 interpolate 到目标尺寸,等价结果、少一次裁剪。
                # 代价是源与档位画幅比不一致时会有轻微拉伸(1344×768 的 1.75 → 1.778,
                # 1.6%),而 SeedVR2 那边是上下各裁 12 像素 —— 两种取舍,量级都可忽略。
                height, width = target_height, target_width
        if height <= 0 or width <= 0:
            raise ValueError(f"SwiftVR output size must be positive, got {height}x{width}")
        height, width = self._clamp_to_ceiling(height, width)
        if require_even:
            height = max(2, int(round(height / 2)) * 2)
            width = max(2, int(round(width / 2)) * 2)
        return height, width

    @staticmethod
    def resolve_input_kind(input_info) -> str:
        has_image = bool(input_info.image_path)
        has_video = bool(input_info.video_path)
        if has_image == has_video:
            raise ValueError("SwiftVR requires exactly one of `image_path` or `video_path`.")
        return "image" if has_image else "video"

    @staticmethod
    def read_image_frame(image_path: str):
        with Image.open(image_path) as image:
            frame = np.array(image.convert("RGB"), dtype=np.uint8)
        source_height = frame.shape[0] // 8 * 8
        source_width = frame.shape[1] // 8 * 8
        if source_height <= 0 or source_width <= 0:
            raise ValueError(f"SwiftVR image is too small after 8-pixel alignment: {frame.shape[:2]}.")
        frames = torch.from_numpy(frame[:source_height, :source_width]).permute(2, 0, 1).contiguous().unsqueeze(0)
        return frames, source_height, source_width

    def restore_frames(
        self,
        frames,
        chunk,
        clip_latents,
        output_height,
        output_width,
        pad_height,
        pad_width,
        stage_marks=None,
        restorer=None,
    ):
        video = self.preprocess_frames(
            frames,
            output_height,
            output_width,
            pad_height,
            pad_width,
            GET_DTYPE(),
            self.init_device,
            self.config.get("upscale_mode", "bilinear"),
        )
        if stage_marks is not None:
            stage_marks.append(mark_stage(self.init_device))
        restored = (restorer or self.restorer).restore_chunk(video, chunk, clip_latents)
        if stage_marks is not None:
            stage_marks.append(mark_stage(self.init_device))
        return restored[..., :output_height, :output_width]

    @staticmethod
    def read_video_frames(reader, chunk, raw_frame_count: int, source_height: int, source_width: int, pin_memory: bool, lead: int = 0):
        """读一块。``lead>0`` 时向前多读几帧,供反相双跑的移位流用。

        反相双跑要的两条流是 [start, start+n) 与 [start-1, start+n-1),并集恰好是
        [start-1, start+n) —— 多读一帧就够,不必读两遍。块首越界的负号夹到 0,
        等价于「复制首帧插到最前面」,与离线配方一致。
        """
        started_at = time.perf_counter()
        indices = [min(max(index, 0), raw_frame_count - 1) for index in range(chunk.start - lead, chunk.start + chunk.frame_count)]
        frames = reader.get_batch(indices)
        if not torch.is_tensor(frames):
            frames = torch.from_numpy(frames.asnumpy())
        frames = frames[:, :source_height, :source_width].permute(0, 3, 1, 2).contiguous()
        if pin_memory:
            frames = frames.pin_memory()
        return frames, time.perf_counter() - started_at

    @staticmethod
    def preprocess_frames(frames, height: int, width: int, pad_height: int, pad_width: int, dtype: torch.dtype, device: torch.device, mode: str):
        frames = frames.to(device=device, dtype=dtype, non_blocking=frames.is_pinned())
        if frames.shape[-2:] != (height, width):
            interpolate_args = {"align_corners": False} if mode in {"linear", "bilinear", "bicubic", "trilinear"} else {}
            frames = F.interpolate(frames, size=(height, width), mode=mode, **interpolate_args)
        frames.div_(255)
        if pad_height or pad_width:
            frames = F.pad(frames, (0, pad_width, 0, pad_height))
        return frames.unsqueeze(0)

    @staticmethod
    def copy_frames_to_cpu(frames, copy_stream):
        if copy_stream is None:
            return frames.cpu(), time.perf_counter()

        cpu_frames = torch.empty(frames.shape, dtype=frames.dtype, device="cpu", pin_memory=True)
        copy_stream.wait_stream(torch.cuda.current_stream(frames.device))
        with torch.cuda.stream(copy_stream):
            cpu_frames.copy_(frames, non_blocking=True)
            frames.record_stream(copy_stream)
            copy_complete = torch.cuda.Event(enable_timing=True)
            copy_complete.record(copy_stream)
        return cpu_frames, copy_complete

    def open_video_writer(self, output_path: str, fps: float):
        quality = self.config.get("quality", 60)
        codec = self.config.get("video_codec", "libx265")
        ffmpeg_params = ["-crf", str(round((100 - quality) * 51 / 100)), "-movflags", "+faststart"]
        if codec == "libx265":
            ffmpeg_params.extend(["-x265-params", "log-level=warning"])
        # Common x264/x265 presets from fastest to slowest:
        # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow, placebo.
        preset = self.config.get("ffmpeg_preset", "")
        if preset:
            ffmpeg_params.extend(["-preset", preset])
        pixel_format = "yuv444p" if self.config.get("save_format") == "yuv444p" else "yuv420p"
        return imageio.get_writer(
            output_path,
            fps=fps,
            codec=codec,
            pixelformat=pixel_format,
            macro_block_size=None,
            ffmpeg_params=ffmpeg_params,
        )

    @staticmethod
    def write_video_frames(writer, frames, stage_marks, device: torch.device):
        stage_durations = measure_stage_durations(stage_marks, device)
        for frame in frames.numpy():
            writer.append_data(frame)
        return stage_durations

    @staticmethod
    def finish_video_write(pending_write: PendingVideoWrite, stage_seconds, chunk_count: int):
        wait_started_at = time.perf_counter()
        preprocess_seconds, restore_seconds, postprocess_seconds = pending_write.future.result()
        writer_wait_seconds = time.perf_counter() - wait_started_at

        stage_seconds["read"] += pending_write.read_seconds
        stage_seconds["reader_wait"] += pending_write.reader_wait_seconds
        stage_seconds["preprocess"] += preprocess_seconds
        stage_seconds["restore"] += restore_seconds
        stage_seconds["postprocess_d2h"] += postprocess_seconds
        stage_seconds["writer_wait"] += writer_wait_seconds
        logger.info(
            f"SwiftVR chunk {pending_write.chunk_index + 1}/{chunk_count} restored {pending_write.frame_count} frames "
            f"(read={pending_write.read_seconds:.3f}s, reader_wait={pending_write.reader_wait_seconds:.3f}s, "
            f"preprocess={preprocess_seconds:.3f}s, restore={restore_seconds:.3f}s, "
            f"postprocess_d2h={postprocess_seconds:.3f}s, writer_wait={writer_wait_seconds:.3f}s)"
        )

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["SwiftVRRunner"],
        profile_memory=True,
    )
    @torch.inference_mode()
    def run_pipeline(self, input_info):
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_count.inc()
        self.input_info = input_info
        input_kind = self.resolve_input_kind(input_info)
        if input_kind == "image":
            return self.run_image_pipeline(input_info)
        return self.run_video_pipeline(input_info)

    def run_image_pipeline(self, input_info):
        if not input_info.return_result_tensor and not input_info.save_result_path:
            raise ValueError("SwiftVR image restoration requires `save_result_path` unless the image is returned in memory.")

        frames, source_height, source_width = self.read_image_frame(input_info.image_path)
        output_height, output_width = self.resolve_output_size(input_info, source_height, source_width)
        pad_height = (-output_height) % 32
        pad_width = (-output_width) % 32
        clip_length = self.config.get("clip_len", 24)
        chunk = build_video_chunks(1, clip_length)[0]
        clip_latents = clip_length // 4

        output_path = None if input_info.return_result_tensor else os.path.abspath(input_info.save_result_path)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        started_at = time.perf_counter()
        # 单张图没有时间轴,反相双跑无从谈起 —— 图像路径固定走正相这一条流,而且用的是
        # **不含反相补偿**的那个 restorer(见 init_modules)。反相关闭时它是 None,
        # restore_frames 会退回 self.restorer。
        self._reset_restorers()
        try:
            restored = self.restore_frames(
                frames,
                chunk,
                clip_latents,
                output_height,
                output_width,
                pad_height,
                pad_width,
                restorer=self.restorer_image,
            )
            images = restored[0].permute(0, 2, 3, 1).contiguous()
            if input_info.return_result_tensor:
                images = images.to(device="cpu", dtype=torch.float32)
            else:
                save_to_image(images, output_path)
                images = None
        finally:
            self._reset_restorers()

        elapsed = time.perf_counter() - started_at
        stats = {
            "frames": 1,
            "seconds": elapsed,
            "fps": 1 / elapsed if elapsed else 0.0,
            "output": output_path,
        }
        if self.progress_callback:
            self.progress_callback(100, 100)
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"SwiftVR restored image to {output_path or 'memory'} in {elapsed:.3f}s")
        return {"images": images, "stats": stats}

    def _seg_barrier(self):
        """控制面汇合点,走 **gloo**,不走 NCCL。

        这里要同步的是「分片文件是否已经落盘」——主机侧的事。NCCL 的集合是 GPU 流上的
        操作,两次实测都翻车:
          1) `dist.barrier()` + `torch.cuda.synchronize()`:rank3 直接冲过屏障返回,
             而 rank1/2 还没关闭 mp4 写出,rank0 提前拼接,成片只有 97 帧(应为 372);
          2) 改成 `all_reduce` + `.item()`:rank0/rank3 配成一对、rank1/rank2 配成
             另一对,双方各自拿到 "2 of 4"。
        gloo 组在 CPU 上跑,barrier 语义就是主机侧阻塞,才是这个场景该用的东西。
        """
        group = self._seg_gloo_group
        if group is None:
            return
        dist.barrier(group=group)

    def _seg_parallel_info(self):
        """段级并行的 ``(rank, world)``;未启用时返回 ``(0, 1)``。

        只有 file 输出这一条路径,所以不像 SeedVR 那样还要判 tensor 输出 ——
        run_video_pipeline 开头已经强制要求 save_result_path。
        """
        parallel = self.config.get("parallel") or {}
        seg_size = int(parallel.get("seg_p_size", 1)) if isinstance(parallel, dict) else 1
        if seg_size <= 1 or not dist.is_available() or not dist.is_initialized():
            return 0, 1
        world = dist.get_world_size()
        if world != seg_size:
            raise ValueError(f"seg_p_size ({seg_size}) must equal world_size ({world}).")
        return dist.get_rank(), world

    @staticmethod
    def _partition_chunks(chunk_count: int, world: int) -> list[tuple[int, int]]:
        """把块按**连续区间**切给各 rank,不是轮转。

        连续切是关键:每个 rank 只需要在自己区间开头预热一次因果状态;轮转的话
        每一块前面都得预热,开销要乘以块数。
        """
        base, extra = divmod(chunk_count, world)
        ranges, start = [], 0
        for r in range(world):
            take = base + (1 if r < extra else 0)
            ranges.append((start, start + take))
            start += take
        return ranges

    def _concat_segment_parts(self, part_paths: list[str], output_path: str):
        """按块号顺序把各 rank 的分片拼成成片。各 rank 编码参数相同,可以 -c copy。"""
        list_path = output_path + ".concat.txt"
        with open(list_path, "w", encoding="utf-8") as handle:
            for path in part_paths:
                handle.write(f"file '{os.path.abspath(path)}'\n")
        command = ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path]
        try:
            subprocess.run(command, check=True)
        finally:
            with contextlib.suppress(OSError):
                os.remove(list_path)

    def run_video_pipeline(self, input_info):
        if not input_info.save_result_path:
            raise ValueError("SwiftVR video restoration requires `save_result_path`.")
        if input_info.return_result_tensor:
            raise ValueError("SwiftVR video restoration does not support `return_result_tensor`.")

        reader = VideoReader(input_info.video_path)
        raw_frame_count = len(reader)
        first_frame = reader[0]
        source_height = first_frame.shape[0] // 8 * 8
        source_width = first_frame.shape[1] // 8 * 8
        output_height, output_width = self.resolve_output_size(
            input_info,
            source_height,
            source_width,
            require_even=True,
        )
        pad_height = (-output_height) % 32
        pad_width = (-output_width) % 32
        fps = self.config.get("fps") or reader.get_avg_fps() or 30

        clip_length = self.config.get("clip_len", 24)
        process_frame_count = padded_frame_count(raw_frame_count)
        chunks = build_video_chunks(process_frame_count, clip_length)
        clip_latents = clip_length // 4

        output_path = os.path.abspath(input_info.save_result_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # ---- 段级数据并行 ----
        # 块之间唯一的真依赖是 ReAE 的因果状态(encoder_state/decoder_state)和
        # StreamingTransformer 的 temporal_offset。实测因果状态在 4-8 帧内就收敛
        # (从第 96 帧冷启动 vs 完整跑,首帧差 2.1、稳态差 1.09 即编码噪声),所以每个
        # rank 只要在自己区间前多跑一块预热就够;temporal_offset 是纯记账,直接播种。
        # 因此这里不需要任何跨卡通信,只在末尾按块号 concat —— PCIe 无 NVLink 也不吃亏。
        seg_rank, seg_world = self._seg_parallel_info()
        # new_group 是集合操作,必须所有 rank 以相同顺序调用 —— 放在这里,
        # 因为无论是否启用段并行,每个 rank 都会执行到这一行。
        if seg_world > 1 and self._seg_gloo_group is None:
            self._seg_gloo_group = dist.new_group(backend="gloo")
        # 预热块数。默认 1 足够,原因分两条:
        #
        # 1) ReAE 的因果状态收敛极快。把源视频从第 96 帧切开单跑、与完整跑逐帧比:
        #    首帧差 2.1、第 4 帧 1.3、第 8 帧起稳定在 1.09(编码噪声级)。一块绰绰有余。
        #
        # 2) dit_overlap 让 DiT 额外看前一块尾部的若干 latent 帧(previous_input)。
        #    预热块跑完就把它设好了,所以段并行**不与 dit_overlap 冲突** —— 实测
        #    4 卡+overlap2 vs 单卡+overlap2 平均差 0.303、差>3 占 0.032%,比 overlap
        #    这个参数本身造成的差异(0.621)还小。
        #
        # ⚠️ 但 seg_warmup_chunks=0 配 dit_overlap>0 是个静默陷阱:各 rank 的首块拿不到
        #    previous_input,那一块会退化成无重叠,不报错,只在块边界留细微不连续。
        #    要把它调成 0,得同时确认 dit_overlap 也是 0。
        warmup_chunks = max(0, int(self.config.get("seg_warmup_chunks", 1)))
        antiphase = self.restorer_shifted is not None
        if antiphase and seg_world > 1 and warmup_chunks < 1:
            # 反相双跑靠 carry 跨块传递「上一块最后一帧」。段并行下,rank r 的首帧要与
            # 前一块的 carry 配对,而那个 carry 正是预热块产生的。没有预热块,每个 rank
            # 都会少发一帧,成片短 (world-1) 帧 —— 末尾的帧数校验会拦住,但报错离根因太远。
            raise ValueError("SwiftVR antiphase_dual_pass with segment parallel requires seg_warmup_chunks >= 1.")
        parts_dir = None
        part_paths = []
        if seg_world > 1:
            seg_ranges = self._partition_chunks(len(chunks), seg_world)
            my_start, my_end = seg_ranges[seg_rank]
            run_start = max(0, my_start - warmup_chunks)
            parts_dir = os.path.join(os.path.dirname(output_path), f".{os.path.basename(output_path)}.segparts")
            if seg_rank == 0:
                shutil.rmtree(parts_dir, ignore_errors=True)
                os.makedirs(parts_dir, exist_ok=True)
            # 汇合点:保证目录建好之后各 rank 才开始写
            self._seg_barrier()
            # 块数 < 卡数时尾部 rank 分不到块。短视频必然如此:clip_len=24 下首块吃 28 帧、
            # 之后每块 24,要凑够 4 块得 74 帧(24fps 才 3.1 秒)——2~5 秒的素材全部落在这里。
            # 这些 rank **不产出分片**,而不是产出 0 字节分片:后者会被 rank0 的完整性检查
            # 当成「这一路挂了」而抛 missing or empty segment parts(上游报告问题 C / G8)。
            # 它们仍然要走两个 gloo 汇合点,否则其余 rank 会卡在屏障上。
            owner_ranks = [r for r, (start, end) in enumerate(seg_ranges) if start < end]
            part_paths = [os.path.join(parts_dir, f"part_{r:03d}.mp4") for r in owner_ranks]
            writer_path = os.path.join(parts_dir, f"part_{seg_rank:03d}.mp4")
            if my_start >= my_end:
                logger.info(f"SwiftVR seg_parallel: rank {seg_rank}/{seg_world} idle — only {len(chunks)} chunk(s) to go around, {len(owner_ranks)} rank(s) working")
                self._seg_barrier()  # 与末尾「分片都已落盘」那个汇合点配对
                return {"video": None, "stats": {"frames": 0, "seconds": 0.0}}
            logger.info(f"SwiftVR seg_parallel: rank {seg_rank}/{seg_world} owns chunks [{my_start}, {my_end}) of {len(chunks)}, warming up from chunk {run_start}")
        else:
            my_start, my_end, run_start = 0, len(chunks), 0
            writer_path = output_path

        # 首块用 clip_len+4 帧进、只吐 clip_len+1 帧出 —— 3 帧被 ReAE 的因果预热吃掉。
        # 所以 chunk.start(名义帧位)比「实际已发出帧数」多 3,拿它算末块裁剪会多剪 3 帧:
        # 实测 4 卡成片 369 帧、单卡 372,差的正是这个。
        first_chunk_priming = max(0, chunks[0].frame_count - (clip_length + 1)) if len(chunks) > 1 else 0

        run_chunks = chunks[run_start:my_end]
        # check_stop() 默认每步做一次全 rank 的 MAX all-reduce。段并行下各 rank 的块数
        # 不一样(31 块切 4 份是 8/8/8/7,再加预热块),集合次数对不上就会死锁 ——
        # 实测:rank0/rank3 跑完退出循环,rank1/rank2 卡在自己多出来的那次 all_reduce 上,
        # GPU 100% 空转,块数永远停在 23。base_runner.check_stop 的文档也写明了这点。
        # 解法同 SeedVR 段并行:整个并行段内改成各 rank 本地判断,只在末尾 gloo 汇合点
        # 统一(汇合点本身就是所有 rank 都会到的)。
        outer_rank_local = getattr(self, "_rank_local_collectives", False)
        self._rank_local_collectives = outer_rank_local or seg_world > 1
        self._reset_restorers()
        # 播种全局 RoPE 时间位置:各 rank 都从 0 起算的话,拼接处会整段错位。
        self._seed_temporal_offset(sum(c.latent_count for c in chunks[:run_start]))
        blender = AntiphaseBlender() if antiphase else None
        read_lead = 1 if antiphase else 0
        writer = self.open_video_writer(writer_path, fps)
        reader_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-reader")
        writer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-writer")
        max_pending = self.config.get("queue_size", 3)
        pending_reads = deque()
        pending_writes = deque()
        written = 0
        stage_seconds = {
            "read": 0.0,
            "reader_wait": 0.0,
            "preprocess": 0.0,
            "restore": 0.0,
            "postprocess_d2h": 0.0,
            "writer_wait": 0.0,
        }
        started_at = time.perf_counter()
        for chunk in run_chunks[:max_pending]:
            pending_reads.append(
                reader_executor.submit(
                    self.read_video_frames,
                    reader,
                    chunk,
                    raw_frame_count,
                    source_height,
                    source_width,
                    self.copy_stream is not None,
                    read_lead,
                )
            )
        try:
            for local_index, chunk in enumerate(run_chunks):
                self.check_stop()
                reader_wait_started_at = time.perf_counter()
                frames, read_seconds = pending_reads.popleft().result()
                reader_wait_seconds = time.perf_counter() - reader_wait_started_at
                next_read_index = local_index + max_pending
                if next_read_index < len(run_chunks):
                    pending_reads.append(
                        reader_executor.submit(
                            self.read_video_frames,
                            reader,
                            run_chunks[next_read_index],
                            raw_frame_count,
                            source_height,
                            source_width,
                            self.copy_stream is not None,
                            read_lead,
                        )
                    )

                stage_marks = [mark_stage(self.init_device)]
                restored = self.restore_frames(
                    frames[read_lead:],
                    chunk,
                    clip_latents,
                    output_height,
                    output_width,
                    pad_height,
                    pad_width,
                    stage_marks,
                )
                if antiphase:
                    shifted = self.restore_frames(
                        frames[:-1],
                        chunk,
                        clip_latents,
                        output_height,
                        output_width,
                        pad_height,
                        pad_width,
                        restorer=self.restorer_shifted,
                    )
                    # 在**浮点**上平均,而不是像离线配方那样 blend 两个已编码的 8bit 视频。
                    restored = blender.push(restored, shifted)
                    del shifted
                    # 第二遍不单独打点:把 restore 的结束点挪到它之后,两遍的算力都记在
                    # restore 头上(第二遍的 preprocess 也一并计入,它本来就是恢复的开销)。
                    stage_marks[-1] = mark_stage(self.init_device)
                    if chunk.index == len(chunks) - 1:
                        # 全片最后一块:carry 是最后一帧,移位流已经没有对应帧能与它配对
                        # (移位流整体比正相流短一帧),原样接在尾巴上。
                        tail = blender.flush()
                        if tail is not None:
                            restored = torch.cat([restored, tail], dim=1)
                # 用**全局已发出帧数**裁尾巴,而不是本 rank 的 written(段并行下它只算自己
                # 那段,会把中间段截没),也不是裸的 chunk.start(名义帧位,比实际多 3)。
                emitted_before = chunk.start - (0 if chunk.index == 0 else first_chunk_priming)
                if antiphase:
                    # 反相双跑把整条输出整体推迟一帧:首块少发一帧(它的末帧成了 carry),
                    # 之后每块都是「上一块的 carry + 本块前 n-1 帧」。所以全局帧位减一。
                    emitted_before = max(0, emitted_before - 1)
                restored = restored[:, : max(0, raw_frame_count - emitted_before)]
                output_frames = (restored[0].permute(0, 2, 3, 1) * 255).clamp_(0, 255).to(torch.uint8)
                cpu_frames, copy_complete = self.copy_frames_to_cpu(output_frames, self.copy_stream)
                stage_marks.append(copy_complete)

                if chunk.index < my_start:
                    # 预热块:只为把因果状态跑起来,输出丢弃(它归上一个 rank 写)
                    del cpu_frames
                    continue

                if len(pending_writes) >= max_pending:
                    self.finish_video_write(pending_writes.popleft(), stage_seconds, len(run_chunks))
                pending_writes.append(
                    PendingVideoWrite(
                        chunk_index=local_index,
                        frame_count=len(cpu_frames),
                        read_seconds=read_seconds,
                        reader_wait_seconds=reader_wait_seconds,
                        future=writer_executor.submit(
                            self.write_video_frames,
                            writer,
                            cpu_frames,
                            stage_marks,
                            self.init_device,
                        ),
                    )
                )
                written += len(cpu_frames)

                if self.progress_callback and seg_rank == 0:
                    # 只有 rank0 上报。按**本 rank 自己的**块序折算:各 rank 均分且末尾
                    # 汇合,本地进度是全局进度的良好估计,而且一定走得到 100%。
                    done = chunk.index - my_start + 1
                    total = max(1, my_end - my_start)
                    self.progress_callback(done / total * 100, 100)
            while pending_writes:
                self.finish_video_write(pending_writes.popleft(), stage_seconds, len(run_chunks))
        finally:
            reader_executor.shutdown(wait=True, cancel_futures=True)
            writer_executor.shutdown(wait=True)
            writer.close()
            if seg_world > 1:
                logger.info(f"SwiftVR seg_parallel: rank {seg_rank} closed writer {os.path.basename(writer_path)}")
            self._rank_local_collectives = outer_rank_local
            self._reset_restorers()

        elapsed = time.perf_counter() - started_at

        if seg_world > 1:
            # 汇合:所有 rank 的分片都已关闭落盘,才能拼。
            self._seg_barrier()
            if seg_rank != 0:
                logger.info(f"SwiftVR seg_parallel: rank {seg_rank} wrote {written} frames to {writer_path}")
                return {"video": None, "stats": {"frames": written, "seconds": elapsed}}
            missing = [path for path in part_paths if not os.path.isfile(path) or os.path.getsize(path) == 0]
            if missing:
                raise RuntimeError(f"SwiftVR seg_parallel: missing or empty segment parts {missing}")
            self._concat_segment_parts(part_paths, output_path)
            # 拼完立刻核帧数:段并行最容易错在边界与裁剪上,而这种错不会报异常,
            # 只会悄悄少几帧。宁可在这里炸,也别交付一个短了的视频。
            # 数**包**不解码:-count_frames 会把整段解一遍,4K 上实测要 28 秒;
            # 拼接是 -c copy,包数即帧数。
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", output_path],
                capture_output=True,
                text=True,
            )
            joined = int((probe.stdout or "0").strip() or 0)
            if joined != raw_frame_count:
                raise RuntimeError(f"SwiftVR seg_parallel: joined video has {joined} frames, expected {raw_frame_count}")
            logger.info(f"SwiftVR seg_parallel: joined {len(part_paths)} parts into {joined} frames")
            # written 到这里只是 rank0 自己那段的帧数(4 卡约 1/4)。下面的 stats 和吞吐
            # 日志描述的是整个请求,必须换成拼接后的总帧数,否则监控里的 frames/fps 会
            # 按卡数缩水。
            written = joined
            # 这里**不**删 parts_dir:清理放在下次运行开头(rank0 建目录前 rmtree)。
            # 之前在这里删,会把仍在写 mp4 trailer 的其他 rank 的分片抽走,
            # 报 "Unable to re-open ... for shifting data"。

        mux_audio_from_video(
            input_info.video_path,
            output_path,
            prefer_copy=self.config.get("audio_mux_prefer_copy", True),
            trim_to_shortest=False,
        )
        stats = {
            "frames": written,
            "seconds": elapsed,
            "fps": written / elapsed if elapsed else 0.0,
            "output": output_path,
            "stage_seconds": stage_seconds,
        }
        if GET_RECORDER_MODE():
            monitor_cli.lightx2v_worker_request_success.inc()
        logger.info(f"SwiftVR restored {written} frames to {output_path} at {stats['fps']:.2f} fps")
        logger.info("SwiftVR stage totals: " + ", ".join(f"{name}={seconds:.3f}s" for name, seconds in stage_seconds.items()))
        return {"video": None, "stats": stats}
