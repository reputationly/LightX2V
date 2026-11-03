import os
import shutil
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from fractions import Fraction

import av
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
from lightx2v.models.networks.swiftvr.antiphase import exchange_antiphase_carry
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.server.metrics import monitor_cli
from lightx2v.utils.envs import GET_DTYPE, GET_RECORDER_MODE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import is_main_process, mux_audio_from_video, save_to_image


@RUNNER_REGISTER("swiftvr")
class SwiftVRRunner(DefaultRunner):
    """Native LightX2V runner for SwiftVR image and video restoration."""

    supported_request_fields_by_task = {
        "sr": COMMON_REQUEST_FIELDS | {"image_path", "sr_ratio", "size", "target_short_edge", "video_path"},
    }

    # Two spatial shapes trigger dynamic compilation before serving requests.
    WARMUP_RESOLUTIONS = ((720, 1280), (2048, 1536))

    def __init__(self, config):
        parallel = config.get("parallel") or {}
        self.chunk_p_size = parallel.get("chunk_p_size", 1)
        if any(parallel.get(name, 1) != 1 for name in ("seq_p_size", "cfg_p_size", "tensor_p_size")):
            raise ValueError("SwiftVR supports chunk parallelism only.")
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if self.chunk_p_size != world_size:
            raise ValueError(f"SwiftVR requires parallel.chunk_p_size ({self.chunk_p_size}) to match world_size ({world_size}).")
        if config.get("cpu_offload"):
            raise NotImplementedError("SwiftVR does not support CPU offload yet.")
        if config.get("queue_size", 3) <= 0:
            raise ValueError("SwiftVR queue_size must be positive.")
        normalize_swiftvr_config(config)
        super().__init__(config)
        self.chunk_p_group = dist.group.WORLD if self.chunk_p_size > 1 else None
        self.is_main_process = is_main_process()
        self.copy_stream = torch.cuda.Stream(device=self.init_device) if self.init_device.type == "cuda" else None

        # 两个数,两件事,别混：
        #
        # restoration_strength 是**用户旋钮**。1.0 = 上游原行为;调低用于抑制「把源本身的
        # 缺陷(水波纹/噪点/压缩块)当成细节一起放大」——SwiftVR 在重退化数据上训过,喂它
        # 轻退化素材(AIGC 直出、干净拍摄)会过度生成。
        #
        # antiphase_strength_gain 是**引擎补偿**。反相平均是逐像素取两条流的均值,这个平均
        # 本身要削掉约三成高频(实测锐度 lap 2905 -> 1969),所以双跑路径得把残差按比例放大
        # 才能回到基线画质(1.40 实测补平:2876~2921)。它不代表「用户想要更强的恢复」。
        self.restoration_strength = float(self.config.get("restoration_strength", 1.0))
        self.antiphase = bool(self.config.get("antiphase_dual_pass"))
        gain = float(self.config.get("antiphase_strength_gain", 1.4)) if self.antiphase else 1.0
        # 视频路径吃补偿,图像路径**不能**吃:单张图没有时间轴,反相平均无从谈起,补偿加
        # 上去就是纯过锐(实测 s1.40 不做反相 lap 4349,是基线 2905 的 1.5 倍)。而
        # run_pipeline 是按输入类型**运行时**分流的,同一个部署会同时收到图和视频,靠约定
        # 隔离必然静默出错 —— 所以图像走 self.restoration_strength,视频走这个。
        self.video_strength = self.restoration_strength * gain
        # rank 0 暂存「上一 batch 从最后一个 rank 收到的块尾」(见 exchange_antiphase_carry)。
        self._antiphase_carry = None

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

        def build_restorer():
            return SwiftVRRestorer(
                autoencoder,
                self.model,
                prompt_embedding,
                overlap=self.config.get("dit_overlap", 0),
                reae_frame_batch_size=self.config.get("reae_frame_batch_size", 1),
            )

        self.restorer = build_restorer()
        # 反相双跑的第二条流。SwiftVRRestorer 只持有流式状态(encoder_state/decoder_state/
        # previous_latents),权重在 autoencoder 与 self.model 里 —— 所以再建一个实例是
        # **共享权重、独立状态**,不吃第二份显存。
        self.restorer_shifted = build_restorer() if self.antiphase else None
        self.config.lock()

    def _set_chunk_p_group(self, group):
        """两套流各有自己的 ReAE 因果状态,ring 组必须一起挂、一起摘。"""
        self.restorer.autoencoder.chunk_p_group = group
        if self.restorer_shifted is not None:
            self.restorer_shifted.autoencoder.chunk_p_group = group

    def _reset_restorers(self):
        for restorer in (self.restorer, self.restorer_shifted):
            if restorer is not None:
                restorer.reset()
        self._antiphase_carry = None

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
            overlap = self.restorer.transformer.overlap
            try:
                self.restorer.transformer.overlap = 0
                self._set_chunk_p_group(self.chunk_p_group)
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
                    # 用视频路径的 strength:正式请求的残差乘的就是这个系数。
                    restored = self.restorer.restore_chunk(video, chunk, clip_latents, self.video_strength)
                    del video, restored
            finally:
                self.restorer.transformer.overlap = overlap
                self._set_chunk_p_group(None)
                # 移位流走同一批权重、同一组形状,编译缓存与它共用,不必单独预热一遍;
                # 但它的因果状态必须干净地进正式请求。
                self._reset_restorers()

        logger.info("[Warmup] Warmup completed")
        self._maybe_freeze_gc()

    def _oriented_target(self, source_height: int, source_width: int) -> tuple[int, int]:
        """配置里的标称输出档位,按源的朝向配对长短边。

        与 SeedVR2 的 `_oriented_target`(seedvr_runner.py:220)同语义:config 写的
        sr_tier_size [1080, 1920] 表示的是「1080P 这一档」,不是死的高×宽 —— 竖版源要出
        1080×1920、横版出 1920×1080、方形塌到短边(横竖都救不了方的)。

        返回 (0, 0) 表示没配档位,调用方退回纯 sr_ratio 放大。
        """
        tier_height, tier_width = self.config.get("sr_tier_size") or (1080, 1920)
        target_height, target_width = int(tier_height or 0), int(tier_width or 0)
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

        这条是兜底护栏,拦的是**显式 size**:标称档位那条路已经被 `_oriented_target`
        封住了,但请求可以直接点名尺寸,而 SwiftVR 的显存随像素数线性涨(实测单卡 1080p
        15.8G / 2K 19.8G / 4K 32.5G,40G 卡上 4K 已是上限)。没有这道闸,一个 8K 的 size
        就能把整台机器的卡打爆。

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
        """定输出尺寸。三条路各有各的封顶,别把它们混起来看。

        ⚠️ sr_ratio 这条路**必须**封顶,这是现网 SR 的既有契约,不是防呆:网关前端
        固定下发 sr_ratio=4.0(new-api `VIDEO_SR_RATIO_UNCAPPED`),注释原话是"一个够不着
        的上限",指望引擎拿 config 档位去 min 掉它 —— SeedVR2 靠 seedvr_runner.py:241
        那个 min 兜住。SwiftVR 早先没有这层,1344×768 的源乘 4 会去出 5376×3072(≈5.3K),
        远超 4K/33G 的实测上限,必 OOM,而且不报参数错、跑起来才炸。

        size 这条路是 SwiftVR 精确点名尺寸的能力(SeedVR2 的 size 还要再过
        match_target_size 的裁切/缩放),一份配置串行吃 1080p/2K/4K 靠的就是它,所以
        **不能**用档位去封 —— 那会把 2K/4K 一起封死。它只过 `_clamp_to_ceiling` 的显存护栏。

        target_short_edge 是给网关用的第三条路:只给一个目标短边,输出按**源的真实画幅**
        等比放大。它存在的理由是「只有引擎知道源有多大」—— 网关手里只有用户选的比例
        标签,而标签和实际画幅并不相等(H3 的 768P/16:9 实际出 1344x768 = 1.75,不是
        1.778;wan 的 720P 是 1280x720 = 1.778),照标签算 size 会带约 1.6% 的横向拉伸,
        且档位越高越明显。按短边对齐则画幅零形变、短边精确命中目标档,1080P/2K/4K 共用
        同一个机制,网关只需下发一个数字。

        三条路的优先级:size(精确尺寸,调用方完全指定) > target_short_edge
        (指定档位、画幅随源) > sr_ratio(倍率,按部署档位封顶)。
        """
        if input_info.size:
            if len(input_info.size) != 2:
                raise ValueError(f"SwiftVR size must be [height, width], got {input_info.size}")
            height, width = input_info.size
        elif input_info.target_short_edge > 0:
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
    def _antiphase_frames(chunk, raw_frame_count: int, frames_to_trim: int) -> int:
        """反相双跑下本块要写出的帧数。

        配对是「本块 direct[i] 配 shifted[i+1],块末帧留给下一块」,而移位流那一帧来自
        **前一个源帧**,所以整条输出比直相前移一帧:窗口两端各减一,起点夹到 0(首块
        没有上一块可配,原地不动、末尾少写一帧),末块 flush 把最后一帧原样补回来。
        合起来正好一帧不少 —— 成片帧数 == 源帧数。
        """
        output = chunk.output_range(raw_frame_count, frames_to_trim)
        start = max(0, output.start - 1)
        stop = min(raw_frame_count, output.stop - 1 + (1 if chunk.is_last else 0))
        return stop - start

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

    @staticmethod
    def read_video_frames(reader, chunk, raw_frame_count: int, source_height: int, source_width: int, pin_memory: bool, lead: int = 0):
        """读一块。``lead>0`` 时向前多读几帧,供反相双跑的移位流用。

        反相双跑要的两条流是 [start, start+n) 与 [start-1, start+n-1),并集恰好是
        [start-1, start+n) —— 多读一帧就够,不必读两遍。块首越界的负号夹到 0,
        等价于「复制首帧插到最前面」,与离线配方一致。
        """
        indices = [min(max(index, 0), raw_frame_count - 1) for index in range(chunk.start - lead, chunk.start + chunk.frame_count)]
        frames = reader.get_batch(indices)
        if not torch.is_tensor(frames):
            frames = torch.from_dlpack(frames.to_dlpack())
        frames = frames[:, :source_height, :source_width].permute(0, 3, 1, 2)
        if pin_memory:
            frames = frames.pin_memory()
        return frames

    def preprocess_frames(self, frames, height: int, width: int):
        # Transfer only byte pixels; convert dtype and layout on the target device.
        frames = frames.to(device=self.init_device, non_blocking=frames.is_pinned())
        frames = frames.to(dtype=GET_DTYPE(), memory_format=torch.contiguous_format)
        if frames.shape[-2:] != (height, width):
            mode = self.config.get("upscale_mode", "bilinear")
            interpolate_args = {"align_corners": False} if mode in {"linear", "bilinear", "bicubic", "trilinear"} else {}
            frames = F.interpolate(frames, size=(height, width), mode=mode, **interpolate_args)
        frames.div_(255)
        pad_height, pad_width = (-height) % 32, (-width) % 32
        if pad_height or pad_width:
            frames = F.pad(frames, (0, pad_width, 0, pad_height))
        return frames.unsqueeze(0)

    @staticmethod
    def copy_frames_to_cpu(frames, copy_stream):
        if copy_stream is None:
            return frames.cpu(), None

        cpu_frames = torch.empty(frames.shape, dtype=frames.dtype, device="cpu", pin_memory=True)
        copy_stream.wait_stream(torch.cuda.current_stream(frames.device))
        with torch.cuda.stream(copy_stream):
            cpu_frames.copy_(frames, non_blocking=True)
            frames.record_stream(copy_stream)
            copy_complete = torch.cuda.Event()
            copy_complete.record(copy_stream)
        return cpu_frames, copy_complete

    def open_video_writer(self, output_path: str, fps: float, height: int, width: int, frame_counts=None):
        """Write one video, or a numbered segment for each count in frame_counts."""
        quality = self.config.get("quality", 60)
        codec = self.config.get("video_codec", "libx265")
        threads = min(16, max(1, len(os.sched_getaffinity(0)) // self.chunk_p_size)) if self.chunk_p_size > 1 else None
        codec_options = {"crf": str(round((100 - quality) * 51 / 100))}
        if codec == "libx265":
            codec_options["x265-params"] = "log-level=warning"
            if threads is not None:
                codec_options["x265-params"] += f":pools={threads}"
        elif threads is not None:
            codec_options["threads"] = str(threads)

        preset = self.config.get("ffmpeg_preset")
        if preset:
            codec_options["preset"] = preset

        pixel_format = "yuv444p" if self.config.get("save_format") == "yuv444p" else "yuv420p"

        def write_video(path):
            if self.chunk_p_size == 1:
                ffmpeg_params = []
                for name, value in codec_options.items():
                    ffmpeg_params.extend([f"-{name}", value])
                if codec == "libx265":
                    ffmpeg_params.extend(["-tag:v", "hvc1"])
                with closing(
                    imageio.get_writer(
                        path,
                        fps=fps,
                        codec=codec,
                        pixelformat=pixel_format,
                        macro_block_size=None,
                        ffmpeg_params=ffmpeg_params,
                    )
                ) as writer:
                    while True:
                        writer.append_data((yield))
            else:
                with av.open(path, "w") as container:
                    stream = container.add_stream(
                        codec,
                        rate=Fraction(f"{fps:.02f}"),
                        options=codec_options,
                        width=width,
                        height=height,
                        pix_fmt=pixel_format,
                    )
                    if codec == "libx265":
                        stream.codec_context.codec_tag = "hvc1"
                    reformatter = av.video.reformatter.VideoReformatter()
                    frames_written = 0
                    try:
                        while True:
                            array = yield
                            # Share the pinned RGB buffer before color conversion.
                            frame = av.VideoFrame.from_numpy_buffer(array, format="rgb24")
                            frame = reformatter.reformat(frame, format=pixel_format, interpolation="BICUBIC", threads=min(4, threads))
                            container.mux(stream.encode(frame))
                            frames_written += 1
                    except GeneratorExit:
                        if frames_written:
                            packets = stream.encode()
                            if codec == "libx265" and frames_written < 3:
                                # Short x265 segments still need the decode delay declared in their headers.
                                decoder = av.CodecContext.create("hevc", "r")
                                decoder.thread_count = 1
                                decoder.extradata = stream.codec_context.extradata
                                decoder.open()
                                for packet in packets:
                                    packet.dts = packet.pts - decoder.reorder_depth
                            container.mux(packets)

        def write_segments():
            for segment, frame_count in enumerate(frame_counts):
                # Each generator owns and releases one segment's encoder and packets.
                with closing(write_video(output_path % segment)) as writer:
                    next(writer)
                    for _ in range(frame_count):
                        writer.send((yield))
            yield

        writer = write_video(output_path) if frame_counts is None else write_segments()
        next(writer)
        return writer

    @staticmethod
    def write_video_frames(writer, frames, copy_complete):
        if copy_complete is not None:
            copy_complete.synchronize()
        for frame in frames.numpy():
            writer.send(frame)

    def join_video_chunks(self, work_dir, chunk_count, output_path, source_path):
        manifest = os.path.join(work_dir, "chunks.txt")
        with open(manifest, "w") as file:
            file.write("ffconcat version 1.0\n")
            for index in range(chunk_count):
                segment, rank = divmod(index, self.chunk_p_size)
                file.write(f"file '{rank:03d}-{segment:08d}.mp4'\n")

        result = mux_audio_from_video(
            source_path,
            manifest,
            output_path=output_path,
            prefer_copy=self.config.get("audio_mux_prefer_copy", True),
            trim_to_shortest=False,
        )
        if result is None:
            raise RuntimeError("SwiftVR failed to assemble the output video.")

    @ProfilingContext4DebugL1(
        "RUN pipeline",
        recorder_mode=GET_RECORDER_MODE(),
        metrics_func=monitor_cli.lightx2v_worker_request_duration,
        metrics_labels=["SwiftVRRunner"],
        profile_memory=True,
    )
    @torch.inference_mode()
    def run_pipeline(self, input_info):
        self.input_info = input_info
        try:
            if self.is_main_process and GET_RECORDER_MODE():
                monitor_cli.lightx2v_worker_request_count.inc()
            if self.resolve_input_kind(input_info) == "image":
                self.check_stop()
                return self.run_image_pipeline(input_info) if self.is_main_process else {}
            self._set_chunk_p_group(self.chunk_p_group)
            return self.run_video_pipeline(input_info)
        finally:
            self._set_chunk_p_group(None)
            self._reset_restorers()

    def run_image_pipeline(self, input_info):
        if not input_info.return_result_tensor and not input_info.save_result_path:
            raise ValueError("SwiftVR image restoration requires `save_result_path` unless the image is returned in memory.")

        frames, source_height, source_width = self.read_image_frame(input_info.image_path)
        output_height, output_width = self.resolve_output_size(input_info, source_height, source_width)
        clip_length = self.config.get("clip_len", 24)
        chunk = build_video_chunks(1, clip_length)[0]
        clip_latents = clip_length // 4

        output_path = None if input_info.return_result_tensor else os.path.abspath(input_info.save_result_path)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        started_at = time.perf_counter()
        video = self.preprocess_frames(frames, output_height, output_width)
        # 单张图没有时间轴,反相双跑无从谈起 —— 图像固定走正相这一条流,而且用**基准**
        # strength(不吃 antiphase_strength_gain,否则就是纯过锐)。
        restored = self.restorer.restore_chunk(video, chunk, clip_latents, self.restoration_strength)[..., :output_height, :output_width]
        del video
        images = restored[0].permute(0, 2, 3, 1).contiguous()
        if input_info.return_result_tensor:
            images = images.to(device="cpu", dtype=torch.float32)
        else:
            save_to_image(images, output_path)
            images = None

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

    def run_video_pipeline(self, input_info):
        reader_executor = writer_executor = writer = None
        work_dir = None
        pending_reads = deque()
        pending_writes = deque()
        started_at = time.perf_counter()
        if not input_info.save_result_path:
            raise ValueError("SwiftVR video restoration requires `save_result_path`.")
        if input_info.return_result_tensor:
            raise ValueError("SwiftVR video restoration does not support `return_result_tensor`.")

        reader = VideoReader(input_info.video_path)
        raw_frame_count = len(reader)
        first_frame = reader[0]
        source_height = first_frame.shape[0] // 8 * 8
        source_width = first_frame.shape[1] // 8 * 8
        output_height, output_width = self.resolve_output_size(input_info, source_height, source_width, require_even=True)
        fps = self.config.get("fps") or reader.get_avg_fps() or 30
        clip_length = self.config.get("clip_len", 24)
        chunks = build_video_chunks(padded_frame_count(raw_frame_count), clip_length)
        clip_latents = clip_length // 4
        frames_to_trim = self.restorer.autoencoder.autoencoder.frames_to_trim
        rank = dist.get_rank(self.chunk_p_group) if self.chunk_p_group is not None else 0
        # Idle ranks in the final batch still participate in ReAE boundary exchanges.
        local_chunks = [chunks[min(start + rank, len(chunks) - 1)] for start in range(0, len(chunks), self.chunk_p_size)]
        output_path = os.path.abspath(input_info.save_result_path)
        antiphase = self.restorer_shifted is not None
        # 双跑的两条流是 [start, start+n) 与 [start-1, start+n-1),并集多一帧就够。
        read_lead = 1 if antiphase else 0

        try:
            try:
                if self.is_main_process:
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    if self.chunk_p_group is not None:
                        work_dir = tempfile.mkdtemp(prefix="swiftvr-", dir=os.path.dirname(output_path))
                    else:
                        writer = self.open_video_writer(output_path, fps, output_height, output_width)
                reader_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-reader")
                writer_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swiftvr-writer")
                max_pending = self.config.get("queue_size", 3)
                # 与上游同款分批:块并行 4 帧一批、单卡 clip_len+4。双跑以前用 0(整块)
                # 是为了拿到块尾做 ring carry,代价是 output_layers 一次性吃下整块,
                # 4C/4K 实测多占 5.3GiB(peak_allocated 反而更低,差额全是 allocator 缓存
                # 的大块 transient)。融合算法本身是流式的,分批一样能算对。
                output_batch_size = 4 if self.chunk_p_group is not None else clip_length + 4
                # Keep the write queue's frame budget proportional to the original chunk budget.
                # 整块解码时一块只有一次写出。
                writes_per_chunk = 1 if output_batch_size == 0 else (clip_length + output_batch_size - 1) // output_batch_size
                max_pending_writes = max_pending * writes_per_chunk

                def chunk_frames(chunk):
                    return self._antiphase_frames(chunk, raw_frame_count, frames_to_trim) if antiphase else len(chunk.output_range(raw_frame_count, frames_to_trim))

                def submit_read(chunk):
                    return reader_executor.submit(
                        self.read_video_frames,
                        reader,
                        chunk,
                        raw_frame_count,
                        source_height,
                        source_width,
                        self.copy_stream is not None,
                        read_lead,
                    )

                for chunk in local_chunks[:max_pending]:
                    pending_reads.append(submit_read(chunk))
                if self.chunk_p_group is not None:
                    directories = [work_dir]
                    dist.broadcast_object_list(directories, src=0, group=self.chunk_p_group)
                    work_dir = directories[0]
                    if rank < len(chunks):
                        frame_counts = [chunk_frames(chunk) for chunk in chunks[rank :: self.chunk_p_size]]
                        writer = self.open_video_writer(os.path.join(work_dir, f"{rank:03d}-%08d.mp4"), fps, output_height, output_width, frame_counts)

                blender = AntiphaseBlender() if antiphase else None
                for local_index, chunk in enumerate(local_chunks):
                    self.check_stop()
                    # 一块一块来:进下一块之前把上一块的 transient 还给驱动。不还的话
                    # allocator 会把它们留着复用,4C/4K 上 rank0 的 reserved 会一路涨到
                    # 36~38GiB(而真正的活张量只有 24.7GiB)。代价是每块一次 empty_cache。
                    if local_index and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    active = local_index * self.chunk_p_size + rank < len(chunks)
                    frames = pending_reads.popleft().result()
                    next_read = local_index + max_pending
                    if next_read < len(local_chunks):
                        pending_reads.append(submit_read(local_chunks[next_read]))

                    if antiphase:
                        # 直相用 frames[1:]、移位流用 frames[:-1]:同一段画面差一帧的两种
                        # 切法,多读的那一帧就是位移量。
                        video = self.preprocess_frames(frames[1:], output_height, output_width)
                        video_shifted = self.preprocess_frames(frames[:-1], output_height, output_width)
                        del frames
                        # 六步固定顺序:直相 encode → 移位 encode → 直相 predict → 移位
                        # predict → 直相 decode → 移位 decode。**所有 rank(含 inactive 的
                        # 重复块)都按这个顺序执行**,两套 ring 上每一次 ReAE 边界交换才能
                        # 一一配对 —— 顺序一换,rank A 的直相交换就配上 rank B 的移位交换。
                        direct_input, direct_offset = self.restorer.encode_chunk(video, chunk, clip_latents)
                        shifted_input, shifted_offset = self.restorer_shifted.encode_chunk(video_shifted, chunk, clip_latents)
                        del video, video_shifted
                        if active:
                            direct_latents = direct_input - self.video_strength * self.model.predict(direct_input, self.restorer.transformer.condition, direct_offset)
                            shifted_latents = shifted_input - self.video_strength * self.model.predict(shifted_input, self.restorer_shifted.transformer.condition, shifted_offset)
                        else:
                            direct_latents, shifted_latents = direct_input, shifted_input
                        del direct_input, shifted_input
                        # 直相整块分批解完(每批过后 output_layers 的临时张量就释放),拿到
                        # 块尾后立刻做 ring carry 交换;移位流随后逐批解、逐批融合写出。
                        # 这样同一时刻只驻留一条流的整块,而不是两条。
                        direct_frames = list(self.restorer.decode_chunk(direct_latents, chunk, output_batch_size))
                        del direct_latents
                        if self.chunk_p_group is not None:
                            # inactive 的重复块也要发同形零张量:ring 的每一次交换在所有
                            # rank 上必须一一配对,少发一次其余 rank 就卡在 P2P 上。
                            last_batch = direct_frames[-1]
                            carry = last_batch[:, -1:] if active else last_batch.new_zeros((last_batch.shape[0], 1, *last_batch.shape[2:]))
                            chunk_carry, self._antiphase_carry = exchange_antiphase_carry(carry, self.chunk_p_group, self._antiphase_carry)
                            blender.carry = chunk_carry
                        if not active:
                            # 两条流都要跑完整块:causal 层的 ring 交换一一配对。
                            for _ in self.restorer_shifted.decode_chunk(shifted_latents, chunk, output_batch_size):
                                pass
                            del direct_frames, shifted_latents
                            continue

                        def fused_batches(direct_batches, shifted_latents, chunk, blender, batch_size):
                            """按批融合:blender 的 carry 语义在批之间同样成立(流式)。"""
                            for index, shifted_batch in enumerate(self.restorer_shifted.decode_chunk(shifted_latents, chunk, batch_size)):
                                yield blender.push(direct_batches[index], shifted_batch)
                                direct_batches[index] = None  # 这一批已经融合完,立刻放手
                            if chunk.index == len(chunks) - 1:
                                # 全片最后一块:carry 是最后一帧,移位流没有对应帧可配
                                # (移位流整体比正相流短一帧),原样接在尾巴上。
                                tail = blender.flush()
                                if tail is not None:
                                    yield tail

                        restored_frames = fused_batches(direct_frames, shifted_latents, chunk, blender, output_batch_size)
                        del shifted_latents
                    else:
                        video = self.preprocess_frames(frames, output_height, output_width)
                        del frames

                        model_input, offset = self.restorer.encode_chunk(video, chunk, clip_latents)
                        del video
                        latents = model_input - self.video_strength * self.model.predict(model_input, self.restorer.transformer.condition, offset) if active else model_input
                        restored_frames = self.restorer.decode_chunk(latents, chunk, output_batch_size)
                        del model_input, latents
                        if not active:
                            del restored_frames
                            continue

                    remaining_frames = chunk_frames(chunk)
                    for restored in restored_frames:
                        if len(pending_writes) >= max_pending_writes:
                            pending_writes.popleft().result()
                        restored = restored[:, :remaining_frames, :, :output_height, :output_width]
                        # Consume the decoded pixel buffer; ReAE already clamps it to [0, 1].
                        output_frames = restored[0].permute(0, 2, 3, 1).mul_(255).to(dtype=torch.uint8, memory_format=torch.contiguous_format)
                        cpu_frames, copy_complete = self.copy_frames_to_cpu(output_frames, self.copy_stream)
                        del output_frames, restored
                        pending_writes.append(writer_executor.submit(self.write_video_frames, writer, cpu_frames, copy_complete))
                        remaining_frames -= len(cpu_frames)
                        if remaining_frames == 0:
                            break
                    if self.is_main_process and self.progress_callback:
                        self.progress_callback(min(chunk.index + self.chunk_p_size, len(chunks)) / len(chunks) * 100, 100)
                    del restored_frames
                while pending_writes:
                    pending_writes.popleft().result()
                # 解码/写出的 transient 已经用完:把 allocator 缓存的那部分还给驱动。
                # 不还的话它一直挂到请求结束(nvidia-smi 上 4C/4K 的峰值就出现在这一段,
                # 而 decode 本身只要 28.2G)。收尾的 join/mux 是纯 CPU/ffmpeg,不需要显存。
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            finally:
                if reader_executor is not None:
                    reader_executor.shutdown(wait=True, cancel_futures=True)
                if writer_executor is not None:
                    writer_executor.shutdown(wait=True)
                if writer is not None:
                    writer.close()

            if self.chunk_p_group is not None:
                dist.barrier(group=self.chunk_p_group)
            if not self.is_main_process:
                return {}
            if work_dir is not None:
                self.join_video_chunks(work_dir, len(chunks), output_path, input_info.video_path)
            else:
                mux_audio_from_video(input_info.video_path, output_path, prefer_copy=self.config.get("audio_mux_prefer_copy", True), trim_to_shortest=False)
            elapsed = time.perf_counter() - started_at
            stats = {
                "frames": raw_frame_count,
                "seconds": elapsed,
                "fps": raw_frame_count / elapsed if elapsed else 0.0,
                "output": output_path,
            }
            if GET_RECORDER_MODE():
                monitor_cli.lightx2v_worker_request_success.inc()
            logger.info(f"SwiftVR restored {raw_frame_count} frames to {output_path} at {stats['fps']:.2f} fps")
            return {"video": None, "stats": stats}
        finally:
            if self.is_main_process and work_dir is not None:
                shutil.rmtree(work_dir)
