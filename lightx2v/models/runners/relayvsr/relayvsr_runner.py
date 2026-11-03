import os
import shutil
import tempfile
import time
from contextlib import closing
from fractions import Fraction
from itertools import islice, repeat
from pathlib import Path
from typing import ClassVar

import av
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from loguru import logger

from lightx2v.common.kvcache.manager import KVCacheManager
from lightx2v.models.networks.relayvsr.attention import LOCAL_ATTENTION_BACKENDS
from lightx2v.models.networks.relayvsr.exact_chunk import NcclKVExchange, build_plan, kv_transfer_plan, split_ranks
from lightx2v.models.networks.relayvsr.geometry import RelayVSRGeometry
from lightx2v.models.networks.relayvsr.model import RelayVSRModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.runners.request_fields import COMMON_REQUEST_FIELDS
from lightx2v.models.schedulers.relayvsr.scheduler import RelayVSRScheduler
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.profiler import ProfilingContext4DebugL1
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import mux_audio_from_video
from lightx2v_platform.base.global_var import AI_DEVICE

from .video_io import AsyncVideoWriter, prefetch_frames


def validate_config(config):
    backend = config.get("attention_backend", "triton")
    if backend not in LOCAL_ATTENTION_BACKENDS:
        raise ValueError(f"RelayVSR attention_backend must be one of {LOCAL_ATTENTION_BACKENDS}, got {backend!r}")
    if config.get("cpu_offload") and config.get("offload_granularity", "block") != "block":
        raise ValueError("RelayVSR supports block weight offload only")
    parallel = config.get("parallel") or {}
    chunk_p_size = int(parallel.get("chunk_p_size", 1))
    if chunk_p_size > 1:
        # 精确块并行:每卡算自己那段连续窗口,模型复制,不走 mesh。
        if int(parallel.get("seq_p_size", 1)) != 1:
            raise ValueError("RelayVSR chunk_p_size is exclusive with seq_p_size")
        if config.get("cpu_offload"):
            raise ValueError("RelayVSR chunk_p_size is exclusive with cpu_offload")
        if config.get("lazy_load"):
            raise ValueError("RelayVSR chunk_p_size is exclusive with lazy_load")
    elif parallel.get("seq_p_size", 1) not in (1, 2):
        raise ValueError("RelayVSR sequence parallelism supports 1 or 2 GPUs")
    for key, default, minimum in (("clip_len", 16, 2), ("queue_size", 16, 1), ("encoder_threads", 16, 1)):
        if config.get(key, default) < minimum:
            raise ValueError(f"RelayVSR {key} must be >= {minimum}")


class RelayVSRRunner(DefaultRunner):
    """Shared streaming lifecycle; registered subclasses fix the checkpoint geometry."""

    geometry: ClassVar[RelayVSRGeometry]
    supported_request_fields_by_task: ClassVar[dict] = {"sr": COMMON_REQUEST_FIELDS | {"video_path", "image_path", "sr_ratio"}}

    WARMUP_RESOLUTIONS = ((768, 1024), (768, 1344))

    def __init__(self, config):
        if config.get("native_scale", self.geometry.scale) != self.geometry.scale:
            raise ValueError(f"{config['model_cls']} requires native_scale={self.geometry.scale}")
        for key, value in (("infer_steps", 1), ("guidance_scale", 1.0), ("enable_cfg", False), ("fps", None)):
            config.setdefault(key, value)
        validate_config(config)
        super().__init__(config)
        if AI_DEVICE != "cuda" or not torch.get_device_module(torch.device(AI_DEVICE)).is_available() or GET_DTYPE() not in (torch.bfloat16, torch.float16):
            raise ValueError("RelayVSR local attention requires a CUDA-compatible accelerator with BF16 or FP16")
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.main_rank = not dist.is_initialized() or dist.get_rank() == 0
        parallel = config.get("parallel") or {}
        self.chunk_p_size = int(parallel.get("chunk_p_size", 1))
        # 块级数据并行:模型是复制的,各 rank 直接和 WORLD 通信(与 SwiftVR 的 chunk_p 同款)。
        self.chunk_p_group = dist.group.WORLD if self.chunk_p_size > 1 else None
        self.seq_p_group = config["device_mesh"].get_group(mesh_dim="seq_p") if config.get("seq_parallel") else None
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        expected = self.chunk_p_size if self.chunk_p_size > 1 else (config.get("parallel") or {}).get("seq_p_size", 1)
        if world_size != expected:
            raise ValueError(f"RelayVSR world size must equal parallel.chunk_p_size/seq_p_size (expected {expected}, got {world_size})")

    def init_scheduler(self):
        self.scheduler = RelayVSRScheduler(self.config, self.geometry)

    @torch.no_grad()
    def init_modules(self):
        started = time.perf_counter()
        self.model = RelayVSRModel(self.config, self.geometry, self.seq_p_group)
        self.model.set_scheduler(self.scheduler)
        torch.cuda.synchronize()
        self.config.lock()
        logger.info("{} native {}x model loaded in {:.2f}s", self.config["model_cls"], self.geometry.scale, time.perf_counter() - started)

    def warmup_resolutions(self):
        configured = self.config.get("warmup_resolutions")
        return [tuple(pair) for pair in configured] if configured else list(self.WARMUP_RESOLUTIONS)

    @torch.no_grad()
    def run_warmup(self):
        if self.config["task"] != "sr":
            raise NotImplementedError("RelayVSR warmup supports the native SR runner only")
        if self.config.get("lazy_load", False):
            raise NotImplementedError("RelayVSR warmup does not support lazy loading")
        if self.chunk_p_size > 1:
            return self._warmup_exact()
        started = time.perf_counter()
        # Fill the causal Wan cache and overwrite every slot once. The Flash model's
        # condition, first-frame and history branches follow the normal stream.
        references = 2 * self.model.wan_config["temporal_context_frames"]
        frames = (references - 1) * (self.config.get("clip_len", 16) - 1) + 1
        for height, width in self.warmup_resolutions():
            frame = prediction = pixels = None
            logger.info("RelayVSR warmup: {}x{}, {} reference frames", height, width, references)
            try:
                frame = torch.zeros((3, height, width), dtype=torch.uint8, pin_memory=True)
                with closing(self.restore_frames(repeat(frame, frames), 42)) as predictions:
                    for prediction in predictions:
                        for scale in (2, 4) if self.geometry.scale == 4 else (2,):
                            pixels = self.postprocess(prediction, (height * scale, width * scale))
                torch.cuda.synchronize()
            finally:
                frame = prediction = pixels = None
                self.input_info = None
        logger.info("RelayVSR warmup cost {:.2f}s", time.perf_counter() - started)
        self._maybe_freeze_gc()

    @torch.no_grad()
    def _warmup_exact(self):
        """Warm both compiled halves and every ``visible`` variant on the real rank layout.

        ``2 * context`` references give each rank enough references that the point-to-point
        exchange is exercised and the ``visible = 1...context`` attention variants are built.
        """
        started = time.perf_counter()
        fps = Fraction(float(self.config.get("fps") or 24)).limit_denominator(1001)
        context = int(self.model.wan_config["temporal_context_frames"])
        world = dist.get_world_size(self.chunk_p_group) if self.chunk_p_group is not None else 1
        # Every rank must own at least ``context + 1`` references, otherwise the warm-up only ever
        # runs the neighbour-dependent half of each layer and the first real request re-traces the rest.
        references = max(2 * context, world * (context + 1))
        frames = int(self.config.get("warmup_frames") or (references - 1) * (self.config.get("clip_len", 16) - 1) + 1)
        for height, width in self.warmup_resolutions():
            logger.info("RelayVSR warmup: {}x{}, {} frames exact chunk", height, width, frames)
            directory = Path(tempfile.mkdtemp(prefix="relayvsr-warmup-"))
            try:
                source = directory / "in.mp4"
                self._write_synthetic_video(source, height, width, frames, fps)
                self._run_video_exact(source, directory / "out.mp4", fps, (height * self.geometry.scale, width * self.geometry.scale), 42)
                torch.cuda.synchronize()
            finally:
                self.input_info = None
                shutil.rmtree(directory, ignore_errors=True)
        logger.info("RelayVSR warmup cost {:.2f}s", time.perf_counter() - started)
        self._maybe_freeze_gc()

    @staticmethod
    def _write_synthetic_video(path, height, width, frames, rate):
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=rate)
            stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
            stream.options = {"crf": "0", "preset": "ultrafast"}
            for index in range(frames):
                frame = av.VideoFrame.from_numpy_buffer(pixels, format="rgb24")
                frame.pts, frame.time_base = index, 1 / rate
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)

    def output_size(self, input_info, height, width):
        return self.geometry.output_size(input_info, height, width)

    def preprocess(self, frame):
        height, width = frame.shape[-2:]
        x = frame[None].to(self.device, non_blocking=frame.is_pinned()).to(GET_DTYPE())
        x = x.div(127.5).sub(1.0)
        padded_height, padded_width = self.geometry.padded_size(height, width)
        return F.pad(x, (0, padded_width - width, 0, padded_height - height), mode="replicate")

    @staticmethod
    def postprocess(prediction, target):
        if prediction.shape[-2:] != target:
            prediction = F.interpolate(prediction.float().clamp(-1, 1), size=target, mode="bicubic", align_corners=False, antialias=True)
        return prediction[0].add(1).mul_(127.5).clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous()

    def init_run(self, height, width, seed, with_wan_cache=True):
        self.scheduler.prepare(seed, self.device)
        wan = self.model.wan_config
        self.wan_cache_manager = KVCacheManager(
            {
                "num_layers": wan["num_layers"],
                "num_heads": wan["num_attention_heads"],
                "dim": wan["num_attention_heads"] * wan["attention_head_dim"],
            },
            device=self.device,
        )
        # 精确块并行不走 frame ring:每层的 K/V 由本层 stage1 现场给出,不需要跨 reference 的环形缓冲。
        self.wan_cache = (
            self.wan_cache_manager.create_self_attn_kv_cache(
                "wan",
                (height // self.geometry.wan_stride) * (width // self.geometry.wan_stride) * wan["temporal_context_frames"],
                kv_cache_scheme="frame_ring",
                num_heads=wan["num_attention_heads"] // self.model.parallel.size,
            )
            if with_wan_cache
            else None
        )
        self.flash_model_cache_manager = KVCacheManager({"num_layers": 10, "num_heads": 2, "dim": 128}, device=self.device)
        flash_model_heads = 2 if self.model.flash_model_transformer.query_parallel else 2 // self.model.parallel.size
        self.flash_model_history = self.flash_model_cache_manager.create_self_attn_kv_cache(
            "flash_model_video", (height // self.geometry.latent_stride) * (width // self.geometry.latent_stride), kv_cache_scheme="static", num_heads=flash_model_heads
        )
        self.reference_count = 0

    def end_run(self):
        self.scheduler.clear()
        self.model.clear()
        self.wan_cache = self.flash_model_history = None
        self.wan_cache_manager = self.flash_model_cache_manager = None

    def reference(self, frame):
        lr = self.preprocess(frame)
        self.scheduler.prepare_latents(*lr.shape[-2:], self.device)
        self.scheduler.step_pre(0)
        self.model.infer(self.scheduler, lr, self.wan_cache, self.reference_count)
        self.scheduler.step_post()
        self.reference_count += 1
        latent = self.scheduler.latents[:, :, 0].to(GET_DTYPE())
        return self.model.flash_model_prepare_condition(latent)

    def restore_frames(self, source, seed):
        """Yield cropped native predictions; overlapping endpoints are emitted once."""
        iterator = iter(source)
        first = next(iterator, None)
        if first is None:
            raise ValueError("RelayVSR cannot restore an empty stream")
        shape = first.shape
        height, width = shape[-2:]
        try:
            padded_height, padded_width = self.geometry.padded_size(height, width)
            self.init_run(padded_height, padded_width, seed)
            start, start_condition = first, self.reference(first)
            first_window = True
            while True:
                window = [start, *islice(iterator, self.config.get("clip_len", 16) - 1)]
                if any(frame.shape != shape for frame in window):
                    raise ValueError("All RelayVSR input frames must have the same dimensions")
                if len(window) == 1 and not first_window:
                    return
                self.check_stop()
                end_condition = self.reference(window[-1]) if len(window) > 1 else start_condition
                conditions = [(torch.cat((a[0], b[0]), 0), torch.cat((a[1], b[1]), 0)) for a, b in zip(start_condition, end_condition)]
                self.flash_model_history.reset()
                for offset, frame in enumerate(window):
                    self.check_stop()
                    lr = self.preprocess(frame)
                    output = self.model.flash_model_decode(lr, conditions, self.flash_model_history, offset)
                    if first_window or offset > 0:
                        yield output[..., : height * self.geometry.scale, : width * self.geometry.scale]
                first_window = False
                start, start_condition = window[-1], end_condition
                if len(window) < self.config.get("clip_len", 16):
                    return
        finally:
            self.end_run()

    # ---- 精确块并行(chunk_p_size>1):reference 逐层同步,窗口独立解码 ----

    def _video_frame_count(self, source):
        with av.open(str(source)) as container:
            stream = container.streams.video[0]
            if stream.frames:
                return int(stream.frames)
            return sum(1 for _ in container.decode(video=0))

    def _decode_reference_frames(self, source, plan, refs):
        """One scan up to this rank's last keyframe; only the keyframes are converted."""
        wanted = {plan.ref_frame(ref): ref for ref in refs}
        if not wanted:
            return {}
        limit = max(wanted)
        frames = {}
        with av.open(str(source)) as container:
            container.streams.video[0].thread_type = "AUTO"
            for index, frame in enumerate(container.decode(video=0)):
                ref = wanted.get(index)
                if ref is not None:
                    frames[ref] = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous()
                if index >= limit:
                    break
        if len(frames) != len(wanted):
            raise ValueError(f"RelayVSR decoded {len(frames)} of {len(wanted)} reference frames from {source}")
        return frames

    def _exchange_boundary_latent(self, spans, span, latents, rank):
        """Hand this rank's first latent to the predecessor and take the successor's."""
        successor = rank + 1
        receives = successor < len(spans) and spans[successor].active
        sends = rank > 0 and spans[rank - 1].active
        if not receives and not sends:
            return None
        boundary = torch.empty_like(latents[0]) if receives else None
        requests = []
        if sends:
            requests.append(dist.P2POp(dist.isend, latents[0], rank - 1, group=self.chunk_p_group))
        if receives:
            requests.append(dist.P2POp(dist.irecv, boundary, successor, group=self.chunk_p_group))
        for handle in dist.batch_isend_irecv(requests):
            handle.wait()
        return boundary

    def _decode_windows(self, source, plan, span, conditions, work_dir, fps, target, height, width, timer=None):
        clip_len = int(self.config.get("clip_len", 16))
        writer = AsyncVideoWriter(str(work_dir / f"{span.rank:03d}.mp4"), fps, self.config)
        written = 0
        try:
            with av.open(str(source)) as container:
                container.streams.video[0].thread_type = "AUTO"
                with closing(prefetch_frames(container, clip_len, skip=span.decode.start)) as frames:
                    stream = iter(frames)
                    window_frames = []
                    for position, window in enumerate(span.windows):
                        length = len(plan.window_frames(window))
                        window_frames = window_frames[-1:] if position else []
                        while len(window_frames) < length:
                            frame = next(stream, None)
                            if frame is None:
                                raise ValueError(f"RelayVSR source {source} ended before frame {span.decode.stop}")
                            window_frames.append(frame)
                        self.check_stop()
                        start_condition = conditions[window]
                        end_condition = conditions[window + 1] if length > 1 else start_condition
                        pair = [(torch.cat((a[0], b[0]), 0), torch.cat((a[1], b[1]), 0)) for a, b in zip(start_condition, end_condition)]
                        self.flash_model_history.reset()
                        for offset, frame in enumerate(window_frames):
                            self.check_stop()
                            output = self.model.flash_model_decode(self.preprocess(frame), pair, self.flash_model_history, offset)
                            if window == 0 or offset > 0:
                                writer.submit(self.postprocess(output[..., : height * self.geometry.scale, : width * self.geometry.scale], target))
                                written += 1
                    if span.decode.stop >= plan.total and next(stream, None) is not None:
                        raise ValueError(f"RelayVSR source {source} has more frames than the counted {plan.total}")
        finally:
            writer.close()
        if timer is not None:
            timer["encode_wait"] = writer.encode_wait
        return written

    def _stage_exact_references(self, source, plan, span, seed, rank):
        """Decode this rank's keyframes, draw its noises and build the flash caches."""
        if not span.active:
            return [(rank, 0, [], [])], None
        frames = self._decode_reference_frames(source, plan, span.refs)
        shape = frames[span.refs.start].shape[-2:]
        padded = self.geometry.padded_size(*shape)
        # init_run reseeds the scheduler; the reference chain needs one draw per reference from
        # index 0 so the Philox offsets line up with the serial run.
        self.init_run(*padded, seed, with_wan_cache=False)
        noises = []
        for ref in range(span.refs.stop):
            self.scheduler.prepare_latents(*padded, self.device)
            if ref >= span.refs.start:
                noises.append(self.scheduler.latents.clone())
        return [(rank, span.refs.start, noises, [self.preprocess(frames[ref]) for ref in span.refs])], shape

    def _run_video_exact(self, source, destination, fps, target, seed):
        """精确块并行跑一条视频:各 rank 一段连续窗口,reference 按层同步,rank0 拼回一个文件。

        与旧的"段首预热"不同,这里不接受任何近似:reference 的每一层都在全体 rank 上前进,
        段尾 ``context - 1`` 个 reference 的 K/V 点对点交给下一段,注意力读到的窗口与串行逐位相同。

        异常收敛:准备阶段与 Phase B 的失败都和取消走同一条 ``_converge`` 路径(全员对齐后一起抛),
        Phase A 一旦进入逐层 P2P,单个 rank 的异常只能等 NCCL 超时 —— 这是方案 §3.5 明确接受的。
        """
        rank = dist.get_rank(self.chunk_p_group)
        world = dist.get_world_size(self.chunk_p_group)
        totals = [self._video_frame_count(source) if rank == 0 else None]
        dist.broadcast_object_list(totals, src=0, group=self.chunk_p_group)
        plan = build_plan(int(totals[0]), int(self.config.get("clip_len", 16)))
        spans = split_ranks(plan, world)
        span = spans[rank]
        context = int(self.model.wan_config["temporal_context_frames"])
        exchange = NcclKVExchange(kv_transfer_plan(spans, context), rank, self.chunk_p_group)
        # 分片写在本机临时目录:拼接读的是本地盘,不和生产产物所在的(NFS)目录抢带宽。
        staging = self.config.get("chunk_work_dir") or tempfile.gettempdir()
        directories = [tempfile.mkdtemp(prefix="relayvsr-exact-", dir=staging) if rank == 0 else None]
        dist.broadcast_object_list(directories, src=0, group=self.chunk_p_group)
        work_dir = Path(directories[0])
        logger.info(
            f"[RelayVSR] exact rank {rank}/{world}: windows [{span.windows.start}, {span.windows.stop}) of {plan.windows}, refs [{span.refs.start}, {span.refs.stop}), frames [{span.decode.start}, {span.decode.stop}) of {plan.total}"
        )

        timer = {} if self.config.get("relay_timing_log") else None
        previous_local = getattr(self, "_rank_local_collectives", False)
        written = 0
        failure = None
        try:
            batches, frame_shape = [(rank, 0, [], [])], None
            if timer is not None:
                timer["decode_keyframes"] = -time.perf_counter()
            try:
                batches, frame_shape = self._stage_exact_references(source, plan, span, seed, rank)
            except BaseException as error:  # noqa: BLE001 - 任一 rank 准备失败都要全员一起退出
                failure = error
            if timer is not None:
                timer["decode_keyframes"] = time.perf_counter() + timer["decode_keyframes"]
            # 谁都不许带着坏状态进入逐层 P2P:准备失败在这里全员一起抛。
            failure = self._converge(failure)
            if failure is not None:
                shutil.rmtree(work_dir, ignore_errors=True)
                raise failure

            # Phase A 是全员逐层同步的(空闲 rank 也跑满层数),所以停止检查在阶段内可以是全局的。
            self._rank_local_collectives = False
            if timer is not None:
                self.model.wan_transformer.timing = timer
                timer["reference"] = -time.perf_counter()
            latents = self.model.infer_references(batches, exchange, context, before_layer=self.check_stop)[0]
            if timer is not None:
                self.model.wan_transformer.timing = None
                timer["reference"] += time.perf_counter()

            if span.active:
                conditions = {ref: self.model.flash_model_prepare_condition(latent) for ref, latent in zip(span.refs, latents)}
                boundary = self._exchange_boundary_latent(spans, span, latents, rank)
                if boundary is not None:
                    conditions[span.refs.stop] = self.model.flash_model_prepare_condition(boundary)
                # 生产上只有 rank0 持有 stop_event,而 Phase B 按 rank 判断停止,所以持有者会立刻抛、
                # 其余 rank 会把自己那段跑完才到下面的 rendezvous —— 取消延迟 ≈ 其它 rank 的剩余工作量
                # (实测见自测报告 §6,30 秒素材约 8~10s)。要缩短它就得每窗口做一次全员轮询,
                # 而各 rank 的窗口数不同(13/13/12/12),那样会引入配对不上的集合通信。
                self._rank_local_collectives = True
                if timer is not None:
                    timer["flash"] = -time.perf_counter()
                try:
                    written = self._decode_windows(source, plan, span, conditions, work_dir, fps, target, *frame_shape, timer=timer)
                except BaseException as error:  # noqa: BLE001 - 编码/解码失败与取消同路收敛
                    failure = error
                if timer is not None:
                    timer["flash"] = time.perf_counter() + timer["flash"]
        finally:
            self._rank_local_collectives = previous_local
            if span.active:
                self.end_run()
        logger.info(f"[RelayVSR] exact rank {rank}: {written} frames written")
        self._chunk_barrier(timer)
        failure = self._converge(failure)
        if failure is not None:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise failure
        if timer is not None:
            timer["assemble"] = -time.perf_counter()
        if rank == 0:
            self._assemble_chunks(work_dir, spans, destination, source, plan.total)
        self._chunk_barrier(timer)
        if timer is not None:
            timer["assemble"] = time.perf_counter() + timer["assemble"]
            timer.pop("_last", None)
            report = " ".join(f"{key}={value:.2f}" for key, value in sorted(timer.items()))
            logger.info(f"[RelayVSR][timing] rank={rank} world={world} frames={plan.total} refs={len(span.refs)} windows={len(span.windows)} {report}")
        return {"video": None}

    def _assemble_chunks(self, work_dir, spans, destination, source, total):
        manifest = work_dir / "chunks.txt"
        with open(manifest, "w") as handle:
            handle.write("ffconcat version 1.0\n")
            for span in spans:
                if span.active:
                    handle.write(f"file '{span.rank:03d}.mp4'\n")
        started = time.perf_counter()
        result = mux_audio_from_video(
            str(source),
            str(manifest),
            output_path=str(destination),
            prefer_copy=self.config.get("audio_mux_prefer_copy", True),
            trim_to_shortest=False,
        )
        if result is None:
            raise RuntimeError("RelayVSR failed to assemble the chunked output video")
        logger.info(f"[RelayVSR] exact assembled {total} frames in {time.perf_counter() - started:.1f}s -> {destination}")
        shutil.rmtree(work_dir, ignore_errors=True)

    def _converge(self, failure):
        """Agree on cancel/failure across ranks once, so nobody waits for the NCCL watchdog.

        Phase A advances in lockstep, so a failure there already reaches every rank.  Phase B
        is rank-local — each rank owns different windows — so a cancelled or failed request
        only surfaces on one rank; without this rendezvous the others would sit in the closing
        barrier until the watchdog fires.
        """
        if self.chunk_p_group is None:
            return failure
        signal = torch.tensor([0 if failure is None else dist.get_rank(self.chunk_p_group) + 1], dtype=torch.int32, device=self.device)
        dist.all_reduce(signal, op=dist.ReduceOp.MAX, group=self.chunk_p_group)
        source = int(signal.item()) - 1
        if source < 0 or failure is not None:
            return failure
        return RuntimeError(f"RelayVSR exact chunk aborted because rank {source} failed")

    def _chunk_barrier(self, timer=None):
        if self.chunk_p_group is None:
            return
        if timer is None:
            dist.barrier(group=self.chunk_p_group)
            return
        started = time.perf_counter()
        dist.barrier(group=self.chunk_p_group)
        timer["barrier_wait"] = timer.get("barrier_wait", 0.0) + time.perf_counter() - started

    @ProfilingContext4DebugL1("🚀 RUN pipeline")
    def run_pipeline(self, input_info):
        if bool(input_info.image_path) == bool(input_info.video_path):
            raise ValueError("RelayVSR requires exactly one image_path or video_path")
        if input_info.return_result_tensor and input_info.video_path:
            raise ValueError("RelayVSR streams videos to save_result_path; returning a whole video tensor is unsupported")
        if not input_info.save_result_path and not input_info.return_result_tensor:
            raise ValueError("RelayVSR requires save_result_path")
        self.input_info = input_info
        if input_info.video_path:
            return self.run_video(input_info)
        return self.run_image(input_info)

    def run_image(self, input_info):
        with Image.open(input_info.image_path) as image:
            frame = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).contiguous()
        target = self.output_size(input_info, *frame.shape[-2:])
        with closing(self.restore_frames([frame], input_info.seed)) as predictions:
            pixels = self.postprocess(next(predictions), target).cpu()
        if self.main_rank and input_info.save_result_path:
            path = Path(input_info.save_result_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels.numpy()).save(path)
        return {"images": pixels[None].float().div_(255) if input_info.return_result_tensor else None}

    def run_video(self, input_info):
        source, destination = Path(input_info.video_path), Path(input_info.save_result_path)
        if source.resolve() == destination.resolve():
            raise ValueError("RelayVSR output must differ from its source video")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        writer = None
        output_frames = 0
        try:
            if self.main_rank:
                fd, temporary = tempfile.mkstemp(prefix=".relayvsr-", suffix=destination.suffix or ".mp4", dir=destination.parent)
                os.close(fd)
            with av.open(str(source)) as container:
                if not container.streams.video:
                    raise ValueError("RelayVSR input contains no video stream")
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                fps = Fraction(str(self.config["fps"])) if self.config.get("fps") is not None else stream.average_rate
                if fps is None or fps <= 0:
                    raise ValueError("Cannot determine source FPS; set startup fps explicitly")
                input_height, input_width = stream.height, stream.width
                target = self.output_size(input_info, input_height, input_width)
                if target[0] % 2 or target[1] % 2:
                    raise ValueError("YUV420 video output requires even height and width")
                input_info.output_fps = float(fps)
                if self.chunk_p_size > 1:
                    # 精确块并行:所有 rank 都参与,由 rank0 拼回一个文件。
                    return self._run_video_exact(source, destination, fps, target, input_info.seed)
                if self.main_rank:
                    writer = AsyncVideoWriter(temporary, fps, self.config)
                with closing(prefetch_frames(container, self.config.get("clip_len", 16))) as frames, closing(self.restore_frames(frames, input_info.seed)) as predictions:
                    for prediction in predictions:
                        if self.main_rank:
                            pixels = self.postprocess(prediction, target)
                            writer.submit(pixels)
                        output_frames += 1
                        if self.main_rank and output_frames % 30 == 0:
                            logger.info("RelayVSR: {} frames written to encoder queue", output_frames)
            if writer is not None:
                writer.close()
            if self.main_rank:
                if mux_audio_from_video(str(source), temporary, prefer_copy=True, trim_to_shortest=False) is None:
                    raise RuntimeError("RelayVSR could not mux the source audio")
                os.replace(temporary, destination)
                temporary = None
                logger.info("RelayVSR video saved to: {}", destination.resolve())
            return {"video": None}
        finally:
            if writer is not None and temporary is not None:
                try:
                    writer.close()
                except (RuntimeError, ValueError, OSError):
                    logger.exception("RelayVSR encoder cleanup failed")
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)


@RUNNER_REGISTER("relayvsr_x4")
class RelayVSRX4Runner(RelayVSRRunner):
    """Native 4x inference, followed by the existing optional output resizing."""

    geometry = RelayVSRGeometry(4)


@RUNNER_REGISTER("relayvsr_x2")
class RelayVSRX2Runner(RelayVSRRunner):
    """Native 2x inference with an exact 2x output contract."""

    geometry = RelayVSRGeometry(2)
    WARMUP_RESOLUTIONS = ((768, 1024), (768, 1344))

    def create_input_info(self, request_data):
        info = super().create_input_info(request_data)
        if info.sr_ratio != 2:
            raise ValueError("relayvsr_x2 supports only sr_ratio=2")
        return info
