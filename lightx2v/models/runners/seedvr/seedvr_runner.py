"""
Runner for SeedVR video super-resolution model.

SeedVR is a video super-resolution model that uses:
- NaDiT (Native Resolution Diffusion Transformer)
- Video VAE for encoding/decoding
- Pre-computed text embeddings
"""

import datetime
import gc
import math
import os
import shutil
import subprocess
import tempfile
import time

import imageio_ffmpeg as ffmpeg
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from torch import Tensor

from lightx2v.models.runners.base_runner import TaskStopped
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.seedvr.scheduler import SeedVRScheduler
from lightx2v.models.video_encoders.hf.seedvr import attn_video_vae_v3_s8_c16_t4_inflation_sd3_init
from lightx2v.models.video_encoders.hf.seedvr.color_fix import wavelet_reconstruction
from lightx2v.utils.envs import *
from lightx2v.utils.profiler import *
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v.utils.utils import mux_audio_from_video, save_to_video, wan_vae_to_comfy
from lightx2v_platform.base.global_var import AI_DEVICE


def _get_read_video():
    """Return ``read_video`` with a 3-level fallback chain.

    torchvision moved ``read_video`` between releases; the last-resort PyAV
    fallback handles environments where torchvision isn't installed at all.
    """
    try:
        from torchvision.io import read_video
    except ImportError:
        try:
            from torchvision.io.video import read_video
        except ImportError:
            import av

            def read_video(filename, start_pts=0, end_pts=None, pts_unit="pts", output_format="THWC"):
                # honor start_pts/end_pts (seconds) — segmented SR relies on them; decoding
                # the whole file here would silently repeat the first segment everywhere
                container = av.open(filename)
                try:
                    if not container.streams.video:
                        raise ValueError(f"No video stream found in {filename}")
                    stream = container.streams.video[0]
                    try:
                        fps = float(stream.average_rate) if stream.average_rate else 0.0
                    except ZeroDivisionError:
                        fps = 0.0
                    time_base = float(stream.time_base) if stream.time_base else None
                    frames = []
                    for frame in container.decode(video=0):
                        ts = frame.pts * time_base if (frame.pts is not None and time_base) else None
                        if ts is not None and pts_unit == "sec":
                            if ts < float(start_pts) - 1e-6:
                                continue
                            if end_pts is not None and ts > float(end_pts) + 1e-6:
                                break
                        img = frame.to_ndarray(format="rgb24")
                        frames.append(img)
                    if not frames:
                        raise ValueError(f"No frames decoded from {filename} in range [{start_pts}, {end_pts}]")
                finally:
                    container.close()
                video = torch.from_numpy(np.stack(frames))  # T H W C
                if output_format == "TCHW":
                    video = video.permute(0, 3, 1, 2)
                return video, torch.zeros(0), {"video_fps": fps}

    return read_video


@RUNNER_REGISTER("seedvr2")
class SeedVRRunner(DefaultRunner):
    """Runner for SeedVR video super-resolution model."""

    def __init__(self, config):
        super().__init__(config)
        self.run_input_encoder = self._run_input_encoder_local_sr
        self.text_encoder_output = None

        model_path_base = config.get("model_path", "ByteDance-Seed/SeedVR2-3B")
        if self.config.get("dit_quantized_ckpt", None):
            self.model_path = self.config.get("dit_quantized_ckpt")
        elif self.config.get("dit_original_ckpt", None):
            self.model_path = self.config.get("dit_original_ckpt")
        else:
            model_size = self.config.get("model_size", "3b")
            self.model_path = os.path.join(model_path_base, f"seedvr2_ema_{model_size}.pth")
        self.vae_path = os.path.join(model_path_base, "ema_vae.pth")
        self.pos_emb_path = os.path.join(model_path_base, "pos_emb.pt")
        self.neg_emb_path = os.path.join(model_path_base, "neg_emb.pt")

    def _build_video_transform(self, img):
        from torchvision.transforms import Normalize

        from lightx2v.models.video_encoders.hf.seedvr.data.image.transforms.divisible_crop import DivisibleCrop
        from lightx2v.models.video_encoders.hf.seedvr.data.image.transforms.na_resize import NaResize
        from lightx2v.models.video_encoders.hf.seedvr.data.video.transforms.rearrange import Rearrange

        target_height = self.config.get("target_height", 720)
        target_width = self.config.get("target_width", 1280)
        resolution = min((self.ori_h * self.ori_w) ** 0.5 * self.input_info.sr_ratio, (target_height * target_width) ** 0.5)

        # Run the transform on the accelerator, not on ``init_device``.
        # ``set_init_device`` sends the *data* to CPU whenever cpu_offload is on
        # (that switch is meant for model weights), which put this whole chain --
        # a bicubic upscale plus an edge pad over every frame -- on one CPU core.
        # Measured on 4xA100/aarch64, 96 frames 1344x768 -> 1920x1104: 43.7s on
        # CPU versus 0.04s here, and the CPU figure swings by 20x run to run
        # because it is competing for memory bandwidth with the other ranks.
        #
        # Uploading first also moves *less* data: the source resolution is 2.5x
        # smaller than the upscaled result, so the host-to-device copy shrinks.
        # The output stays on the device on purpose -- ``vae_encode`` would move
        # it there anyway, and ``run_vae_decoder``'s color_fix="gpu" reads
        # ``self._input`` straight off it instead of re-uploading per segment.
        img = img.to(AI_DEVICE, non_blocking=True)

        img = NaResize(
            resolution=resolution,
            mode="area",
            downsample_only=False,
        )(img)

        img.clamp_(0.0, 1.0)

        img = DivisibleCrop((16, 16))(img)

        Normalize(0.5, 0.5, inplace=True)(img)

        img = Rearrange("t c h w -> c t h w")(img)

        return img

    def _get_sr_segment_params(self):
        # sr_overlap is the cross-fade window, not just a dropped guard frame:
        # adjacent segments are denoised independently, so their shared frames
        # disagree on hallucinated detail and the boundary has to be ramped
        # across. One frame leaves nothing to ramp over (that is what produced
        # the visible jump at 3.3s); 8 frames is a third of a second at 24fps.
        seg_len = int(self.config.get("sr_segment_length", 81))
        overlap = int(self.config.get("sr_overlap", 8))
        if seg_len <= 0:
            return None, 0
        if overlap >= seg_len:
            overlap = max(seg_len - 1, 0)
            logger.warning(f"[SeedVRRunner] sr_overlap >= sr_segment_length, clamp to {overlap}")
        if 0 < overlap < 2:
            logger.warning(f"[SeedVRRunner] sr_overlap={overlap} is too small to cross-fade; segment boundaries may still jump. Use >= 2 (8 recommended).")
        return seg_len, overlap

    def _set_output_fps(self, fps):
        if fps is None:
            return
        try:
            fps = float(fps)
        except Exception:
            return
        if fps <= 0:
            return
        with self.config.temporarily_unlocked():
            self.config["fps"] = fps

    def _probe_video(self, video_path):
        # torchvision >= 0.23 removed read_video_timestamps; fall back to PyAV (same as _get_read_video)
        try:
            from torchvision.io import read_video_timestamps
        except ImportError:
            read_video_timestamps = None

        pts, fps = [], None
        if read_video_timestamps is not None:
            try:
                pts, fps = read_video_timestamps(video_path, pts_unit="sec")
            except Exception as e:
                logger.warning(f"[SeedVRRunner] read_video_timestamps failed: {e}")
                pts, fps = [], None
        else:
            try:
                import av

                with av.open(video_path) as container:
                    stream = container.streams.video[0]
                    fps = float(stream.average_rate) if stream.average_rate else None
                    time_base = float(stream.time_base) if stream.time_base else None
                    pts = [float(p.pts) * time_base for p in container.demux(stream) if p.pts is not None and time_base]
                    pts.sort()
            except Exception as e:
                logger.warning(f"[SeedVRRunner] PyAV probe failed: {e}")
                pts, fps = [], None

        total_frames = len(pts) if pts is not None else 0
        fps_for_seek = fps
        if fps_for_seek is None or fps_for_seek == 0:
            fps_for_seek = float(self.config.get("fps", 16))
        if fps is not None and fps != 0:
            self._set_output_fps(fps)
        return total_frames, fps_for_seek, pts

    def _build_sr_segments(self, total_frames, seg_len, overlap):
        """Split into overlapping segments of near-equal length.

        Balanced rather than greedy. A greedy walk fills every segment to
        ``seg_len`` and leaves the remainder as a runt: 124 frames at
        ``seg_len=121`` splits 121+3, and a 3-frame tail is both too short to
        cross-fade and too short to denoise with the same temporal context as
        its neighbour. That is not a corner case here -- MiniMax H3's 17n+5
        frame grid (124/243/362) never lands at or below 121, so every H3 clip
        hits it.

        Balancing costs no extra diffusion pass: the segment *count* is what a
        greedy walk would produce, only the lengths are evened out. Every
        segment stays <= ``seg_len``, so the caller's VRAM ceiling still holds
        (peak can only drop).
        """
        if total_frames <= seg_len:
            return [(0, total_frames)]
        stride = max(seg_len - overlap, 1)
        num_segments = max(1, math.ceil((total_frames - overlap) / stride))
        # n segments of length L overlapping by `overlap` cover n*L - (n-1)*overlap
        # frames, so solve that for L. Round up, then clamp: L must exceed the
        # overlap (or a segment would be entirely swallowed by its neighbours)
        # and must not exceed the memory-derived seg_len.
        balanced_len = math.ceil((total_frames + (num_segments - 1) * overlap) / num_segments)
        balanced_len = min(max(balanced_len, overlap + 1), seg_len)
        segments = []
        start = 0
        while start < total_frames:
            end = min(start + balanced_len, total_frames)
            segments.append((start, end))
            if end >= total_frames:
                break
            start = end - overlap
            if start < 0:
                start = 0
        return segments

    @staticmethod
    def _sr_blend_weights(count, device, dtype):
        """Cross-fade ramp over ``count`` frames: 0 on the first, 1 on the last.

        Anchoring the ends at exactly 0 and 1 keeps the ramped window
        continuous with the single-segment frames on either side of it. A ramp
        that stopped short of its endpoints would only relocate the seam to the
        window edges instead of removing it.
        """
        if count <= 1:
            return torch.full((max(count, 1),), 0.5, device=device, dtype=dtype)
        return torch.linspace(0.0, 1.0, count, device=device, dtype=dtype)

    def _blend_sr_overlap(self, prev_tail, segment):
        """Cross-fade the previous segment's tail into this segment's head.

        ``prev_tail`` and the head of ``segment`` are the SAME source frames,
        denoised twice -- once by each segment. The old path kept one copy and
        discarded the other, which left two different diffusion realizations of
        the same shot on adjacent frames; that is the jump. Ramping between
        them spreads the disagreement across the whole overlap window.

        Both tensors are ``[B, C, T, H, W]``. ``prev_tail`` is parked on CPU
        between segments so it stays out of the VRAM peak, hence the ``.to()``.
        """
        count = min(prev_tail.shape[2], segment.shape[2])
        if count <= 0:
            return segment
        tail = prev_tail[:, :, -count:].to(device=segment.device, dtype=segment.dtype)
        weights = self._sr_blend_weights(count, segment.device, segment.dtype).view(1, 1, count, 1, 1)
        blended = tail * (1.0 - weights) + segment[:, :, :count] * weights
        return torch.cat([blended, segment[:, :, count:]], dim=2)

    # ---------------------------------------------------------------- seg parallel
    #
    # Once boundaries cross-fade, segments have no sequential dependency left:
    # the overlap frames are denoised INDEPENDENTLY by both neighbours and then
    # averaged, so nothing a segment produces feeds the next one's diffusion.
    # That makes whole segments a data-parallel axis -- one segment per rank,
    # with only the held-back boundary frames crossing the wire.
    #
    # This is the right axis for SeedVR specifically. Measured on a 15.8s clip
    # (832x480 -> 1664x960, 4 segments, A100): DiT is 14% of a segment's 82s,
    # VAE decode ~62%. Sequence-parallelising the DiT therefore caps out around
    # a 10% end-to-end win, while segment parallel scales the whole 82s.
    # Speedup ceiling is the segment count, not the card count.

    _SR_TAIL_META = 5  # [n_frames, B, C, H, W]; n_frames == 0 means "no tail"
    # Sized for the straggler, not the average: a rank's whole segment (read,
    # diffuse, decode, encode) has to fit inside one rendezvous wait.
    _SR_CTRL_TIMEOUT = datetime.timedelta(hours=2)
    # Much tighter than the control plane, because a tail is ready the instant
    # its sender's segment is: the receiver is at most one inter-rank skew
    # behind, so minutes of silence can only mean the peer is gone.
    _SR_TAIL_TIMEOUT = datetime.timedelta(minutes=15)

    @staticmethod
    def _sr_segment_owner(idx, world):
        """Round-robin. Consecutive segments land on different ranks whenever
        ``world > 1``, which is what lets the tail hand-off be a plain
        point-to-point send instead of a self-send special case."""
        return idx % world

    def _sr_seg_parallel_info(self, num_segments, file_output, vfi_target):
        """``(rank, world)`` for segment parallel, or ``(0, 1)`` when it is off.

        Off unless every one of these holds:
        - ``seg_p_size > 1`` was configured and the process group is up;
        - output goes to a file -- returning one stitched tensor would need a
          variable-size gather, and the caller only ever consumes rank 0's;
        - RIFE interpolation is off -- it threads a global target-frame grid
          through the segment loop, which is a real sequential dependency.
        """
        if not self.config.get("seg_parallel", False) or not dist.is_available() or not dist.is_initialized():
            return 0, 1
        rank, world = dist.get_rank(), dist.get_world_size()
        if world <= 1 or num_segments <= 1:
            return 0, 1
        reason = None
        if not file_output:
            reason = "tensor output needs a variable-size gather"
        elif vfi_target:
            reason = "frame interpolation carries a global frame grid across segments"
        if reason is not None:
            if rank == 0:
                logger.warning(f"[SeedVRRunner] seg_parallel off ({reason}); rank 0 runs all {num_segments} segments serially")
            return 0, 1
        if rank == 0:
            logger.info(f"[SeedVRRunner] seg_parallel: {num_segments} segments over {world} ranks, expected speedup x{min(num_segments, world)}")
        return rank, world

    def _sr_noise_generator(self, device):
        """RNG for this segment's noise, keyed on (seed, global segment index).

        ``torch.randn_like`` draws from the global RNG, whose state depends on
        how many segments this *process* has already run. That makes the output
        a function of the work split: on one card segment 2 gets an advanced
        state, while under segment parallelism every rank's first segment draws
        the identical tensor -- so the same clip yields a different file at 1 vs
        4 cards, and neighbouring segments can end up sharing one noise pattern.
        Keying the stream on the segment's global index removes the coupling in
        both directions: rank layout stops mattering, and each segment still
        gets its own noise.

        Returns ``None`` when there is no segmentation in play, which keeps
        whole-clip runs on the global RNG exactly as before.
        """
        index = getattr(self, "_sr_segment_index", None)
        if index is None or getattr(self, "_sr_segment", None) is None:
            return None
        seed = getattr(self.input_info, "seed", None)
        if seed is None:
            return None
        generator = torch.Generator(device=device if device is not None else "cpu")
        # Mixed rather than added so (seed, index) pairs cannot collide across seeds.
        generator.manual_seed((int(seed) * 1000003 + int(index)) % (2**63 - 1))
        return generator

    @staticmethod
    def _sr_randn_like(latent, generator):
        if generator is None:
            return torch.randn_like(latent)
        return torch.randn(latent.shape, generator=generator, dtype=latent.dtype, device=latent.device, layout=latent.layout)

    def _sr_ctrl_group(self):
        """The CPU process group every control-plane collective runs on.

        Created once and cached: ``new_group`` is itself collective, so it has
        to be reached by every rank through the same branches. Returns None when
        the build has no gloo, leaving callers on the default group.

        Gloo rather than NCCL, because a NCCL rendezvous is the wrong tool here
        twice over. It spin-waits on the GPU, so ranks that finished early burn
        a full SM's worth of power waiting on the straggler; and its watchdog
        aborts the whole process group after ``TORCH_NCCL_TIMEOUT`` (10 minutes
        by default). Segments are near-equal in compute but their video encodes
        are not -- a first run on 4xA100 had three ranks waiting ~10 minutes for
        the fourth and the watchdog killed the job just as the last segment was
        being written. Gloo sleeps instead of spinning and gets a timeout sized
        for a slow encode, so a straggler costs wall-clock rather than the run.

        Exactly one collective ever runs on this group: the all-reduce in
        :meth:`_sr_seg_rendezvous`. That is deliberate. Ranks reach the meeting
        points down different code paths -- normal return, cancellation, a
        raise, a rank-0-only fallback -- and if any of those paths used a
        *different* op (a plain ``barrier``, say) then two ranks meeting from
        two paths would pair a barrier against an all-reduce and both would wait
        out the two-hour timeout. One op means any two ranks that arrive here
        match, no matter how they got here. Do not add a second one.
        """
        if not hasattr(self, "_sr_ctrl_pg"):
            try:
                self._sr_ctrl_pg = dist.new_group(backend="gloo", timeout=self._SR_CTRL_TIMEOUT)
            except Exception as e:  # pragma: no cover - depends on the build
                logger.warning(f"[SeedVRRunner] gloo rendezvous unavailable ({e}); falling back to the default process group")
                self._sr_ctrl_pg = None
        return self._sr_ctrl_pg

    def _sr_seg_rendezvous(self, failed):
        """Meet the other ranks and agree on how far the request got.

        The *only* way ranks meet in this file. Every point where they have to
        agree -- the scratch dir being ready, the segments all being written,
        the rank-0-only fallback finishing -- calls this, so the ranks pair the
        same op no matter which path brought them there (see
        :meth:`_sr_ctrl_group`). Reached the same number of times on every rank
        per request, which is what the ``agreed`` bookkeeping in
        :meth:`_run_sr_segments` exists to guarantee.

        Under segment parallelism the ranks run different amounts of work, so
        ``check_stop``'s per-step all-reduce cannot be used to agree on
        cancellation: a rank owning no segment never reaches it. That is not a
        theoretical hazard -- a 124-frame request (2 segments) on 4 ranks left
        ranks 0 and 1 spinning in the step all-reduce until NCCL's watchdog
        aborted the whole server 600 s later. So the denoise loop is put in
        rank-local mode and everything is settled here instead: one MAX
        all-reduce of (stopped, paused, failed) that every rank reaches exactly
        once, from a ``finally``, whether it finished, cancelled, or raised.

        Deferring agreement costs at most one segment of stop latency.

        The ``failed`` flag turns one rank's traceback into a clean failure for
        the rest, but only because every blocking point between the raise and
        this call is bounded. That is not free: it is why the tail hand-off
        does not use NCCL (see :meth:`_sr_tail_transport`). A peer left waiting
        on a dead rank's boundary frames gives up on its own timeout and
        arrives here with ``failed=True`` of its own -- otherwise it would
        never arrive, and this flag would agree on nothing.
        """
        signals = torch.tensor(
            [
                1 if getattr(self, "stop_signal", False) else 0,
                1 if getattr(self, "pause_signal", False) else 0,
                1 if failed else 0,
            ],
            dtype=torch.int32,
        )
        group = self._sr_ctrl_group()
        if group is None:
            signals = signals.to(AI_DEVICE)
            dist.all_reduce(signals, op=dist.ReduceOp.MAX)
        else:
            dist.all_reduce(signals, op=dist.ReduceOp.MAX, group=group)
        stopped, paused, any_failed = (int(v) for v in signals)
        if stopped or paused:
            reason = "stop_signal" if stopped else "pause_signal"
            try:
                self.end_run()
            except Exception as e:
                logger.warning(f"[SeedVRRunner] end_run failed during {reason} teardown: {e}")
            raise TaskStopped(f"find rank: {dist.get_rank()} {reason}, stop running, it's an expected behavior")
        if any_failed and not failed:
            # Someone else's work is missing, so carrying on would either fail
            # the concat's count check or silently emit a short video. Fail
            # loudly here instead.
            raise RuntimeError("SeedVR seg_parallel: a peer rank failed; see that rank's traceback")

    def _sr_tail_transport(self):
        """How a tail crosses ranks: ``"gloo"`` (default) or ``"file"``.

        Neither is NCCL, and that is the point. A NCCL point-to-point op has no
        timeout and cannot be cancelled, so a rank that raises mid-segment
        parks its neighbour in ``dist.recv`` -- or the previous rank in the
        sender's ``work.wait()`` -- *before* either can reach
        :meth:`_sr_seg_rendezvous` and hear that the peer is gone. The failure
        never propagates; the watchdog aborts the process group ten minutes
        later and takes the whole server with it. Both transports here are
        bounded by ``_SR_TAIL_TIMEOUT`` and surface a dead peer as an ordinary
        exception, which the handler turns into a reported failure.

        ``gloo`` keeps the hand-off in-memory and is the default. ``file``
        publishes through the shared scratch dir instead: the sender never
        waits at all, so even the wave-crossing edge is free, at the cost of a
        ~200 MiB write and read per boundary on whatever backs the output dir.
        """
        transport = str(self.config.get("sr_tail_transport", "gloo")).lower()
        if transport not in ("gloo", "file"):
            logger.warning(f"[SeedVRRunner] unknown sr_tail_transport={transport!r}; using gloo")
            return "gloo"
        return transport

    def _sr_tail_group(self):
        """The CPU group the gloo hand-off runs on, or None to use files.

        Deliberately not ``_sr_ctrl_group``: that timeout is sized for a
        straggler finishing an entire segment, which is far too long to wait on
        a tail. Created once and cached, and because ``new_group`` is itself
        collective :meth:`_run_sr_segments` calls this on every rank up front --
        a rank that owns no segment never reaches a hand-off to create it
        lazily.
        """
        if not hasattr(self, "_sr_tail_pg"):
            try:
                self._sr_tail_pg = dist.new_group(backend="gloo", timeout=self._SR_TAIL_TIMEOUT)
            except Exception as e:  # pragma: no cover - depends on the build
                logger.warning(f"[SeedVRRunner] gloo tail transport unavailable ({e}); handing tails off through the scratch dir instead")
                self._sr_tail_pg = None
        return self._sr_tail_pg

    def _sr_enter_rank_local(self, seg_world):
        """Take the denoise loop rank-local, returning the caller's setting.

        Ranks diffuse different numbers of segments, so ``check_stop``'s
        per-step all-reduce cannot be world-wide; ``_sr_seg_rendezvous`` is
        where the ranks agree instead.

        OR-ed with what the caller had, never assigned over it. Under the
        rank-0 fallback :meth:`_run_sr_segments` is entered with
        ``seg_world == 1`` from inside :meth:`_sr_run_on_rank0`, which has
        already gone rank-local for its own reason -- only rank 0 is running
        and the peers are parked in a barrier, so a world all-reduce has no
        counterpart. A plain assignment there silently undid it and rank 0 sat
        in ``check_stop`` until NCCL's watchdog aborted the server 600 s later,
        which is exactly what a 4-card RIFE request did on the box.
        """
        previous = getattr(self, "_rank_local_collectives", False)
        self._rank_local_collectives = previous or seg_world > 1
        return previous

    def _sr_send_tail(self, tail, dst, idx):
        """Hand segment ``idx``'s held-back boundary frames to the next owner.

        Never blocks on the peer, because only some of these hand-offs are in
        lockstep. Group the segments into waves of ``world`` under the
        round-robin ownership: within a wave the sender and the receiver
        diffuse together and post send/recv at the same wall-clock moment, so
        the hand-off costs inter-rank skew. The edge that crosses a wave -- the
        last rank's segment into rank 0's next one -- does not, because the
        receiver diffuses its own segment *before* posting the recv. That tail
        is ready a full segment early, so a blocking send parks the last rank
        for the whole of it, and the delay walks backwards one rank per wave
        until the parallel speedup is gone.

        Sent as fp32 from CPU: the tail is already parked there between
        segments to stay out of the VRAM peak, and ``_blend_sr_overlap`` moves
        and casts it on arrival. ~200 MiB for an 8-frame 1920x1104 tail.
        """
        group = None if self._sr_tail_transport() == "file" else self._sr_tail_group()
        if group is None:
            self._sr_write_tail_file(tail, idx)
            return
        # Drain the previous hand-off before posting this one. A whole wave has
        # gone by, so it has long since been matched and the wait is free, and
        # it caps the memory held for in-flight tails at a single boundary
        # rather than letting it grow with the segment count.
        self._sr_drain_tail_sends()
        # A fixed-size meta tensor goes first so the receiver can allocate.
        if tail is None:
            meta = torch.zeros(self._SR_TAIL_META, dtype=torch.int64)
            self._sr_tail_sends.append((dist.isend(meta, dst=dst, group=group), meta))
            return
        payload = tail.detach().to(device="cpu", dtype=torch.float32).contiguous()
        b, c, t, h, w = payload.shape
        meta = torch.tensor([t, b, c, h, w], dtype=torch.int64)
        self._sr_tail_sends.append((dist.isend(meta, dst=dst, group=group), meta))
        self._sr_tail_sends.append((dist.isend(payload, dst=dst, group=group), payload))

    def _sr_drain_tail_sends(self):
        """Wait out the in-flight tail sends, then drop their payloads.

        Holding the tensors is what keeps them alive while gloo reads them, so
        the list has to be cleared here rather than by the caller. The wait is
        bounded by the tail group's timeout, so an unmatched send raises rather
        than hangs -- which is what lets the caller report the failure instead
        of stalling short of the rendezvous.
        """
        pending, self._sr_tail_sends = getattr(self, "_sr_tail_sends", []), []
        for work, _payload in pending:
            work.wait()

    def _sr_recv_tail(self, src, idx):
        """Take segment ``idx``'s tail from its owner.

        ``None`` when the sender had no tail to give (its segment produced no
        frames). Bounded either way: a peer that died before posting its
        hand-off surfaces here as an exception within ``_SR_TAIL_TIMEOUT``
        instead of parking this rank short of :meth:`_sr_seg_rendezvous`.
        """
        group = None if self._sr_tail_transport() == "file" else self._sr_tail_group()
        if group is None:
            return self._sr_read_tail_file(idx)
        meta = torch.empty(self._SR_TAIL_META, dtype=torch.int64)
        dist.recv(meta, src=src, group=group)
        t = int(meta[0])
        if t <= 0:
            return None
        b, c, h, w = (int(v) for v in meta[1:])
        payload = torch.empty((b, c, t, h, w), dtype=torch.float32)
        dist.recv(payload, src=src, group=group)
        return payload

    def _sr_tail_file(self, idx):
        return os.path.join(self._sr_tail_dir, f"tail_{idx:05d}.pt")

    def _sr_write_tail_file(self, tail, idx):
        """Publish segment ``idx``'s tail into the shared scratch dir.

        No rendezvous at all: the sender writes and walks away, so it never
        pays for the wave-crossing edge. Written to a ``.part`` sibling and
        renamed, because rename within a directory is atomic while a plain
        write is not, and the receiver polls for the final name -- it must
        never open a half-written file. Sharing the scratch dir across ranks is
        not a new assumption: rank 0 already lists it to collect everyone's
        segment videos.
        """
        path = self._sr_tail_file(idx)
        partial = f"{path}.part"
        payload = None if tail is None else tail.detach().to(device="cpu", dtype=torch.float32).contiguous()
        with open(partial, "wb") as fh:
            torch.save(payload, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(partial, path)

    def _sr_read_tail_file(self, idx):
        """Wait for segment ``idx``'s tail file, then consume it.

        Polls ``listdir`` rather than ``exists``: over NFS a negative lookup is
        cached per entry, while a readdir revalidates the whole directory, so
        ``exists`` can keep answering False well after the writer's rename.
        Removed once loaded, so a rank owning several segments does not carry
        every boundary it has ever received; the dir itself goes at the end of
        the request.
        """
        path = self._sr_tail_file(idx)
        name = os.path.basename(path)
        deadline = time.monotonic() + self._SR_TAIL_TIMEOUT.total_seconds()
        while name not in os.listdir(self._sr_tail_dir):
            if time.monotonic() >= deadline:
                raise RuntimeError(f"SeedVR seg_parallel: segment {idx}'s boundary frames never arrived at {path}; its owner most likely died mid-segment")
            time.sleep(0.25)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        os.remove(path)
        return payload

    def _sr_vfi_target(self, save_fps):
        """RIFE 目标帧率，未启用插帧时为 None。

        SR 插帧：config 配了 video_frame_interpolation 且请求 target_fps 高于源
        帧率才启用（不降帧）。分段文件路径逐段插帧，避免整段拼接的显存峰值。
        跨段用「全局目标帧栅格」保证节奏连续：每段带上一段末帧（prepend）作插值
        锚点，并传入本段的全局源帧偏移 + 全局目标帧区间，使非整数倍率下段边界也
        不重启相位、不与源音频错位——这条全局栅格是段间真实的顺序依赖，也是
        seg_parallel 遇到插帧必须退回串行的原因。
        """
        if self.vfi_model is None:
            return None
        target = (self.config.get("video_frame_interpolation") or {}).get("target_fps")
        return target if target and target > save_fps else None

    def _sr_run_on_rank0(self, fn):
        """Run ``fn`` on rank 0 only, with everyone meeting afterwards.

        Used for the requests segment parallel cannot take (single segment,
        tensor output, RIFE). Without this every rank would run the same job and
        race to write the same output file. The rendezvous is in a ``finally``
        so a failure on rank 0 does not strand the others until the timeout.

        The denoise loop goes rank-local for the same reason it does under
        segment parallelism, and it is not optional here: only rank 0 runs
        steps, so ``check_stop``'s per-step world all-reduce has no counterpart
        -- every other rank is already parked in the rendezvous below -- and
        rank 0 would sit in that all-reduce until NCCL's watchdog aborted the
        server.

        The failure flag is carried across rather than just met: a bare meeting
        would let the peers return ``None`` as if the request had succeeded
        while rank 0 unwound a traceback, leaving them one turn out of step on
        what the request actually did. Swallowed on the rank that already has an
        exception in flight, so the real traceback survives the ``finally``.
        """
        if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
            return fn()
        previous = getattr(self, "_rank_local_collectives", False)
        self._rank_local_collectives = True
        failed = True
        try:
            result = fn() if dist.get_rank() == 0 else None
            failed = False
            return result
        finally:
            self._rank_local_collectives = previous
            if failed:
                try:
                    self._sr_seg_rendezvous(failed=True)
                except Exception as e:
                    logger.warning(f"[SeedVRRunner] could not report this rank's failure to its peers: {e}")
            else:
                self._sr_seg_rendezvous(failed=False)

    def _read_video_segment(self, video_path, start_idx, end_idx):
        read_video = _get_read_video()

        total_len = max(end_idx - start_idx, 0)
        if total_len == 0:
            return torch.empty(0, 3, 0, 0)

        start_pts = None
        end_pts = None
        if getattr(self, "_sr_pts", None):
            start_pts = float(self._sr_pts[start_idx])
            end_pts = float(self._sr_pts[end_idx - 1]) + 1.0 / max(self._sr_fps, 1.0)
        else:
            start_pts = float(start_idx) / max(self._sr_fps, 1.0)
            end_pts = float(end_idx - 1) / max(self._sr_fps, 1.0) + 1.0 / max(self._sr_fps, 1.0)

        video, _, info = read_video(
            video_path,
            start_pts=start_pts,
            end_pts=end_pts,
            pts_unit="sec",
            output_format="TCHW",
        )
        if info is not None and self._sr_fps in [None, 0]:
            self._sr_fps = info.get("video_fps", self._sr_fps)
            self._set_output_fps(self._sr_fps)

        if video.shape[0] > total_len:
            video = video[:total_len]
        return video

    def _run_sr_single_segment(self, seg_idx=0, seg_total=1):
        """扩散一段。``seg_idx`` / ``seg_total`` 只用来把进度映射到整段任务。

        每段都以 ``segment_idx=0`` 调 ``run_segment`` 是有意的：``end_run_segment``
        按「是不是最后一段」决定释放 ``self.inputs``，而 SR 的每一段都在循环里重建
        它（见 ``_run_sr_segments``），逐段释放才对。

        代价是 ``run_segment`` 里那句进度上报按 ``segment_idx / video_segment_num``
        折算，恒等于 100%——``init_run`` 每段又把 ``video_segment_num`` 重置回 1。
        实测（9 段任务）：前 40 秒 0%，第一段做完直接跳满，再卡到结束。这里把回调
        包一层，按「已完成段数 + 段内进度」重算百分比。

        ``seg_idx`` / ``seg_total`` 是**本 rank 自己的**段序号与段数，不是全局的 ——
        理由见 ``_run_sr_segments`` 里算 ``local_total`` 处的注释（只有 rank 0 上报，
        用全局段号会让进度停在末尾附近）。
        """
        cached_input_info = self.input_info
        cached_cb = self.progress_callback

        if cached_cb is not None and seg_total > 1:

            def _scaled(current, cap):
                frac = (current / cap) if cap else 0.0
                cached_cb(((seg_idx + frac) / seg_total) * 100.0, 100.0)

            self.progress_callback = _scaled

        try:
            self.init_run()
            if self.config.get("compile", False) and hasattr(self.model, "comple"):
                self.model.select_graph_for_compile(self.input_info)

            segment_idx = 0
            self.init_run_segment(segment_idx)
            latents = self.run_segment(segment_idx)
            self.gen_video = self.run_vae_decoder(latents)
            self.end_run_segment(segment_idx)
            raw_video = self.gen_video_final
            self.end_run()
        finally:
            # 恢复必须在 finally 里：包装过的回调若泄漏到下一段，段号就永远停在
            # 这一段上；input_info 同理（原本在函数末尾恢复，异常路径会漏）。
            self.progress_callback = cached_cb
            self.input_info = cached_input_info
        return raw_video

    def _save_sr_segment_video(self, raw_video, output_path, fps):
        video = wan_vae_to_comfy(raw_video).float().clamp(0.0, 1.0)
        save_to_video(video, output_path, fps=fps, method="ffmpeg")
        del video

    def _concat_sr_segment_videos(self, segment_paths, output_path):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        if len(segment_paths) == 1:
            shutil.move(segment_paths[0], output_path)
            return

        concat_path = os.path.join(os.path.dirname(output_path) or ".", f".{os.path.basename(output_path)}.concat.txt")
        try:
            with open(concat_path, "w", encoding="utf-8") as f:
                for path in segment_paths:
                    escaped = os.path.abspath(path).replace("\\", "\\\\").replace("'", "\\'")
                    f.write(f"file '{escaped}'\n")

            command = [
                ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_path,
                "-c",
                "copy",
                output_path,
            ]
            process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, check=False)
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg concat failed: {process.stderr.strip()}")
        finally:
            if os.path.exists(concat_path):
                os.remove(concat_path)

    def _cut_videos(self, videos, sp_size):
        t = videos.size(1)
        if t == 1:
            return videos
        if t <= 4 * sp_size:
            padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - t + 1)
            padding = torch.cat(padding, dim=1)
            videos = torch.cat([videos, padding], dim=1)
            return videos
        if (t - 1) % (4 * sp_size) == 0:
            return videos
        padding = [videos[:, -1].unsqueeze(1)] * (4 * sp_size - ((t - 1) % (4 * sp_size)))
        padding = torch.cat(padding, dim=1)
        videos = torch.cat([videos, padding], dim=1)
        return videos

    def init_scheduler(self):
        """Initialize the scheduler for SeedVR."""
        self.scheduler = SeedVRScheduler(self.config)

    def load_transformer(self):
        """Load the SeedVR transformer model."""
        from lightx2v.models.networks.seedvr import SeedVRNaDiTModel

        model = SeedVRNaDiTModel(
            model_path=self.model_path,
            config=self.config,
            device=self.init_device,
        )
        return model

    def load_text_encoder(self):
        """Load text encoder for SeedVR.

        SeedVR uses pre-computed text embeddings (pos_emb.pt, neg_emb.pt).
        We load them from disk and cache them.
        """
        # For SeedVR, text embeddings are pre-computed
        # Load them during run_text_encoder
        return []

    def load_image_encoder(self):
        """SeedVR SR task doesn't use separate image encoder.

        The input video/image is encoded by VAE directly.
        """
        return None

    def load_vae_encoder(self):
        vae_causal_slice_size = int(self.config.get("vae_causal_slice_size", 4))
        vae_memory_limit_gb = float(self.config.get("vae_memory_limit_gb", 0.5))
        vae_memory_limit = None if vae_memory_limit_gb <= 0 else vae_memory_limit_gb
        vae = attn_video_vae_v3_s8_c16_t4_inflation_sd3_init(
            device=AI_DEVICE,
            dtype=GET_DTYPE(),
            weights_path=self.vae_path,
            weights_map_location="cpu",
            weights_mmap=True,
            strict=False,
            cpu_offload=self.config.get("cpu_offload", False),
            use_tiling=self.config.get("use_tiling_vae", False),
            tile_size=int(self.config.get("vae_tile_size", 512)),
            tile_overlap=int(self.config.get("vae_tile_overlap", 64)),
        )
        vae.requires_grad_(False).eval()
        vae.set_causal_slicing(split_size=vae_causal_slice_size if vae_causal_slice_size > 0 else None, memory_device="same")
        vae.set_memory_limit(conv_max_mem=vae_memory_limit, norm_max_mem=vae_memory_limit)
        logger.info(
            f"[SeedVRRunner] VAE config: tiling={self.config.get('use_tiling_vae', False)}, "
            f"tile={self.config.get('vae_tile_size', 512)}, overlap={self.config.get('vae_tile_overlap', 64)}, "
            f"causal_slice={vae_causal_slice_size if vae_causal_slice_size > 0 else 'off'}, "
            f"memory_limit={vae_memory_limit_gb if vae_memory_limit_gb > 0 else 'off'}GiB"
        )
        return vae

    def load_vae_decoder(self):
        pass

    def load_vae(self):
        """Load VAE encoder and decoder for SeedVR.

        SeedVR's VAE is a single model that can both encode and decode,
        so we return the same instance for both.
        """
        vae_encoder = self.load_vae_encoder()
        # Use the same VAE for encoding and decoding
        vae_decoder = vae_encoder
        return vae_encoder, vae_decoder

    def _restore_target_size(self, sample):
        if self.config.get("resize_mode") == "adaptive":
            return sample
        target_height = int(self.config.get("target_height", sample.shape[-2]) or sample.shape[-2])
        target_width = int(self.config.get("target_width", sample.shape[-1]) or sample.shape[-1])
        if target_height <= 0 or target_width <= 0:
            return sample

        height, width = sample.shape[-2:]
        if (height, width) == (target_height, target_width):
            return sample

        if height >= target_height and width >= target_width:
            top = (height - target_height) // 2
            left = (width - target_width) // 2
            logger.info(f"[SeedVRRunner] center crop SR output from {width}x{height} to {target_width}x{target_height}")
            return sample[..., top : top + target_height, left : left + target_width]

        logger.info(f"[SeedVRRunner] resize SR output from {width}x{height} to {target_width}x{target_height}")
        dtype = sample.dtype
        device = sample.device
        return F.interpolate(sample.float(), size=(target_height, target_width), mode="bilinear", align_corners=False).to(device=device, dtype=dtype)

    def run_vae_decoder(self, latents):
        samples = self.vae_decoder.vae_decode(latents)
        sample = [(rearrange(video[:, None], "c t h w -> t c h w") if video.ndim == 3 else rearrange(video, "c t h w -> t c h w")) for video in samples][0]
        if self._ori_length < sample.shape[0]:
            sample = sample[: self._ori_length]

        color_fix = str(self.config.get("color_fix", "cpu")).lower()
        if color_fix not in ("cpu", "gpu", "off"):
            logger.warning(f"[SeedVRRunner] Unknown color_fix={color_fix}; fallback to cpu")
            color_fix = "cpu"
        if color_fix != "off":
            input = rearrange(self._input[:, None], "c t h w -> t c h w") if self._input.ndim == 3 else rearrange(self._input, "c t h w -> t c h w")
            fix_device = torch.device("cpu") if color_fix == "cpu" else sample.device
            sample = wavelet_reconstruction(sample.to(fix_device), input[: sample.size(0)].to(fix_device))
        sample = self._restore_target_size(sample)
        sample = rearrange(sample[:, None], "t c h w -> c t h w") if sample.ndim == 3 else rearrange(sample, "t c h w -> c t h w")
        sample = sample[None, :]

        return sample

    def run_text_encoder(self, input_info):
        """Run text encoder for SeedVR.

        SeedVR uses pre-computed text embeddings.
        Load them from disk and return as context.
        """
        if self.text_encoder_output is not None:
            return self.text_encoder_output
        # Load positive embeddings
        if self.pos_emb_path:
            try:
                pos_emb = torch.load(self.pos_emb_path, map_location="cpu")
                pos_emb = pos_emb.to(self.init_device)
            except Exception as e:
                print(f"[SeedVRRunner] Failed to load pos_emb: {e}")
                pos_emb = None
        else:
            pos_emb = None

        # Load negative embeddings
        if self.neg_emb_path:
            try:
                neg_emb = torch.load(self.neg_emb_path, map_location="cpu")
                neg_emb = neg_emb.to(self.init_device)
            except Exception as e:
                print(f"[SeedVRRunner] Failed to load neg_emb: {e}")
                neg_emb = None
        else:
            neg_emb = None

        # Return text encoder output
        text_encoder_output = {
            "texts_pos": [pos_emb],
            "texts_neg": [neg_emb],
        }
        self.text_encoder_output = text_encoder_output

        return text_encoder_output

    def run_image_encoder(self, img):
        """SeedVR SR task doesn't use separate image encoder."""
        return None

    def get_latent_shape_with_lat_hw(self, latent_h, latent_w):
        """Get latent shape for SeedVR.

        Args:
            latent_h: Latent height
            latent_w: Latent width

        Returns:
            [num_channels_latents, latent_h, latent_w]
        """
        latent_shape = [
            self.num_channels_latents,
            latent_h,
            latent_w,
        ]
        return latent_shape

    def get_condition(self, latent: Tensor, latent_blur: Tensor, task: str) -> Tensor:
        t, h, w, c = latent.shape
        cond = torch.zeros([t, h, w, c + 1], device=latent.device, dtype=latent.dtype)
        if task == "t2v" or t == 1:
            # t2i or t2v generation.
            if task == "sr":
                cond[:, ..., :-1] = latent_blur[:]
                cond[:, ..., -1:] = 1.0
            return cond
        if task == "i2v":
            # i2v generation.
            cond[:1, ..., :-1] = latent[:1]
            cond[:1, ..., -1:] = 1.0
            return cond
        if task == "v2v":
            # v2v frame extension.
            cond[:2, ..., :-1] = latent[:2]
            cond[:2, ..., -1:] = 1.0
            return cond
        if task == "sr":
            # sr generation.
            cond[:, ..., :-1] = latent_blur[:]
            cond[:, ..., -1:] = 1.0
            return cond
        raise NotImplementedError

    def _run_input_encoder_local_sr(self):
        """Run input encoder for SR task.

        Args:
            input_info: Input information

        Returns:
            Dictionary with encoder outputs
        """
        # Read input video/image
        # Check video_path first (priority for SR task)
        if "video_path" in self.input_info.__dataclass_fields__ and self.input_info.video_path:
            video_path = self.input_info.video_path
            read_video = _get_read_video()

            if getattr(self, "_sr_segment", None) is not None:
                start_idx, end_idx = self._sr_segment
                video = self._read_video_segment(video_path, start_idx, end_idx)
            else:
                video, _, info = read_video(video_path, output_format="TCHW")
                if info is not None:
                    self._set_output_fps(info.get("video_fps", None))
            if video.numel() == 0:
                raise ValueError(f"Failed to read video from {video_path}")

            img = video.to(GET_DTYPE()).div_(255.0).to(self.init_device)

        elif "image_path" in self.input_info.__dataclass_fields__ and self.input_info.image_path:
            from PIL import Image

            img_path = self.input_info.image_path
            img = Image.open(img_path).convert("RGB")
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            img = img.unsqueeze(0)  # [1, C, H, W]
            img = img.to(self.init_device)
        else:
            raise ValueError("SR task requires image_path or video_path")

        # Apply SeedVR-style video transforms
        _, _, ori_h, ori_w = img.shape
        self.ori_h = ori_h
        self.ori_w = ori_w
        img = self._build_video_transform(img)
        self._input = img
        self._ori_length = img.shape[1]

        # Apply cut_videos and add_noise similar to original logic
        sp_size = 1
        img = self._cut_videos(img, sp_size)
        cond_latents = [img]
        cond_latents = self.vae_encoder.vae_encode(cond_latents)
        text_encoder_output = self.run_text_encoder(self.input_info)

        noise_gen = self._sr_noise_generator(cond_latents[0].device if cond_latents else None)
        noises = [self._sr_randn_like(latent, noise_gen) for latent in cond_latents]
        aug_noises = [self._sr_randn_like(latent, noise_gen) for latent in cond_latents]
        conditions = [
            self.get_condition(
                noise,
                task="sr",
                latent_blur=self.scheduler._add_noise(latent_blur, aug_noise),
            )
            for noise, aug_noise, latent_blur in zip(noises, aug_noises, cond_latents)
        ]

        # # Get latent shape
        # B, C, T, H, W = cond_latent.shape
        # latent_shape = [B, C, T, H, W]
        # self.input_info.latent_shape = latent_shape  # Important: set latent_shape in input_info

        torch.cuda.empty_cache()
        gc.collect()

        first_latent = cond_latents[0]
        latent_shape = [1, first_latent.shape[-1], first_latent.shape[0], first_latent.shape[1], first_latent.shape[2]]

        return {
            "x": cond_latents[0],
            "conditions": conditions,
            "noises": noises,
            "vae_encoder_out": cond_latents[0],
            "image_encoder_output": None,
            "text_encoder_output": text_encoder_output,
            "latent_shape": latent_shape,
        }

    @ProfilingContext4DebugL1("RUN pipeline")
    def run_pipeline(self, input_info):
        self.input_info = input_info

        if self.config["use_prompt_enhancer"]:
            self.input_info.prompt_enhanced = self.post_prompt_enhancer()

        def _run_unsegmented():
            self.inputs = self.run_input_encoder()
            return self.run_main()

        video_path = getattr(self.input_info, "video_path", "")
        seg_len, overlap = self._get_sr_segment_params()
        if not video_path or seg_len is None:
            return self._sr_run_on_rank0(_run_unsegmented)

        total_frames, fps, pts = self._probe_video(video_path)
        if total_frames <= seg_len or total_frames == 0:
            return self._sr_run_on_rank0(_run_unsegmented)

        self._sr_fps = fps
        self._sr_pts = pts
        segments = self._build_sr_segments(total_frames, seg_len, overlap)
        logger.info(f"[SeedVRRunner] SR segmenting: total_frames={total_frames}, seg_len={seg_len}, overlap={overlap}, segments={len(segments)}")

        original_save_path = self.input_info.save_result_path
        file_output = bool(original_save_path) and not bool(self.input_info.return_result_tensor)
        save_fps = self.config.get("fps", 16)
        vfi_target = self._sr_vfi_target(save_fps)
        if vfi_target:
            logger.info(f"[SeedVRRunner] SR VFI enabled: {save_fps} -> {vfi_target} fps (per-segment, global-grid stitched)")

        seg_rank, seg_world = self._sr_seg_parallel_info(len(segments), file_output, vfi_target)
        if seg_world > 1:
            return self._run_sr_segments(segments, seg_rank, seg_world)
        # world==1 runs inline; world>1 with segment parallel ruled out has to be
        # pinned to rank 0, or every rank races to write the same output file.
        return self._sr_run_on_rank0(lambda: self._run_sr_segments(segments, 0, 1))

    def _run_sr_segments(self, segments, seg_rank, seg_world):
        """Diffuse ``segments``, cross-fade the boundaries, emit one video.

        With ``seg_world > 1`` this rank only diffuses the segments it owns and
        the boundary frames are exchanged over the process group; rank 0 does
        the final concat once everyone's segment files are on disk.
        """
        original_save_path = self.input_info.save_result_path
        original_return_tensor = self.input_info.return_result_tensor
        file_output = bool(original_save_path) and not bool(original_return_tensor)
        raw_segments = [] if not file_output else None
        segment_paths = []
        tmp_dir = None
        save_fps = self.config.get("fps", 16)
        vfi_target = self._sr_vfi_target(save_fps)
        vfi_prev_tail = None
        vfi_src_offset = 0.0  # 当前段 images[0] 的全局源帧索引
        vfi_next_g = 0  # 下一个待发的全局目标帧索引（跨段累进，不重启）
        outer_rank_local = self._sr_enter_rank_local(seg_world)
        # Tail hand-offs never block on the peer; see _sr_send_tail. Both of
        # these are set for real below, once seg_world > 1 has a scratch dir.
        # Anything left in the send list belongs to a previous request that
        # failed: released here rather than in that request's teardown, because
        # a gloo SendWork owns the only reference to its payload and dropping it
        # mid-flight aborts the transfer under a peer that may still be reading
        # it. By now that peer has long since left the failed request.
        if getattr(self, "_sr_tail_sends", None):
            logger.warning(f"[SeedVRRunner] releasing {len(self._sr_tail_sends)} tail hand-off(s) abandoned by a failed request")
        self._sr_tail_sends = []
        self._sr_tail_dir = None
        # False while this rank still owes its peers a rendezvous. Every exit
        # path checks it, so the count of rendezvous per request stays equal
        # across ranks however the request ends.
        agreed = False
        try:
            if file_output:
                output_dir = os.path.dirname(original_save_path) or "."
                os.makedirs(output_dir, exist_ok=True)
                if seg_world > 1:
                    # All ranks write into ONE scratch dir so rank 0 can collect
                    # the segments, so the name has to be derivable rather than
                    # random. Rank 0 clears it first: a glob is how the segments
                    # are collected, and leftovers from a crashed run would be
                    # spliced into this one's output.
                    tmp_dir = os.path.join(output_dir, f".{os.path.basename(original_save_path)}.segments")
                    setup_error = None
                    if seg_rank == 0:
                        try:
                            shutil.rmtree(tmp_dir, ignore_errors=True)
                            os.makedirs(tmp_dir, exist_ok=True)
                        except Exception as e:
                            # A read-only or full output filesystem fails here on
                            # rank 0 alone. Raising straight out would leave the
                            # peers meeting an absent rank; carry the failure
                            # into the rendezvous below instead, which is the
                            # first place they can be told.
                            setup_error = e
                    # Doubles as the guarantee that the dir exists before anyone
                    # writes into it. Marked agreed across the call, not after:
                    # if it raises, it raised on every rank at once and none of
                    # them owes another. Cleared again only once this rank knows
                    # it is not the one that failed.
                    agreed = True
                    self._sr_seg_rendezvous(failed=setup_error is not None)
                    if setup_error is not None:
                        raise setup_error
                    agreed = False
                    # The tail transport is set up here, past the rendezvous that
                    # guarantees the dir exists and on a line every rank runs:
                    # new_group is collective, and a rank owning no segment
                    # never reaches a hand-off to create the group lazily.
                    self._sr_tail_dir = tmp_dir
                    if self._sr_tail_transport() == "gloo":
                        self._sr_tail_group()
                else:
                    tmp_dir = tempfile.mkdtemp(prefix=f".{os.path.basename(original_save_path)}.segments.", dir=output_dir)
            else:
                self.input_info.save_result_path = ""
                self.input_info.return_result_tensor = True

            # 段边界 cross-fade：重叠帧被两段各自独立去噪，两份结果对细节的
            # 「幻觉」不一致，硬切会把两个不同实现放在相邻帧上（seg81 在 3.3s
            # 处的跳变）。做法是把上一段的重叠尾巴扣下来（pending_tail，暂存
            # CPU 以免抬高 VAE 阶段的显存尖峰），与本段同一批帧按线性权重融合
            # 后再发出。边界宽度按 segments 元组现算，不用 overlap 常量——
            # _build_sr_segments 可能因夹紧而给出更窄的边界。
            pending_tail = None
            # 进度按**本 rank 自己的段序列**折算，不是全局段号：只有 rank 0 能上报
            # （worker._attach_progress_callback 把其余 rank 的回调置空），而段的归属
            # 是 idx % world —— 用全局段号的话，最后一段只要不归 rank 0（4 卡时 3/4
            # 的概率），进度就永远停在末尾附近直到任务突然结束。段并行是均衡分配 +
            # tail 交换同步的，本 rank 的局部进度是全局进度的良好估计，而且一定走得到
            # 100%。rank 0 提前跑完不会显示「满了却没结束」：这里报的是 denoise 阶段，
            # 门面还要按 PHASE_WEIGHTS 折进全局，后面还有 decode / save 继续推进。
            local_total = sum(1 for i in range(len(segments)) if seg_world <= 1 or self._sr_segment_owner(i, seg_world) == seg_rank)
            local_done = 0
            for idx, (start_idx, end_idx) in enumerate(segments):
                if seg_world > 1 and self._sr_segment_owner(idx, seg_world) != seg_rank:
                    continue
                where = f" [rank {seg_rank}]" if seg_world > 1 else ""
                logger.info(f"[SeedVRRunner] Processing segment {idx + 1}/{len(segments)}{where}: frames {start_idx}:{end_idx}")
                self._sr_segment = (start_idx, end_idx)
                self._sr_segment_index = idx
                self.inputs = self.run_input_encoder()
                raw = self._run_sr_single_segment(local_done, local_total)
                local_done += 1
                if seg_world > 1 and idx > 0:
                    # Posted after this rank's own diffusion so the two sides
                    # meet: the previous segment's owner was computing in
                    # lockstep and hands off the moment it finishes.
                    pending_tail = self._sr_recv_tail(self._sr_segment_owner(idx - 1, seg_world), idx - 1)
                if raw is not None:
                    if pending_tail is not None:
                        raw = self._blend_sr_overlap(pending_tail, raw)
                        pending_tail = None
                    if idx + 1 < len(segments):
                        # 扣下与下一段共享的帧，等下一段算完再融合发出。
                        hold = min(end_idx - segments[idx + 1][0], raw.shape[2] - 1)
                        if hold > 0:
                            pending_tail = raw[:, :, -hold:].detach().to("cpu", copy=True)
                            raw = raw[:, :, : raw.shape[2] - hold]
                elif pending_tail is not None:
                    # This segment produced nothing, so the held-back frames have
                    # no partner to fade into and are no longer adjacent to
                    # whatever comes next. Drop them rather than splice them onto
                    # a non-neighbouring segment.
                    logger.warning(f"[SeedVRRunner] segment {idx + 1}/{len(segments)} produced no frames; discarding {pending_tail.shape[2]} held-back boundary frames")
                    pending_tail = None

                if seg_world > 1 and idx + 1 < len(segments):
                    # Hand the held-back frames to the next segment's owner.
                    # Unconditional: the receiver posts a matching recv either
                    # way, and a zero-length meta tells it there was no tail.
                    self._sr_send_tail(pending_tail, self._sr_segment_owner(idx + 1, seg_world), idx)
                    pending_tail = None

                if file_output:
                    segment_path = os.path.join(tmp_dir, f"segment_{idx:05d}.mp4")
                    if vfi_target:
                        comfy = wan_vae_to_comfy(raw).float().clamp(0.0, 1.0)
                        del raw
                        raw = None
                        # images[0] 为上一段末帧（首段为本段首帧），供 RIFE 补齐段
                        # 边界间的中间帧。全局目标栅格从 vfi_next_g 累进，边界帧只由
                        # 上一段发出一次，本段从其后继续——无重发、无遗漏、无相位重启。
                        tail = comfy[-1:].clone()
                        if vfi_prev_tail is None:
                            frames = comfy
                        else:
                            frames = torch.cat([vfi_prev_tail, comfy], dim=0)
                        del comfy
                        n_local = frames.shape[0]
                        # 本段负责的最后一个全局目标帧（其全局源位落在本段范围内）
                        g_end = int((vfi_src_offset + n_local - 1) * vfi_target / save_fps + 1e-6)
                        out = self.vfi_model.interpolate_frames(
                            frames,
                            source_fps=save_fps,
                            target_fps=vfi_target,
                            source_frame_offset=vfi_src_offset,
                            target_idx_start=vfi_next_g,
                            target_idx_end=g_end,
                        )
                        del frames
                        vfi_next_g = g_end + 1
                        # 下一段 images[0] = 本段末帧，其全局源索引 = 本段末帧全局索引
                        vfi_src_offset = vfi_src_offset + (n_local - 1)
                        vfi_prev_tail = tail
                        save_to_video(out, segment_path, fps=vfi_target, method="ffmpeg")
                        del out
                    else:
                        self._save_sr_segment_video(raw, segment_path, fps=save_fps)
                    segment_paths.append(segment_path)
                    del raw
                    self.gen_video = None
                    self.gen_video_final = None
                    self._input = None
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    raw_segments.append(raw)

            if seg_world > 1:
                # The last tail is still in flight. Settle it before the
                # rendezvous, which is the first point every rank is known to
                # have arrived. Bounded by the tail group's timeout, so a peer
                # that died without posting its recv surfaces as this rank's
                # failure at the rendezvous rather than as a silent hang.
                self._sr_drain_tail_sends()
                # Every rank's segment files are closed and visible after this,
                # and a rank that died mid-segment is reported rather than
                # leaving a hole for the concat to find.
                # Marked before the call, not after: the rendezvous raises when
                # it agrees on a stop or a peer failure, and re-entering it from
                # the handler would leave this rank one collective ahead.
                agreed = True
                self._sr_seg_rendezvous(failed=False)
                if seg_rank != 0:
                    return {"video": None, "save_result_path": original_save_path}
                # Segment files are zero-padded and named by global index, so
                # sorting the scratch dir restores playback order regardless of
                # which rank produced what.
                segment_paths = [os.path.join(tmp_dir, name) for name in sorted(os.listdir(tmp_dir)) if name.startswith("segment_") and name.endswith(".mp4")]
                if len(segment_paths) != len(segments):
                    raise RuntimeError(f"SeedVR seg_parallel expected {len(segments)} segment files, found {len(segment_paths)} in {tmp_dir}")

            if file_output:
                if not segment_paths:
                    raise RuntimeError("SeedVR produced no video segments to save.")
                self._concat_sr_segment_videos(segment_paths, original_save_path)
                input_video_path = getattr(self.input_info, "video_path", "")
                if input_video_path:
                    mux_audio_from_video(input_video_path, original_save_path)
                logger.info(f"✅ Video saved successfully to: {original_save_path} ✅")
                return {"video": None, "save_result_path": original_save_path}
        except TaskStopped:
            # Cancelled, not broken: the peers must learn of it, but this rank
            # did not leave a hole behind.
            if seg_world > 1 and not agreed:
                agreed = True
                self._sr_seg_rendezvous(failed=False)
            raise
        except Exception:
            if seg_world > 1 and not agreed:
                # The peers are waiting on a rendezvous only this rank can
                # complete. Report the failure rather than let them sit out the
                # control-plane timeout, then re-raise the real traceback.
                agreed = True
                try:
                    self._sr_seg_rendezvous(failed=True)
                except Exception as e:
                    logger.warning(f"[SeedVRRunner] could not report this rank's failure to its peers: {e}")
            raise
        finally:
            # Critical: restore per-request output mode even when cancelled/interrupted.
            # Restored to the caller's setting, not to False: _sr_run_on_rank0
            # is still rank-local around this call and clearing it here would
            # strand rank 0 in the next all-reduce.
            self._rank_local_collectives = outer_rank_local
            # Deliberately NOT cleared, and deliberately not waited on either.
            # Waiting would bury this rank's traceback under fifteen idle
            # minutes when the peer is dead and will never post its recv;
            # clearing would drop the last reference to a gloo SendWork, whose
            # destructor tears the buffer down under a peer that may still be
            # reading it. So an abandoned send is simply left owned by the
            # runner (~200 MiB of host memory, error path only) and released at
            # the top of the next request, by which point the rendezvous above
            # has proved every peer is out of this one.
            self._sr_tail_dir = None
            self._sr_segment = None
            self._sr_segment_index = 0
            self.input_info.save_result_path = original_save_path
            self.input_info.return_result_tensor = original_return_tensor
            # Under seg_parallel the scratch dir is shared, so only rank 0 --
            # which is the one still reading it -- may remove it.
            if tmp_dir is not None and (seg_world == 1 or seg_rank == 0) and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)

        self.gen_video_final = torch.cat(raw_segments, dim=2)
        gen_video_final = self.process_images_after_vae_decoder()
        self.end_run()
        return gen_video_final
