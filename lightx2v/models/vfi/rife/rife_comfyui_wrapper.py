import os
from typing import List, Optional, Tuple

import torch
from torch.nn import functional as F

from lightx2v.utils.profiler import *


class RIFEWrapper:
    """Wrapper for RIFE model to work with ComfyUI Image tensors"""

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, model_path, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup torch for optimal performance
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True

        # Load model
        from .train_log.RIFE_HDv3 import Model

        self.model = Model()
        with ProfilingContext4DebugL2("Load RIFE model"):
            self.model.load_model(model_path, -1)
            self.model.eval()
            self.model.device()

    @ProfilingContext4DebugL2("Interpolate frames")
    def interpolate_frames(
        self,
        images: torch.Tensor,
        source_fps: float,
        target_fps: float,
        scale: float = 1.0,
        source_frame_offset: float = 0.0,
        target_idx_start: int = 0,
        target_idx_end: int = None,
    ) -> torch.Tensor:
        """
        Interpolate frames from source FPS to target FPS

        Args:
            images: ComfyUI Image tensor [N, H, W, C] in range [0, 1]
            source_fps: Source frame rate
            target_fps: Target frame rate
            scale: Scale factor for processing
            source_frame_offset: global source-frame index of images[0]. For
                segmented interpolation (SeedVR SR) this keeps the target-time
                grid continuous across segment boundaries so non-integer fps
                ratios stay in sync with muxed audio. Default 0 = whole video.
            target_idx_start / target_idx_end: global target-frame index range
                this call owns (inclusive). Callers carry them across segments.
                Defaults reproduce the original whole-video grid.

        Returns:
            Interpolated ComfyUI Image tensor [M, H, W, C] in range [0, 1]
        """
        # Validate input
        assert images.dim() == 4 and images.shape[-1] == 3, "Input must be [N, H, W, C] with C=3"

        # 契约保持：target < source 时按位置映射降采样（时长正确），
        # 与既有调用方（wan_audio_runner 录制/推流）的语义一致。
        # 「不降帧」的决策在调用点做（default_runner / seedvr_runner
        # 均只在 target > source 时才调用本函数）。
        if source_fps == target_fps:
            return images

        # RIFE (IFNet) weights are fp32; the diffusion pipeline emits bf16 frames
        # (use_bfloat16), which would hit "Input type (BFloat16) and bias type
        # (float) should be the same" in the conv layers. Cast frames to fp32 to
        # match the model (RIFE is tiny + numerically sensitive → fp32 is right).
        images = images.float()

        total_source_frames = images.shape[0]
        height, width = images.shape[1:3]

        # Calculate padding for model
        tmp = max(128, int(128 / scale))
        ph = ((height - 1) // tmp + 1) * tmp
        pw = ((width - 1) // tmp + 1) * tmp
        padding = (0, pw - width, 0, ph - height)

        # Calculate target frame positions
        frame_positions = self._calculate_target_frame_positions(
            source_fps,
            target_fps,
            total_source_frames,
            source_frame_offset=source_frame_offset,
            target_idx_start=target_idx_start,
            target_idx_end=target_idx_end,
        )

        # Prepare output tensor
        output_frames = []

        for source_idx1, source_idx2, interp_factor in frame_positions:
            if interp_factor == 0.0 or source_idx1 == source_idx2:
                # No interpolation needed, use the source frame directly
                output_frames.append(images[source_idx1])
            else:
                # Get frames to interpolate
                frame1 = images[source_idx1]
                frame2 = images[source_idx2]

                # Convert ComfyUI format [H, W, C] to RIFE format [1, C, H, W]
                # Also convert from [0, 1] to [0, 1] (already in correct range)
                I0 = frame1.permute(2, 0, 1).unsqueeze(0).to(self.device)
                I1 = frame2.permute(2, 0, 1).unsqueeze(0).to(self.device)

                # Pad images
                I0 = F.pad(I0, padding)
                I1 = F.pad(I1, padding)

                # Perform interpolation
                with torch.no_grad():
                    interpolated = self.model.inference(I0, I1, timestep=interp_factor, scale=scale)

                # Convert back to ComfyUI format [H, W, C]
                # Crop to original size and permute dimensions
                interpolated_frame = interpolated[0, :, :height, :width].permute(1, 2, 0).cpu()
                output_frames.append(interpolated_frame)

        # Stack all frames
        return torch.stack(output_frames, dim=0)

    def _calculate_target_frame_positions(
        self,
        source_fps: float,
        target_fps: float,
        total_source_frames: int,
        source_frame_offset: float = 0.0,
        target_idx_start: int = 0,
        target_idx_end: int = None,
    ) -> List[Tuple[int, int, float]]:
        """
        Calculate which frames need to be generated for the target frame rate.

        The target-frame index is GLOBAL: target frame g sits at global source
        position g * source_fps / target_fps. `source_frame_offset` is the global
        source index of images[0], so a segment maps global positions into its
        LOCAL frame indices. This keeps the cadence continuous across segments
        (no per-segment phase reset) for non-integer fps ratios.

        Defaults (offset=0, start=0, end=None) reproduce the original whole-video
        grid: end = floor((N-1) * target/source), range [0, end].

        Returns:
            List of (local_source_idx1, local_source_idx2, interpolation_factor).
        """
        if target_idx_end is None:
            seg_end_pos = source_frame_offset + (total_source_frames - 1)
            # +eps so exact-integer boundaries (integer fps ratios) don't get
            # floored down by float error and drop the last frame.
            target_idx_end = int(seg_end_pos * target_fps / source_fps + 1e-6)

        frame_positions = []
        for target_idx in range(target_idx_start, target_idx_end + 1):
            # Global source position of this target frame, mapped to local indices.
            source_position = (target_idx * source_fps / target_fps) - source_frame_offset

            source_idx1 = int(source_position)
            if source_idx1 < 0:
                source_idx1 = 0
            source_idx2 = min(source_idx1 + 1, total_source_frames - 1)

            # Calculate interpolation factor (0 means use frame1, 1 means use frame2)
            if source_idx1 == source_idx2:
                interpolation_factor = 0.0
            else:
                interpolation_factor = source_position - source_idx1

            frame_positions.append((source_idx1, source_idx2, interpolation_factor))

        return frame_positions
