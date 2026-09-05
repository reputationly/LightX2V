"""Causal chunking and streaming state for SwiftVR restoration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch

from .reae import StreamingAutoencoder


class ChunkType(Enum):
    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


@dataclass(frozen=True)
class VideoChunk:
    type: ChunkType
    start: int
    frame_count: int
    index: int

    @property
    def is_first(self) -> bool:
        return self.index == 0

    @property
    def is_last(self) -> bool:
        return self.type is ChunkType.LAST

    @property
    def latent_count(self) -> int:
        return (self.frame_count - 1) // 4 + 1


def padded_frame_count(frame_count: int) -> int:
    """Return the smallest ``4k+1`` count that contains all input frames."""

    return ((frame_count - 1 + 3) // 4) * 4 + 1


def build_video_chunks(frame_count: int, clip_length: int) -> list[VideoChunk]:
    if clip_length % 4:
        raise ValueError(f"SwiftVR clip_len must be a multiple of 4, got {clip_length}")
    if frame_count <= clip_length + 4:
        return [VideoChunk(ChunkType.LAST, 0, frame_count, 0)]

    chunks = [VideoChunk(ChunkType.FIRST, 0, clip_length + 4, 0)]
    start = clip_length + 4
    while start < frame_count:
        remaining = frame_count - start
        chunk_type = ChunkType.LAST if remaining <= clip_length else ChunkType.MIDDLE
        chunk_frames = remaining if chunk_type is ChunkType.LAST else clip_length
        chunks.append(VideoChunk(chunk_type, start, chunk_frames, len(chunks)))
        start += chunk_frames
    return chunks


class StreamingTransformer:
    def __init__(self, model, condition, overlap: int = 0, strength: float = 1.0):
        self.model = model
        self.condition = condition
        self.overlap = overlap
        # 恢复强度:给 DiT 预测的**残差**乘一个系数(见 restore:输出 = 输入 - strength*残差)。
        #   1.0  = 上游原行为
        #   <1.0 = 弱化恢复。用来压住「把源本身的缺陷(水波纹、压缩噪点、颗粒)当成要恢复的
        #          细节一起放大」—— SwiftVR 在**重退化**数据上训过,喂它轻退化素材(AIGC
        #          直出、干净拍摄)必然过度生成,是这类生成式修复的通病,SeedVR2 官方也警告过。
        #          实测:雨林微距锐度 821(源) -> 1848,远超"恢复"该有的量。
        #   0.0  = 残差置零,但**仍然走 ReAE 编解码** —— 拿到的是一次自编码往返,**不是纯放大**。
        #          「两帧一顿」(2beat)长在 ReAE 解码器的 TemporalGrow 里,把 strength 关到 0
        #          也修不掉:只跑 ReAE 实测 2beat 0.5639,而双线性对照是 1.0000。GPU 也一分不省。
        #          要真·纯放大请走 ffmpeg,别指望这个旋钮。
        #   >1.0 = 见 runner 的 antiphase_strength_gain:反相平均会削掉约三成高频,双跑路径
        #          靠放大残差补回来。它是引擎补偿,不是「用户想要更强的恢复」。
        self.strength = float(strength)
        self.reset()

    def reset(self):
        self.temporal_offset = 0
        self.previous_input = None
        self.previous_latents = None

    def seed_temporal_offset(self, latent_offset: int):
        """Start this stream as if ``latent_offset`` latent frames had already been consumed.

        Only used by segment parallel: a rank that begins mid-video must place its
        chunks at their *global* RoPE positions, otherwise every rank would restart
        the temporal axis at 0 and the joined output would drift at every seam.
        The causal AE state is rebuilt by actually running a warm-up chunk; only
        this counter cannot be recovered that way, because it is pure bookkeeping.
        """
        self.temporal_offset = int(latent_offset)

    @torch.inference_mode()
    def restore(self, latents: torch.Tensor, clip_latents: int) -> torch.Tensor:
        low_quality = latents.permute(0, 2, 1, 3, 4).contiguous()
        overlap = self.previous_input.shape[2] if self.previous_input is not None and self.overlap else 0
        model_input = torch.cat([self.previous_input.to(low_quality.device), low_quality], dim=2) if overlap else low_quality

        restored = model_input - self.strength * self.model.predict(model_input, self.condition, self.temporal_offset - overlap)
        if overlap:
            restored = restored[:, :, overlap:]

        keep = min(self.overlap, low_quality.shape[2])
        self.previous_input = low_quality[:, :, -keep:].detach().cpu().clone() if keep else None
        self.previous_latents = low_quality[:, :, -clip_latents:].detach().cpu().clone()
        self.temporal_offset += low_quality.shape[2]
        return restored.permute(0, 2, 1, 3, 4).contiguous()

    @torch.inference_mode()
    def restore_last(self, latents: torch.Tensor, latent_count: int, clip_latents: int) -> torch.Tensor:
        low_quality = latents.permute(0, 2, 1, 3, 4).contiguous()
        padding = clip_latents + 1 - latent_count
        if padding:
            if self.previous_latents is None:
                prefix = torch.zeros(
                    low_quality.shape[0],
                    low_quality.shape[1],
                    padding,
                    low_quality.shape[3],
                    low_quality.shape[4],
                    dtype=low_quality.dtype,
                    device=low_quality.device,
                )
            else:
                prefix = self.previous_latents[:, :, -padding:].to(low_quality.device)
            low_quality = torch.cat([prefix, low_quality], dim=2)

        restored = low_quality - self.strength * self.model.predict(
            low_quality,
            self.condition,
            max(0, self.temporal_offset - padding),
        )
        self.temporal_offset += latent_count
        return restored[:, :, -latent_count:].permute(0, 2, 1, 3, 4).contiguous()


class AntiphaseBlender:
    """把「正相」与「移位一帧」两条流的输出对齐后逐帧平均,用来对消 2beat。

    ReAE 解码器最后一个 `TemporalGrow(stride=2)` 把 3 抽头时间卷积关在每一对帧内部
    (`x.unsqueeze(2)` 让帧轴变成 batch 轴),之后再无时间混合,输出帧因此天生两两成对 ——
    匀速输入也会「走两帧顿一下」。把输入整体后移 1 帧(奇数)就把配对格点的相位翻过来,
    两条流平均即可对消这个周期 2 分量。缺陷在官方模型本体,改 TemporalGrow 试过,画质更差。

    移位流的第 j 帧对应正相流的第 j-1 帧,所以配对天然差一格:本块最后一帧要等下一块的
    移位流才能配上。`carry` 就是这一帧,`flush` 交出全片最后那一帧 —— 它没有配对对象,
    因为移位流比正相流短一帧,只能原样交付。
    """

    def __init__(self):
        self.carry = None

    def reset(self):
        self.carry = None

    def push(self, direct: torch.Tensor, shifted: torch.Tensor) -> torch.Tensor:
        if self.carry is None:
            # 首块没有 carry。这里必须 clone:下面要就地累加,而 direct[:, :-1] 是
            # direct 的视图,就地写会把调用方手里的 direct 一起改掉。
            paired, partner = direct[:, :-1].clone(), shifted[:, 1:]
        else:
            paired, partner = torch.cat([self.carry, direct[:, :-1]], dim=1), shifted
        # clone 而不是留视图:切片会把整块 direct 的显存钉住(1080p/24 帧约 300MB,
        # 4K 约 1.19GB),拷成独立的一帧只要十几 MB。
        self.carry = direct[:, -1:].clone()
        # 就地累加再就地折半。写成 (paired + partner) * 0.5 会多分配两整块:
        # 加法一块、乘法一块,叠加上仍存活的 direct/shifted/paired,峰值瞬间是 5 块。
        # 1080p 每块约 300MB 时还淹没在 DiT 峰值下面,4K 每块 1.19GB 就会顶出来。
        # paired 两个分支都自有存储,partner 是 shifted 的视图,存储不重叠。
        return paired.add_(partner).mul_(0.5)

    def flush(self):
        carry, self.carry = self.carry, None
        return carry


class SwiftVRRestorer:
    def __init__(
        self,
        autoencoder,
        model,
        prompt_embedding,
        overlap: int = 0,
        reae_frame_batch_size: int = 1,
        strength: float = 1.0,
    ):
        self.autoencoder = StreamingAutoencoder(autoencoder, reae_frame_batch_size)
        self.transformer = StreamingTransformer(model, model.prepare_condition(prompt_embedding), overlap, strength)

    def reset(self):
        self.autoencoder.reset()
        self.transformer.reset()

    def seed_temporal_offset(self, latent_offset: int):
        self.transformer.seed_temporal_offset(latent_offset)

    @torch.inference_mode()
    def restore_chunk(self, video: torch.Tensor, chunk: VideoChunk, clip_latents: int) -> torch.Tensor:
        latents = self.autoencoder.encode(video, chunk.is_last)
        if chunk.is_last:
            latents = self.transformer.restore_last(latents, chunk.latent_count, clip_latents)
        else:
            latents = self.transformer.restore(latents, clip_latents)
        return self.autoencoder.decode(latents, chunk.is_first)
