"""反相双跑:相差一帧的两条流融合,以及对消 ReAE 2beat 的跨 rank 块尾传递。

`AntiphaseBlender` 负责单条顺序流里的配对(块内 direct[i] 配 shifted[i+1],
块末帧留给下一块);`chunk_p_size > 1` 时块按 `k % P` 交错落在各 rank 上,
配对对象在相邻 rank 手里,由 `exchange_antiphase_carry` 沿 ring 传递。
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from .parallel import exchange_chunk_boundary


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


def exchange_antiphase_carry(carry: torch.Tensor, chunk_p_group, stored: torch.Tensor | None):
    """把本块的块尾一帧交给下一个 rank,换回上一个 rank 的块尾。

    返回 ``(本块要用的 carry, 留给下一个 batch 的 carry)``。

    接收规则照抄上游 ``reae.run_causal_layers``:rank r>0 收到的正是 rank r-1 的块尾
    一帧 —— 就是本块 ``direct[i]`` 该配的那一帧;rank 0 收到的是 rank P-1 的块尾,它配
    的是**下一个 batch** 的首块,所以存起来,本 batch 用上一 batch 存下的那个。全局
    chunk.index==0 时它恒为 None(首块前面没有帧可配)。

    inactive 的重复块也必须调用(发同形零张量):ring 的每一次交换在所有 rank 上必须
    一一配对,少发一次其余 rank 就卡在 P2P 上。
    """
    received = exchange_chunk_boundary(carry.contiguous(), chunk_p_group)
    if dist.get_rank(chunk_p_group) != 0:
        return received, stored
    return stored, received
