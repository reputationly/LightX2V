"""反相双跑 × 块并行:本地融合 + ring carry 必须与单 rank 顺序融合逐帧相等。

这一层是纯张量语义。P=1 时 upstream 的 ``chunk_p_group`` 是 None(不交换),carry 在
同一个 blender 里自然传递;P>1 时块 k 落在 rank ``k % P`` 上,块末帧当 carry 沿 ring
交给下一个 rank,rank0 收到的那份留给下一个 batch。
"""

from collections import defaultdict, deque

import pytest
import torch

from lightx2v.models.networks.swiftvr import antiphase


class _Group:
    """只要一个 rank 号的进程组替身(真实现是 torch 的 ProcessGroup)。"""

    def __init__(self, rank, world):
        self.rank = rank
        self.world = world


@pytest.fixture()
def ring(monkeypatch):
    """把 RingComm 换成可观察的 ring:rank r 收到 rank r-1 在本 batch 发的那份。

    ``staged[r]`` 由测试在每次交换前按 ring 语义摆好 —— 真实实现里它是
    ``enqueue_send_recv`` 的匹配结果。
    """
    staged = {}
    sent = defaultdict(deque)

    def fake_exchange(tensor, group):
        sent[group.rank].append(tensor.clone())
        return staged[group.rank]

    monkeypatch.setattr(antiphase, "exchange_chunk_boundary", fake_exchange)
    monkeypatch.setattr(antiphase.dist, "get_rank", lambda group: group.rank)
    return sent, staged


def _chunks(count, *, frames=(5, 3, 4, 3), seed=0):
    """造 ``count`` 块 (direct, shifted),块长按 ``frames`` 循环。"""
    generator = torch.Generator().manual_seed(seed)
    out = []
    for index in range(count):
        shape = (1, frames[index % len(frames)], 3, 2, 2)  # (B, T, C, H, W):dim 1 是时间轴
        out.append((torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)))
    return out


def _sequential(chunks):
    """单 rank、单流:顺序 push,最后 flush 交出没有配对对象的末帧。"""
    blender = antiphase.AntiphaseBlender()
    out = [blender.push(direct, shifted) for direct, shifted in chunks]
    out.append(blender.flush())
    return torch.cat(out, dim=1)


def _parallel(chunks, world, ring):
    """模拟 runner:每个 batch 各 rank 交换一次 carry,然后本地融合。"""
    sent, staged = ring
    batches = (len(chunks) + world - 1) // world
    blenders = [antiphase.AntiphaseBlender() for _ in range(world)]
    stored = [None] * world
    results = {}

    for batch in range(batches):
        indices = [batch * world + rank for rank in range(world)]
        carries = {}
        for rank, index in enumerate(indices):
            direct = chunks[min(index, len(chunks) - 1)][0]
            active = index < len(chunks)
            # inactive 的重复块也要发同形零张量:ring 上少一次交换,其余 rank 就卡在 P2P
            carries[rank] = direct[:, -1:].clone() if active else direct.new_zeros((direct.shape[0], 1, *direct.shape[2:]))

        if world > 1:
            for rank in range(world):
                staged[rank] = carries[(rank - 1) % world]

        for rank, index in enumerate(indices):
            if index >= len(chunks):
                continue
            direct, shifted = chunks[index]
            if world > 1:
                group = _Group(rank, world)
                chunk_carry, stored[rank] = antiphase.exchange_antiphase_carry(carries[rank], group, stored[rank])
                blenders[rank].carry = chunk_carry
            results[index] = blenders[rank].push(direct, shifted)

    final_blender = blenders[(len(chunks) - 1) % world]
    return results, final_blender, sent


def _assemble(results, final_blender, count):
    return torch.cat([results[index] for index in range(count)] + [final_blender.flush()], dim=1)


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("count", [1, 3, 4, 7])
def test_parallel_matches_single_rank_frame_by_frame(world, count, ring):
    chunks = _chunks(count)
    expected = _sequential(chunks)

    results, final_blender, _sent = _parallel(chunks, world, ring)
    got = _assemble(results, final_blender, count)

    assert torch.equal(got, expected)
    # 首块少写一帧(-1)、末块 flush 补回一帧(+1),整片帧数与块长之和相同
    assert got.shape[1] == sum(direct.shape[1] for direct, _ in chunks)


@pytest.mark.parametrize("world", [2, 4])
@pytest.mark.parametrize("count", [3, 4, 7])
def test_carry_chain_matches_the_previous_chunk_tail(world, count, ring):
    """rank r 发到 ring 上的第 j 份必须是块 ``j*world + r`` 的末帧。"""
    chunks = _chunks(count)
    _parallel(chunks, world, ring)
    sent = ring[0]

    for rank in range(world):
        for batch_index, payload in enumerate(sent[rank]):
            index = batch_index * world + rank
            if index >= len(chunks):
                continue
            assert torch.equal(payload, chunks[index][0][:, -1:]), f"rank {rank} batch {batch_index} sent the wrong tail"


@pytest.mark.parametrize("world", [1, 2, 4])
def test_the_global_first_chunk_gets_no_carry(world, ring, monkeypatch):
    """全局 chunk.index==0 前面没有帧:那次 push 拿到的 carry 必须是 None。"""
    seen = []
    original_push = antiphase.AntiphaseBlender.push

    def spy(self, direct, shifted):
        seen.append(self.carry)
        return original_push(self, direct, shifted)

    monkeypatch.setattr(antiphase.AntiphaseBlender, "push", spy)

    chunks = _chunks(3)
    _parallel(chunks, world, ring)

    assert seen[0] is None, "the first chunk must start without a carry"
    assert all(carry is not None for carry in seen[1:]), "every later chunk pairs against the previous tail"
    # 首块少写一帧(末帧变成下一块的 carry)
    assert seen.count(None) == 1


@pytest.mark.parametrize("world", [1, 2, 4])
def test_single_chunk_video_round_trips(world, ring):
    """一块的短片:-1 的首块与 +1 的末帧相互抵消,帧数不变。"""
    chunks = _chunks(1, frames=(6,))
    expected = _sequential(chunks)
    results, final_blender, _sent = _parallel(chunks, world, ring)
    got = _assemble(results, final_blender, 1)

    assert got.shape[1] == 6
    assert torch.equal(got, expected)
