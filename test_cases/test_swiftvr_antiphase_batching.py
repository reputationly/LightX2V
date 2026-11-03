"""反相融合的逐批实现必须与整块实现逐位相同。

第 2 轮验收 §2:双跑以前用 ``output_batch_size=0``(整块解码)只为拿到块尾做 ring carry,
代价是 output_layers 一次性处理整块,allocator 把大块 transient 缓存成 reserved
(4C/4K 实测 +5.3GiB)。``AntiphaseBlender`` 本身是流式算法,所以改成分批跑后必须保证
结果**逐位不变**。这里直接对张量算:同一块切成若干批,push 的拼接 == 整块 push。
"""

import pytest
import torch

from lightx2v.models.networks.swiftvr.antiphase import AntiphaseBlender


def _chunk(frames, seed=0):
    generator = torch.Generator().manual_seed(seed)
    shape = (1, frames, 3, 2, 2)  # (B, T, C, H, W):dim 1 是时间轴
    return torch.randn(shape, generator=generator), torch.randn(shape, generator=generator)


def _push_all(blender, direct, shifted, batch_size):
    """按 batch_size 分批 push,返回拼接后的输出。"""
    parts = [blender.push(d, s) for d, s in zip(direct.split(batch_size, dim=1), shifted.split(batch_size, dim=1))]
    return torch.cat(parts, dim=1)


@pytest.mark.parametrize("frames", [4, 8, 12, 24, 28])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batched_push_is_bit_identical_to_whole_chunk(frames, batch_size):
    direct, shifted = _chunk(frames)

    whole = AntiphaseBlender()
    expected = whole.push(direct, shifted)

    batched = AntiphaseBlender()
    got = _push_all(batched, direct, shifted, batch_size)

    assert torch.equal(got, expected)
    # carry 也必须一致:块尾一帧原样留给下一块
    assert torch.equal(batched.carry, whole.carry)
    assert whole.carry.shape[1] == 1


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_push_with_inbound_carry(batch_size):
    """块首带上一块传来的 carry 时,分批也要给出同样的结果。"""
    direct, shifted = _chunk(12, seed=3)
    carry = torch.randn(1, 1, 3, 2, 2, generator=torch.Generator().manual_seed(99))

    whole = AntiphaseBlender()
    whole.carry = carry.clone()
    expected = whole.push(direct, shifted)

    batched = AntiphaseBlender()
    batched.carry = carry.clone()
    got = _push_all(batched, direct, shifted, batch_size)

    assert torch.equal(got, expected)
    assert torch.equal(batched.carry, whole.carry)


@pytest.mark.parametrize("batch_size", [1, 4])
def test_batched_sequence_across_chunks_matches_whole(batch_size):
    """跨多块 + 末块 flush:分批与整块的整条序列逐位相同。"""
    chunks = [_chunk(frames, seed=index) for index, frames in enumerate((8, 8, 4))]

    whole = AntiphaseBlender()
    expected_parts = [whole.push(d, s) for d, s in chunks]
    expected_parts.append(whole.flush())

    batched = AntiphaseBlender()
    got_parts = [_push_all(batched, d, s, batch_size) for d, s in chunks]
    got_parts.append(batched.flush())

    expected = torch.cat(expected_parts, dim=1)
    got = torch.cat(got_parts, dim=1)
    assert torch.equal(got, expected)
    # 首块少写一帧(-1)、末块 flush 补回(+1) → 总帧数等于各块长度之和
    assert got.shape[1] == sum(d.shape[1] for d, _ in chunks)
