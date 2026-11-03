"""RelayVSR 块级数据并行:帧区间切分与预热长度。

段并行的正确性依赖两件事:
1. 所有 rank 对「总帧数」和「自己那段」有完全一致的视图(否则 rank0 拼起来的视频长度不对);
2. 每段只在自己起点之前预热一个 window(输出丢弃),不吃别人的输出、也不留空档。
"""

import pytest

from lightx2v.models.runners.relayvsr.relayvsr_runner import RelayVSRRunner


def _spans(total, size):
    runner = RelayVSRRunner.__new__(RelayVSRRunner)
    runner.chunk_p_size = size
    return runner._chunk_spans(total)


@pytest.mark.parametrize("size", [1, 2, 4, 8])
@pytest.mark.parametrize("total", [0, 1, 15, 16, 17, 124, 374, 748, 1000])
def test_spans_tile_the_whole_video_without_gap_or_overlap(size, total):
    spans = _spans(total, size)
    assert len(spans) == size
    covered = 0
    previous_end = 0
    for start, end in spans:
        assert start == previous_end, "段必须首尾相接,不能有空档"
        assert end >= start
        covered += end - start
        previous_end = end
    assert covered == total, "所有段加起来必须正好覆盖全片"
    assert previous_end == total


@pytest.mark.parametrize("size", [2, 3, 4])
def test_tail_rank_takes_the_remainder(size):
    total = 1000
    spans = _spans(total, size)
    lengths = [end - start for start, end in spans]
    assert lengths[-1] <= lengths[0]
    assert spans[-1][1] == total


def test_warmup_is_capped_by_the_segment_start():
    """首段没有前文可预热;其余段最多预热一个 window。"""
    clip_len = 16
    spans = _spans(374, 4)
    warmups = [min(clip_len, start) for start, _ in spans]
    assert warmups[0] == 0
    assert all(0 < w <= clip_len for w in warmups[1:])
