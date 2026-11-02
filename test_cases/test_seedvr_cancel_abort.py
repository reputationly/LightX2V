"""段边界取消协议:提前离开段循环的 rank 要把还欠着的边界帧标成 abort。

背景(第 1 轮验收 A2):4 卡段并行时,rank0 在前处理阶段被取消,它没发本段的 tail 就进了
rendezvous;rank1 在 ``_sr_recv_tail`` 上等,rank2/3 依次等上游 —— 一个取消变成整机挂住。
修法是:任何 rank 提前离开段循环时,给它**还欠着的每一条边**发 abort 标记;收到 abort 的
rank 按同样规则把自己欠的边也标 abort(级联),然后统一汇合退出。
"""

from types import SimpleNamespace

import pytest
import torch

from lightx2v.models.runners.seedvr import seedvr_runner as sr


def _runner(num_segments, world, rank, tmp_path):
    """只装配协议需要的那几个属性的 runner。"""
    runner = sr.SeedVRRunner.__new__(sr.SeedVRRunner)
    runner.config = {"task": "sr"}
    runner._sr_tail_dir = str(tmp_path)  # 让 transport 守卫认为已经在段循环里
    runner._sr_tail_pg = None
    runner._sr_tail_sends = []
    runner._sr_segment_owner = lambda idx, seg_world: idx % seg_world
    return runner


def _owed_edges(num_segments, world, rank):
    """rank 欠的边 = 它拥有的段 idx 且 idx+1 < n;单卡没有段边界,一条都不欠。"""
    if world <= 1:
        return []
    return [idx for idx in range(num_segments) if idx % world == rank and idx + 1 < num_segments]


@pytest.mark.parametrize("world", [1, 2, 3, 4])
@pytest.mark.parametrize("num_segments", [1, 2, 3, 4, 5, 8, 9])
def test_abort_covers_exactly_the_edges_this_rank_owes(world, num_segments, tmp_path):
    sent = []
    for rank in range(world):
        runner = _runner(num_segments, world, rank, tmp_path)
        runner._sr_send_tail = lambda tail, dst, idx, abort=False, sent=sent: sent.append((rank, idx, dst, abort))
        edge_sent = set()
        my_indices = [i for i in range(num_segments) if i % world == rank]
        runner._sr_abort_owed_edges(my_indices, edge_sent, world, num_segments)

        mine = [(r, idx, dst, abort) for r, idx, dst, abort in sent if r == rank]
        assert [idx for _r, idx, _dst, _abort in mine] == _owed_edges(num_segments, world, rank)
        assert all(abort for *_rest, abort in mine), "abort markers must be flagged as such"
        assert all(dst == (idx + 1) % world for _r, idx, dst, _abort in mine), "each marker goes to the next segment's owner"


def test_edges_already_sent_are_not_aborted(tmp_path):
    """已经正常交出边界帧的边不能再补 abort,否则对端会把真实结果当成中止。"""
    sent = []
    runner = _runner(6, 3, 0, tmp_path)
    runner._sr_send_tail = lambda tail, dst, idx, abort=False: sent.append((idx, abort))
    my_indices = [0, 3]
    runner._sr_abort_owed_edges(my_indices, edge_sent={0}, seg_world=3, num_segments=6)
    assert sent == [(3, True)]


def test_single_card_never_sends_abort(tmp_path):
    sent = []
    runner = _runner(4, 1, 0, tmp_path)
    runner._sr_send_tail = lambda tail, dst, idx, abort=False: sent.append(idx)
    runner._sr_abort_owed_edges([0, 1, 2, 3], set(), seg_world=1, num_segments=4)
    assert sent == []


def test_no_marker_before_the_tail_transport_exists(tmp_path):
    """临时目录/进程组还没建好时会话还没开始等 tail,不该白造一个 abort。"""
    sent = []
    runner = _runner(6, 3, 0, tmp_path)
    runner._sr_tail_dir = None
    runner._sr_tail_pg = None
    runner._sr_send_tail = lambda tail, dst, idx, abort=False: sent.append(idx)
    runner._sr_abort_owed_edges([0, 3], set(), seg_world=3, num_segments=6)
    assert sent == []


def test_file_transport_round_trips_abort_and_no_tail(tmp_path):
    runner = _runner(4, 2, 0, tmp_path)
    tail = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 2, 2)

    runner._sr_write_tail_file(tail, 0)
    assert torch.equal(runner._sr_read_tail_file(0), tail)

    runner._sr_write_tail_file(None, 1)
    assert runner._sr_read_tail_file(1) is None, "a segment that produced no frames is not an abort"

    runner._sr_write_tail_file(None, 2, abort=True)
    assert runner._sr_read_tail_file(2) is sr._SR_ABORT


def test_gloo_meta_encodes_abort_and_no_tail_distinctly(monkeypatch, tmp_path):
    """abort 用 meta[0] = -1,no-tail 用 meta[0] = 0 —— 接收方靠这个区分。"""
    encoded = []

    class _FakeWork:
        def wait(self):
            return None

    fake_dist = SimpleNamespace(isend=lambda tensor, dst=None, group=None: (encoded.append(tensor.clone()), _FakeWork())[1])
    monkeypatch.setattr(sr, "dist", fake_dist)

    runner = _runner(4, 2, 0, tmp_path)
    runner._sr_tail_group = lambda: object()
    runner._sr_tail_transport = lambda: "gloo"

    runner._sr_send_tail(None, dst=1, idx=0, abort=True)
    runner._sr_send_tail(None, dst=1, idx=1)
    runner._sr_send_tail(torch.zeros(1, 2, 2, 2, 2), dst=1, idx=2)

    assert int(encoded[0][0]) == -1, "abort marker"
    assert int(encoded[1][0]) == 0, "no-tail marker"
    assert int(encoded[2][0]) == 2, "real tail: frame count"


def test_abort_sentinel_is_not_confused_with_none():
    assert sr._SR_ABORT is not None
    assert repr(sr._SR_ABORT) == "<SR_TAIL_ABORT>"
