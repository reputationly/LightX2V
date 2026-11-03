"""段边界协议必须**收发对称**:任何 rank 提前退出都不能留下没人收的消息。

第 2 轮验收 §1 的协议漏洞:提前退出的 rank 只发自己欠的 abort、不再收别人发给它的
边界帧。一旦某个 rank 拥有 ≥2 段(4 卡下段数 ≥5),就会留下没人收的 isend —— 发送方等满
15 分钟,或者这条消息被**下一个请求**收走,把下一个请求误判成 abort / 写成坏片。

这个测试用线程 + 每条边一个队列来模拟 ring,并直接驱动**真实的**
``_sr_abort_owed_edges`` / ``_sr_drain_inbound_edges``(只把 ``_sr_send_tail`` /
``_sr_recv_tail`` 换成队列),断言:
  - 每个 rank 都退出(没有死锁);
  - 每条消息都被恰好一个 recv 收走(队列全空 = 每个 (src,dst) 的 send 数 == recv 数);
  - 紧接着跑的干净请求只收到属于自己的消息。
"""

import queue
import threading
from types import SimpleNamespace

import pytest

from lightx2v.models.runners.seedvr import seedvr_runner as sr

ABORT = "ABORT"
REAL = "REAL"


class _Edge:
    """一条 (idx-1 -> idx) 边,只有一个发送方和一个接收方。"""

    def __init__(self):
        self.queue = queue.Queue()
        self.sent = 0
        self.recv = 0


class _Rig:
    """一次请求:构造 w 个 rank 的假 transport,并记录收发计数。"""

    def __init__(self, world, num_segments):
        self.world = world
        self.num_segments = num_segments
        self.edges = [_Edge() for _ in range(num_segments)]
        self.threads = []
        self.errors = []

    def owner(self, idx):
        return idx % self.world

    def make_rank(self, rank):
        runner = sr.SeedVRRunner.__new__(sr.SeedVRRunner)
        runner.config = {"task": "sr"}
        runner._sr_tail_dir = "/tmp/fake"  # 让 transport 守卫认为段循环已就绪
        runner._sr_tail_pg = None
        runner._sr_tail_sends = []
        runner._sr_segment_owner = lambda idx, seg_world, r=rank: idx % seg_world

        def send_tail(tail, dst, idx, abort=False):
            edge = self.edges[idx]
            edge.sent += 1
            edge.queue.put(ABORT if abort else REAL)

        def recv_tail(src, idx):
            edge = self.edges[idx]
            edge.recv += 1
            return edge.queue.get(timeout=5)

        runner._sr_send_tail = send_tail
        runner._sr_recv_tail = recv_tail
        return runner

    def run(self, cancel_rank=None, cancel_phase=None, raise_rank=None):
        """跑一次请求。cancel_phase 形如 ("before", idx) / ("after", idx)。"""

        def body(rank):
            runner = self.make_rank(rank)
            my_indices = [idx for idx in range(self.num_segments) if self.owner(idx) == rank]
            edge_sent, edge_recv = set(), set()
            try:
                for idx in my_indices:
                    if rank == raise_rank and cancel_phase and cancel_phase[1] == idx and cancel_phase[0] == "before":
                        raise RuntimeError("boom")
                    if rank == cancel_rank and cancel_phase == ("before", idx):
                        runner._sr_unwind_segment_loop(my_indices, edge_sent, edge_recv, self.world, self.num_segments)
                        return
                    if idx > 0:
                        message = runner._sr_recv_tail(self.owner(idx - 1), idx - 1)
                        edge_recv.add(idx)
                        if message is sr._SR_ABORT:
                            # 收到中止:把自己欠的边也中止掉,再把剩下的入边收干净
                            runner._sr_unwind_segment_loop(my_indices, edge_sent, edge_recv, self.world, self.num_segments)
                            return
                    if rank == cancel_rank and cancel_phase == ("after", idx):
                        runner._sr_unwind_segment_loop(my_indices, edge_sent, edge_recv, self.world, self.num_segments)
                        return
                    if idx + 1 < self.num_segments:
                        runner._sr_send_tail("tail", self.owner(idx + 1), idx)
                        edge_sent.add(idx)
            except Exception as error:  # noqa: BLE001 - 线程里记下来交给主线程断言
                if rank == raise_rank:
                    runner._sr_unwind_segment_loop(my_indices, edge_sent, edge_recv, self.world, self.num_segments)
                else:
                    self.errors.append((rank, error))

        self.threads = [threading.Thread(target=body, args=(rank,), daemon=True) for rank in range(self.world)]
        for thread in self.threads:
            thread.start()
        for thread in self.threads:
            thread.join(timeout=15)
        stuck = [index for index, thread in enumerate(self.threads) if thread.is_alive()]
        return stuck

    def assert_balanced(self):
        for idx, edge in enumerate(self.edges):
            assert edge.sent == edge.recv, f"edge {idx - 1}->{idx}: send {edge.sent} != recv {edge.recv}"
            assert edge.queue.empty(), f"edge {idx - 1}->{idx}: {edge.queue.qsize()} message(s) left unread"


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("num_segments", list(range(1, 13)))
def test_clean_request_is_balanced(world, num_segments):
    rig = _Rig(world, num_segments)
    assert rig.run() == []
    rig.assert_balanced()
    assert rig.errors == []


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("num_segments", list(range(1, 13)))
def test_cancel_by_any_rank_at_any_segment_is_balanced(world, num_segments):
    """关键回归:某个 rank 拥有 ≥2 段(4 卡下 n≥5)时,退出的 rank 也必须把入边收干净。"""
    for cancel_rank in range(world):
        for idx in range(num_segments):
            if idx % world != cancel_rank:
                continue
            for phase in ("before", "after"):
                rig = _Rig(world, num_segments)
                stuck = rig.run(cancel_rank=cancel_rank, cancel_phase=(phase, idx))
                assert stuck == [], f"deadlock: w={world} n={num_segments} rank={cancel_rank} {phase} {idx} stuck={stuck}"
                rig.assert_balanced()
                assert rig.errors == [], f"w={world} n={num_segments} rank={cancel_rank} {phase} {idx}: {rig.errors}"


@pytest.mark.parametrize("world", [2, 3, 4])
@pytest.mark.parametrize("num_segments", [5, 8, 12])
def test_exception_path_is_balanced(world, num_segments):
    for raise_rank in range(world):
        idx = raise_rank  # 它拥有的第一段
        rig = _Rig(world, num_segments)
        stuck = rig.run(raise_rank=raise_rank, cancel_phase=("before", idx))
        assert stuck == []
        rig.assert_balanced()


@pytest.mark.parametrize("world", [2, 4])
def test_next_request_starts_with_empty_channels(world):
    """取消之后紧接着的干净请求只能收到自己的消息。"""
    rig = _Rig(world, 8)
    rig.run(cancel_rank=1, cancel_phase=("before", 1))
    rig.assert_balanced()

    rig_second = _Rig(world, 8)
    assert rig_second.run() == []
    rig_second.assert_balanced()
