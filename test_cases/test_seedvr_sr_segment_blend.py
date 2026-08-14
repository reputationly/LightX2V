import datetime
import inspect
import os
import tempfile
import time
import unittest
from importlib import import_module
from types import SimpleNamespace

import torch

os.environ.setdefault("SKIP_PLATFORM_CHECK", "1")


def _make_runner(config):
    """A SeedVRRunner with only ``config`` populated.

    The segmentation and blend helpers touch nothing else on the runner, so
    skipping __init__ keeps the test free of model weights and CUDA.
    """
    seedvr_runner = import_module("lightx2v.models.runners.seedvr.seedvr_runner")
    runner = seedvr_runner.SeedVRRunner.__new__(seedvr_runner.SeedVRRunner)
    runner.config = config
    return runner


def _greedy_segment_count(total_frames, seg_len, overlap):
    """Segment count of the pre-balancing greedy walk, as a reference."""
    count, start = 0, 0
    while start < total_frames:
        count += 1
        end = min(start + seg_len, total_frames)
        if end >= total_frames:
            break
        start = end - overlap
    return count


def _emitted_frames(segments, total_frames):
    """Frames each segment contributes to the output under hold-back-and-blend.

    Mirrors run_pipeline: a non-final segment holds back the frames it shares
    with its successor, and the successor emits them after cross-fading.
    """
    emitted = []
    for idx, (start, _end) in enumerate(segments):
        stop = segments[idx + 1][0] if idx + 1 < len(segments) else total_frames
        emitted.append(stop - start)
    return emitted


class SeedVRSegmentationTest(unittest.TestCase):
    GRIDS = [124, 243, 362]  # MiniMax H3's 17n+5 frame grid at 24fps

    def _segments(self, total_frames, seg_len=121, overlap=8):
        runner = _make_runner({"sr_segment_length": seg_len, "sr_overlap": overlap})
        return runner._build_sr_segments(total_frames, seg_len, overlap)

    def test_defaults_cross_fade_rather_than_drop_one_frame(self):
        runner = _make_runner({})
        seg_len, overlap = runner._get_sr_segment_params()
        self.assertEqual(seg_len, 81)
        self.assertGreaterEqual(overlap, 2, "default overlap must be wide enough to ramp across")

    def test_overlap_wider_than_segment_is_clamped(self):
        runner = _make_runner({"sr_segment_length": 8, "sr_overlap": 32})
        seg_len, overlap = runner._get_sr_segment_params()
        self.assertEqual((seg_len, overlap), (8, 7))

    def test_disabled_when_segment_length_non_positive(self):
        runner = _make_runner({"sr_segment_length": 0})
        self.assertEqual(runner._get_sr_segment_params(), (None, 0))

    def test_short_clip_is_a_single_segment(self):
        self.assertEqual(self._segments(121), [(0, 121)])
        self.assertEqual(self._segments(0), [(0, 0)])

    def test_segments_respect_the_memory_ceiling(self):
        for total in self.GRIDS + [122, 200, 500, 1000]:
            for seg_len, overlap in [(121, 8), (81, 8), (81, 1), (41, 8)]:
                for start, end in self._segments(total, seg_len, overlap):
                    self.assertLessEqual(end - start, seg_len, f"total={total} seg_len={seg_len}")

    def test_no_runt_segment_on_h3_frame_grids(self):
        # 124 frames greedily split 121+3; a 3-frame tail cannot be cross-faded
        # and gets none of the temporal context its neighbour had.
        for total in self.GRIDS:
            segments = self._segments(total)
            lengths = [end - start for start, end in segments]
            self.assertGreater(min(lengths), 8, f"total={total} lengths={lengths}")
            self.assertLessEqual(max(lengths) - min(lengths), 8, f"total={total} lengths={lengths}")

    def test_balancing_does_not_add_a_diffusion_pass(self):
        for total in self.GRIDS + [122, 200, 500, 1000]:
            for seg_len, overlap in [(121, 8), (81, 8), (41, 8)]:
                self.assertEqual(
                    len(self._segments(total, seg_len, overlap)),
                    _greedy_segment_count(total, seg_len, overlap),
                    f"total={total} seg_len={seg_len} overlap={overlap}",
                )

    def test_segments_cover_every_frame_with_a_usable_boundary(self):
        for total in self.GRIDS + [122, 200, 500, 1000]:
            for seg_len, overlap in [(121, 8), (81, 8), (81, 1), (41, 8)]:
                segments = self._segments(total, seg_len, overlap)
                self.assertEqual(segments[0][0], 0)
                self.assertEqual(segments[-1][1], total)
                for idx in range(len(segments) - 1):
                    boundary = segments[idx][1] - segments[idx + 1][0]
                    seg_frames = segments[idx][1] - segments[idx][0]
                    # Wide enough to ramp across, and never so wide that the
                    # hold-back would consume the whole segment.
                    self.assertGreater(boundary, 0)
                    self.assertLessEqual(boundary, overlap)
                    self.assertLess(boundary, seg_frames)

    def test_every_source_frame_is_emitted_exactly_once(self):
        for total in self.GRIDS + [122, 200, 500, 1000]:
            for seg_len, overlap in [(121, 8), (81, 8), (81, 1), (41, 8)]:
                segments = self._segments(total, seg_len, overlap)
                emitted = _emitted_frames(segments, total)
                self.assertTrue(all(n > 0 for n in emitted), f"total={total} emitted={emitted}")
                self.assertEqual(sum(emitted), total, f"total={total} emitted={emitted}")


class SeedVRBlendTest(unittest.TestCase):
    def _runner(self):
        return _make_runner({"sr_segment_length": 121, "sr_overlap": 8})

    @staticmethod
    def _video(value, frames):
        return torch.full((1, 3, frames, 2, 2), float(value))

    def test_ramp_spans_the_full_window(self):
        runner = self._runner()
        blended = runner._blend_sr_overlap(self._video(0.0, 8), self._video(1.0, 30))
        window = blended[0, 0, :8, 0, 0]
        expected = torch.linspace(0.0, 1.0, 8)
        self.assertTrue(torch.allclose(window, expected))
        # Ends pinned to the neighbouring single-segment frames: no relocated seam.
        self.assertEqual(window[0].item(), 0.0)
        self.assertEqual(window[-1].item(), 1.0)
        self.assertTrue(torch.all(window[1:] > window[:-1]))

    def test_length_and_tail_are_untouched(self):
        runner = self._runner()
        segment = torch.randn(1, 3, 30, 2, 2)
        blended = runner._blend_sr_overlap(torch.randn(1, 3, 8, 2, 2), segment)
        self.assertEqual(blended.shape, segment.shape)
        self.assertTrue(torch.equal(blended[:, :, 8:], segment[:, :, 8:]))

    def test_agreeing_segments_blend_to_themselves(self):
        # When both segments hallucinate the same detail the cross-fade must be
        # a no-op -- otherwise it would soften frames that had no seam.
        runner = self._runner()
        segment = torch.randn(1, 3, 30, 2, 2)
        blended = runner._blend_sr_overlap(segment[:, :, :8].clone(), segment)
        self.assertTrue(torch.allclose(blended, segment, atol=1e-6))

    def test_single_frame_overlap_averages(self):
        runner = self._runner()
        blended = runner._blend_sr_overlap(self._video(0.0, 1), self._video(1.0, 4))
        self.assertAlmostEqual(blended[0, 0, 0, 0, 0].item(), 0.5)

    def test_tail_longer_than_segment_uses_the_adjacent_frames(self):
        runner = self._runner()
        tail = torch.randn(1, 3, 8, 2, 2)
        segment = torch.randn(1, 3, 4, 2, 2)
        blended = runner._blend_sr_overlap(tail, segment)
        self.assertEqual(blended.shape, segment.shape)
        # The window is anchored on the tail's LAST frames, which are the ones
        # adjacent to the segment.
        self.assertTrue(torch.allclose(blended[:, :, 0], tail[:, :, -4]))

    def test_empty_tail_is_a_no_op(self):
        runner = self._runner()
        segment = torch.randn(1, 3, 4, 2, 2)
        blended = runner._blend_sr_overlap(torch.randn(1, 3, 0, 2, 2), segment)
        self.assertTrue(torch.equal(blended, segment))

    def test_weights_are_finite_for_every_window_size(self):
        runner = self._runner()
        for count in range(1, 17):
            weights = runner._sr_blend_weights(count, torch.device("cpu"), torch.float32)
            self.assertEqual(weights.shape, (count,))
            self.assertTrue(torch.all(torch.isfinite(weights)))
            self.assertTrue(torch.all((weights >= 0.0) & (weights <= 1.0)))


class SeedVRSegmentParallelTest(unittest.TestCase):
    """Segment-to-rank assignment. The distributed hand-off itself needs a
    process group, but the ownership rule it rests on is pure arithmetic."""

    def _owner(self, idx, world):
        return _make_runner({})._sr_segment_owner(idx, world)

    def test_every_segment_has_exactly_one_owner(self):
        for world in range(1, 9):
            for num_segments in range(1, 20):
                owners = [self._owner(idx, world) for idx in range(num_segments)]
                self.assertEqual(len(owners), num_segments)
                for owner in owners:
                    self.assertIn(owner, range(world))

    def test_neighbouring_segments_live_on_different_ranks(self):
        # The tail hand-off is a point-to-point send, which would have to be
        # special-cased if a rank could ever be its own predecessor.
        for world in range(2, 9):
            for idx in range(1, 20):
                self.assertNotEqual(self._owner(idx, world), self._owner(idx - 1, world))

    def test_work_is_evenly_spread(self):
        # Segments are near-equal in length after balancing, so an even count
        # per rank is an even wall-clock split; the slowest rank sets the pace.
        for world in (2, 3, 4):
            for num_segments in range(world, 4 * world + 1):
                loads = [sum(1 for idx in range(num_segments) if self._owner(idx, world) == r) for r in range(world)]
                self.assertLessEqual(max(loads) - min(loads), 1)

    def test_segment_noise_depends_on_the_segment_not_the_rank(self):
        # The whole point of the per-segment stream: a segment's noise is a
        # function of (seed, global index), so splitting the same clip across
        # 4 ranks reproduces what one card would have produced.
        latent = torch.zeros(1, 16, 4, 8, 8)

        def noise(index, seed=42):
            runner = _make_runner({})
            runner.input_info = SimpleNamespace(seed=seed)
            runner._sr_segment = (0, 121)
            runner._sr_segment_index = index
            gen = runner._sr_noise_generator(latent.device)
            return runner._sr_randn_like(latent, gen)

        self.assertTrue(torch.equal(noise(2), noise(2)), "same segment must be reproducible")
        for a, b in [(0, 1), (1, 2), (0, 3)]:
            self.assertFalse(torch.equal(noise(a), noise(b)), f"segments {a} and {b} must not share noise")
        self.assertFalse(torch.equal(noise(0, seed=1), noise(0, seed=2)), "the seed must still matter")

    def test_segment_noise_seeds_do_not_collide_across_seeds(self):
        # seed+index would make (seed=1, idx=1) and (seed=2, idx=0) identical.
        runner = _make_runner({})
        runner._sr_segment = (0, 121)
        seeds = set()
        for seed in range(8):
            for index in range(8):
                runner.input_info = SimpleNamespace(seed=seed)
                runner._sr_segment_index = index
                seeds.add(runner._sr_noise_generator(torch.device("cpu")).initial_seed())
        self.assertEqual(len(seeds), 64)

    def test_whole_clip_runs_keep_the_global_rng(self):
        # No segmentation -> no generator, so unsegmented SR is byte-for-byte
        # what it was before the per-segment stream existed.
        runner = _make_runner({})
        runner.input_info = SimpleNamespace(seed=42)
        runner._sr_segment = None
        runner._sr_segment_index = 0
        self.assertIsNone(runner._sr_noise_generator(torch.device("cpu")))

    def test_rank_local_mode_keeps_the_denoise_loop_off_the_world_group(self):
        # The bug this guards: check_stop all-reduces every denoise step, but
        # under segment parallelism only the ranks holding a segment run steps.
        # A 124-frame request (2 segments) on 4 ranks hung there until NCCL's
        # watchdog aborted the server 600s later.
        base_runner = import_module("lightx2v.models.runners.base_runner")
        runner = _make_runner({})
        runner.stop_signal = False
        runner.pause_signal = False
        calls = []
        stub = SimpleNamespace(
            is_initialized=lambda: True,
            get_rank=lambda: 2,
            get_world_size=lambda: 4,
            all_reduce=lambda *a, **k: calls.append(a),
            ReduceOp=SimpleNamespace(MAX=0),
        )
        original = base_runner.dist
        base_runner.dist = stub
        try:
            runner._rank_local_collectives = True
            runner.check_stop()
            self.assertEqual(calls, [], "rank-local mode must not touch the world group")
            runner._rank_local_collectives = False
            runner.check_stop()
            self.assertEqual(len(calls), 1, "the ordinary path must still agree across ranks")
        finally:
            base_runner.dist = original

    def _rank0_fallback_stub(self, meetings):
        """A dist stub whose only collective is the rendezvous all-reduce.

        No ``barrier`` attribute on purpose: reaching for one raises
        AttributeError rather than quietly pairing a barrier against the
        all-reduce a peer arriving from another path would post.
        """

        def all_reduce(tensor, op=None, group=None):
            meetings.append((group, [int(v) for v in tensor]))

        return SimpleNamespace(
            is_available=lambda: True,
            is_initialized=lambda: True,
            get_rank=lambda: 0,
            get_world_size=lambda: 4,
            all_reduce=all_reduce,
            ReduceOp=SimpleNamespace(MAX=0),
        )

    def test_rank0_fallback_also_keeps_the_denoise_loop_rank_local(self):
        # Same hazard as above, on the other path: requests segment parallel
        # cannot take (single segment, tensor output, RIFE) run on rank 0 alone
        # while the peers wait in the rendezvous. A world all-reduce inside the
        # loop has nobody to meet there either.
        seedvr_runner = import_module("lightx2v.models.runners.seedvr.seedvr_runner")
        runner = _make_runner({})
        runner._sr_ctrl_pg = "gloo-pg"
        runner.stop_signal = False
        runner.pause_signal = False
        seen = []
        meetings = []
        original = seedvr_runner.dist
        seedvr_runner.dist = self._rank0_fallback_stub(meetings)
        try:
            result = runner._sr_run_on_rank0(lambda: seen.append(runner._rank_local_collectives) or "done")
        finally:
            seedvr_runner.dist = original
        self.assertEqual(seen, [True], "the pinned request must run with the world collective off")
        self.assertEqual(result, "done")
        self.assertEqual(meetings, [("gloo-pg", [0, 0, 0])], "the peers must still be released, on the one shared op")
        self.assertFalse(runner._rank_local_collectives, "and the flag must not leak past the request")

    def test_the_inner_segment_loop_cannot_undo_the_rank0_fallback(self):
        # The hole the fallback test above could not see. _sr_run_on_rank0 calls
        # _run_sr_segments with seg_world == 1, and that used to *assign*
        # `seg_world > 1` over the flag -- clearing exactly what the fallback
        # had just set. Measured on 4xA100: a RIFE request got as far as
        # `step_index: 1 / 1` and then rank 0 sat in check_stop's all-reduce
        # until the watchdog aborted the process group 600 s later.
        runner = _make_runner({})
        runner._rank_local_collectives = True  # as _sr_run_on_rank0 leaves it
        previous = runner._sr_enter_rank_local(seg_world=1)
        self.assertTrue(previous, "the caller's setting has to come back out")
        self.assertTrue(runner._rank_local_collectives, "the pinned request is still rank-0-only")
        runner._rank_local_collectives = previous
        self.assertTrue(runner._rank_local_collectives, "and restoring must not clear it either")

    def test_the_segment_loop_still_goes_rank_local_on_its_own(self):
        runner = _make_runner({})
        self.assertFalse(runner._sr_enter_rank_local(seg_world=4), "nothing to restore at the top level")
        self.assertTrue(runner._rank_local_collectives, "ranks own different segment counts")
        runner = _make_runner({})
        self.assertFalse(runner._sr_enter_rank_local(seg_world=1))
        self.assertFalse(runner._rank_local_collectives, "a plain single-process run keeps the world all-reduce")

    def test_rank0_fallback_restores_the_flag_when_the_request_raises(self):
        seedvr_runner = import_module("lightx2v.models.runners.seedvr.seedvr_runner")
        runner = _make_runner({})
        runner._sr_ctrl_pg = "gloo-pg"
        runner.stop_signal = False
        runner.pause_signal = False
        meetings = []
        original = seedvr_runner.dist
        seedvr_runner.dist = self._rank0_fallback_stub(meetings)

        def boom():
            raise RuntimeError("segment blew up")

        try:
            with self.assertRaises(RuntimeError):
                runner._sr_run_on_rank0(boom)
        finally:
            seedvr_runner.dist = original
        self.assertEqual(len(meetings), 1, "a failure on rank 0 must not strand the peers")
        # And the peers must not be told the request went fine: the fallback
        # used to meet them with a bare barrier, so they returned None as a
        # success while rank 0 unwound a traceback.
        self.assertEqual(meetings[0][1], [0, 0, 1], "the failure has to reach the peers, not just the meeting")
        self.assertFalse(runner._rank_local_collectives)

    def _rendezvous(self, failed, agreed_signals):
        """Run _sr_seg_rendezvous against a stubbed collective.

        ``agreed_signals`` is what the MAX all-reduce settles on across ranks.
        """
        seedvr_runner = import_module("lightx2v.models.runners.seedvr.seedvr_runner")
        runner = _make_runner({})
        runner.stop_signal = False
        runner.pause_signal = False
        runner.end_run = lambda: None
        runner._sr_ctrl_pg = "gloo-pg"
        calls = []

        def all_reduce(tensor, op=None, group=None):
            calls.append(group)
            tensor.copy_(torch.tensor(agreed_signals, dtype=tensor.dtype))

        stub = SimpleNamespace(get_rank=lambda: 1, all_reduce=all_reduce, ReduceOp=SimpleNamespace(MAX=0))
        original = seedvr_runner.dist
        seedvr_runner.dist = stub
        try:
            runner._sr_seg_rendezvous(failed=failed)
        finally:
            seedvr_runner.dist = original
        return calls

    def test_rendezvous_is_quiet_when_every_rank_finished(self):
        self.assertEqual(self._rendezvous(False, [0, 0, 0]), ["gloo-pg"])

    def test_rendezvous_propagates_a_peers_cancellation(self):
        seedvr_runner = import_module("lightx2v.models.runners.seedvr.seedvr_runner")
        with self.assertRaises(seedvr_runner.TaskStopped):
            self._rendezvous(False, [1, 0, 0])
        with self.assertRaises(seedvr_runner.TaskStopped):
            self._rendezvous(False, [0, 1, 0])

    def test_rendezvous_fails_the_survivors_when_a_peer_died(self):
        # Otherwise rank 0 concatenates a scratch dir with a segment missing.
        with self.assertRaises(RuntimeError):
            self._rendezvous(False, [0, 0, 1])
        # The rank that actually failed re-raises its own traceback instead.
        self.assertEqual(self._rendezvous(True, [0, 0, 1]), ["gloo-pg"])

    def test_parallel_is_off_without_configuration(self):
        # No seg_p_size in the config means the runner must not touch the
        # process group, whether or not one happens to be initialised.
        runner = _make_runner({})
        self.assertEqual(runner._sr_seg_parallel_info(4, True, None), (0, 1))
        runner = _make_runner({"seg_parallel": False})
        self.assertEqual(runner._sr_seg_parallel_info(4, True, None), (0, 1))


class SeedVRCollectiveInvariantTest(unittest.TestCase):
    """Guards on the two invariants this file's bugs all violated.

    Source-level rather than behavioural, and that is the point: both defects
    are about code that is *reachable* on a path no unit test drives -- a rank
    raising in the middle of a distributed request. What can be checked cheaply
    is that the shapes which make those paths unsafe are simply not present.
    """

    def _source(self):
        return inspect.getsource(import_module("lightx2v.models.runners.seedvr.seedvr_runner"))

    def test_the_control_group_only_ever_sees_one_collective(self):
        # Ranks reach the meeting points down four different paths (returned,
        # cancelled, raised, rank-0 fallback). While a plain barrier lived
        # alongside the rendezvous all-reduce, any two paths that disagreed
        # paired a barrier against an all-reduce on the same gloo group and both
        # ranks waited out the two-hour timeout -- which is what a rank-0
        # failure to create the scratch dir did.
        self.assertNotIn("dist.barrier", self._source(), "the rendezvous all-reduce is the only op allowed on the control group")

    def test_the_teardown_does_not_release_an_in_flight_tail(self):
        # A gloo SendWork owns the only reference to its payload, so dropping
        # the handle tears the buffer down under a peer that may still be
        # reading it. The abandoned sends are released at the top of the next
        # request instead, past the rendezvous that proves the peers are out.
        body = inspect.getsource(import_module("lightx2v.models.runners.seedvr.seedvr_runner").SeedVRRunner._run_sr_segments)
        self.assertEqual(body.count("self._sr_tail_sends = []"), 1, "cleared once, at the start of a request -- never in the failure teardown")


class SeedVRTailHandoffTest(unittest.TestCase):
    """The default gloo hand-off must not stall the sender, or use NCCL.

    Group the segments into waves of ``world`` under round-robin ownership.
    Within a wave the two sides diffuse together and the hand-off costs
    inter-rank skew, but the edge crossing a wave is not in lockstep: the
    receiver diffuses its own segment before posting the recv, so the tail sits
    ready for a full segment. A blocking send parked the sender for the whole
    of it and the delay walked backwards one rank per wave.
    """

    TAIL_PG = "tail-pg"

    class _Work:
        def __init__(self, log, tag):
            self.log, self.tag, self.waited = log, tag, False

        def wait(self):
            self.waited = True
            self.log.append(("wait", self.tag))

    def _runner(self, config=None):
        seedvr_runner = import_module("lightx2v.models.runners.seedvr.seedvr_runner")
        runner = _make_runner({} if config is None else config)
        runner._sr_tail_sends = []
        # Pre-seeded so _sr_tail_group short-circuits: new_group is collective
        # and there is no process group in a unit test.
        runner._sr_tail_pg = self.TAIL_PG
        log = []
        works = []

        def isend(tensor, dst, group=None):
            work = self._Work(log, (dst, tuple(tensor.shape)))
            works.append(work)
            log.append(("isend", dst, tuple(tensor.shape), group, str(tensor.device), tensor.dtype))
            return work

        stub = SimpleNamespace(isend=isend)
        return seedvr_runner, runner, stub, log, works

    def _with_stub(self, seedvr_runner, stub, fn):
        original = seedvr_runner.dist
        seedvr_runner.dist = stub
        try:
            return fn()
        finally:
            seedvr_runner.dist = original

    def test_send_does_not_block_on_the_peer(self):
        seedvr_runner, runner, stub, log, works = self._runner()
        tail = torch.zeros(1, 3, 8, 4, 4)
        self._with_stub(seedvr_runner, stub, lambda: runner._sr_send_tail(tail, dst=0, idx=1))
        self.assertEqual([e[0] for e in log], ["isend", "isend"], "meta then payload, neither awaited")
        self.assertFalse(any(w.waited for w in works))
        self.assertEqual(len(runner._sr_tail_sends), 2, "handles and payloads must stay alive")

    def test_the_handoff_stays_off_the_default_process_group(self):
        # The default group is NCCL, whose point-to-point ops have no timeout
        # and cannot be cancelled: a peer that dies mid-segment would park this
        # rank here forever, short of the rendezvous that reports the failure.
        seedvr_runner, runner, stub, log, _works = self._runner()
        self._with_stub(seedvr_runner, stub, lambda: runner._sr_send_tail(torch.zeros(1, 3, 8, 4, 4), dst=0, idx=1))
        self.assertTrue(all(entry[3] == self.TAIL_PG for entry in log), "every send must carry the bounded tail group")
        for entry in log:
            self.assertEqual(entry[4], "cpu", "gloo cannot send CUDA tensors")
        self.assertEqual(log[1][5], torch.float32, "payload is normalised to fp32 for the receiver")

    def test_absent_tail_still_posts_a_meta_so_the_receiver_is_not_stranded(self):
        seedvr_runner, runner, stub, log, _works = self._runner()
        self._with_stub(seedvr_runner, stub, lambda: runner._sr_send_tail(None, dst=2, idx=3))
        self.assertEqual(len(log), 1, "zero-length meta, no payload")
        self.assertEqual(log[0][:3], ("isend", 2, (5,)))

    def test_the_previous_handoff_is_drained_before_the_next_one(self):
        # Bounds the memory held for in-flight tails at one boundary (~200 MiB
        # at 1080p) instead of letting it grow with the segment count.
        seedvr_runner, runner, stub, log, works = self._runner()
        tail = torch.zeros(1, 3, 8, 4, 4)
        self._with_stub(seedvr_runner, stub, lambda: runner._sr_send_tail(tail, dst=1, idx=0))
        first = list(works)
        self._with_stub(seedvr_runner, stub, lambda: runner._sr_send_tail(tail, dst=1, idx=4))
        self.assertTrue(all(w.waited for w in first), "wave N-1's send must be settled")
        self.assertEqual(len(runner._sr_tail_sends), 2, "only wave N's send is still held")
        self.assertEqual([e[0] for e in log], ["isend", "isend", "wait", "wait", "isend", "isend"])

    def test_drain_is_idempotent_and_releases_the_payloads(self):
        seedvr_runner, runner, stub, _log, works = self._runner()
        self._with_stub(seedvr_runner, stub, lambda: runner._sr_send_tail(torch.zeros(1, 3, 8, 4, 4), dst=1, idx=0))
        runner._sr_drain_tail_sends()
        self.assertTrue(all(w.waited for w in works))
        self.assertEqual(runner._sr_tail_sends, [])
        runner._sr_drain_tail_sends()  # nothing left to wait on
        self.assertEqual(runner._sr_tail_sends, [])

    def test_transport_defaults_to_gloo_and_rejects_nonsense(self):
        self.assertEqual(_make_runner({})._sr_tail_transport(), "gloo")
        self.assertEqual(_make_runner({"sr_tail_transport": "FILE"})._sr_tail_transport(), "file")
        self.assertEqual(_make_runner({"sr_tail_transport": "nccl"})._sr_tail_transport(), "gloo", "an unknown transport must not silently disable the hand-off")


class SeedVRTailFileTransportTest(unittest.TestCase):
    """``sr_tail_transport: file`` publishes tails through the scratch dir.

    The sender never waits for anyone, and the receiver's wait is bounded by
    its own deadline rather than by a peer that may never arrive. Nothing here
    may touch ``dist``: a hand-off that needs a live process group is exactly
    what makes a dead rank unrecoverable.
    """

    def _runner(self, tail_dir, timeout_s=None):
        seedvr_runner = import_module("lightx2v.models.runners.seedvr.seedvr_runner")
        runner = _make_runner({"sr_tail_transport": "file"})
        runner._sr_tail_sends = []
        runner._sr_tail_dir = tail_dir
        if timeout_s is not None:
            runner._SR_TAIL_TIMEOUT = datetime.timedelta(seconds=timeout_s)
        return seedvr_runner, runner

    def _with_dist_forbidden(self, seedvr_runner, fn):
        def boom(*_a, **_k):
            raise AssertionError("the file transport must not use the process group")

        original = seedvr_runner.dist
        seedvr_runner.dist = SimpleNamespace(isend=boom, recv=boom, send=boom)
        try:
            return fn()
        finally:
            seedvr_runner.dist = original

    def test_a_tail_survives_the_round_trip_without_the_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            seedvr_runner, runner = self._runner(tmp)
            tail = torch.arange(96, dtype=torch.float32).reshape(1, 3, 2, 4, 4)
            self._with_dist_forbidden(seedvr_runner, lambda: runner._sr_send_tail(tail, dst=1, idx=7))
            self.assertEqual(runner._sr_tail_sends, [], "the sender holds nothing; the write already landed")
            got = self._with_dist_forbidden(seedvr_runner, lambda: runner._sr_recv_tail(src=0, idx=7))
            self.assertTrue(torch.equal(got, tail))

    def test_an_absent_tail_round_trips_as_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            seedvr_runner, runner = self._runner(tmp)
            self._with_dist_forbidden(seedvr_runner, lambda: runner._sr_send_tail(None, dst=1, idx=2))
            self.assertIsNone(self._with_dist_forbidden(seedvr_runner, lambda: runner._sr_recv_tail(src=0, idx=2)))

    def test_a_half_written_tail_is_never_visible_to_the_receiver(self):
        # The receiver polls for the final name, so the payload has to appear
        # under it atomically -- torch.save straight to the target would let a
        # reader in mid-write.
        with tempfile.TemporaryDirectory() as tmp:
            _seedvr_runner, runner = self._runner(tmp)
            runner._sr_send_tail(torch.zeros(1, 3, 2, 4, 4), dst=1, idx=5)
            names = sorted(os.listdir(tmp))
            self.assertEqual(names, ["tail_00005.pt"], "the .part sibling must be renamed away, not left behind")

    def test_the_tail_is_consumed_so_it_cannot_accumulate(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seedvr_runner, runner = self._runner(tmp)
            runner._sr_send_tail(torch.zeros(1, 3, 2, 4, 4), dst=1, idx=0)
            runner._sr_recv_tail(src=1, idx=0)
            self.assertEqual(os.listdir(tmp), [], "a rank owning several segments would otherwise keep every boundary")

    def test_a_tail_that_never_arrives_raises_instead_of_hanging(self):
        # The regression this transport exists for. Before, a rank that raised
        # while diffusing segment idx-1 left the owner of segment idx blocked in
        # an un-timed NCCL recv, so it never reached _sr_seg_rendezvous and the
        # failure it was supposed to report never propagated -- the watchdog
        # took the whole process group down instead.
        with tempfile.TemporaryDirectory() as tmp:
            _seedvr_runner, runner = self._runner(tmp, timeout_s=0.5)
            started = time.monotonic()
            with self.assertRaises(RuntimeError) as caught:
                runner._sr_recv_tail(src=2, idx=3)
            self.assertLess(time.monotonic() - started, 30, "the wait must be bounded by the deadline")
            self.assertIn("died mid-segment", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
