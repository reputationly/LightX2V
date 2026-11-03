"""Parity tests for the exact chunk plan against a pure-Python replica of the serial relay chain."""

from itertools import chain

import pytest

from lightx2v.models.networks.relayvsr.exact_chunk import build_plan, kv_transfer_plan, split_ranks

TOTALS = (1, 2, 15, 16, 17, 30, 31, 32, 46, 96, 97, 360, 374, 720, 748)
CLIP_LENGTHS = (16, 32)
RANKS = (1, 2, 3, 4, 8)
CONTEXTS = (1, 2, 3, 4, 8)


def serial_stream(total: int, clip_len: int) -> tuple[list[list[int]], list[int]]:
    """Mirror ``RelayVSRRunner.restore_frames``: emitted frames per window and reference frames."""
    emitted: list[list[int]] = []
    refs = [0]
    start = 0
    first_window = True
    while True:
        window = list(range(start, min(start + clip_len, total)))
        if len(window) == 1 and not first_window:
            return emitted, refs
        if len(window) > 1:
            refs.append(window[-1])
        emitted.append(window if first_window else window[1:])
        first_window = False
        start = window[-1]
        if len(window) < clip_len:
            return emitted, refs


@pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
@pytest.mark.parametrize("total", TOTALS)
def test_plan_matches_serial_stream(total: int, clip_len: int) -> None:
    plan = build_plan(total, clip_len)
    emitted, refs = serial_stream(total, clip_len)

    assert plan.windows == len(emitted)
    assert plan.refs == len(refs)
    assert [plan.ref_frame(j) for j in range(plan.refs)] == refs
    for window, frames in enumerate(emitted):
        assert list(plan.emitted_frames(window)) == frames
        assert set(frames) <= set(plan.window_frames(window))

    # Every frame is produced exactly once, in order.
    assert list(chain.from_iterable(emitted)) == list(range(total))


@pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
@pytest.mark.parametrize("total", TOTALS)
@pytest.mark.parametrize("world", RANKS)
def test_split_ranks_partitions_the_video(total: int, clip_len: int, world: int) -> None:
    plan = build_plan(total, clip_len)
    spans = split_ranks(plan, world)

    emitted = list(chain.from_iterable(span.emit for span in spans))
    assert emitted == list(range(total)), "rank output ranges must tile the video in order"

    for span in spans:
        assert set(span.decode) <= set(range(total))
        if span.active:
            window_frames = set(chain.from_iterable(plan.window_frames(w) for w in span.windows))
            ref_frames = {plan.ref_frame(j) for j in span.refs}
            assert window_frames | ref_frames <= set(span.decode)

    refs = list(chain.from_iterable(span.refs for span in spans))
    assert sorted(refs) == list(range(plan.refs)), "every reference is owned exactly once"


@pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
@pytest.mark.parametrize("total", TOTALS)
@pytest.mark.parametrize("world", RANKS)
@pytest.mark.parametrize("context", CONTEXTS)
def test_kv_transfer_plan_matches_serial_visibility(total: int, clip_len: int, world: int, context: int) -> None:
    plan = build_plan(total, clip_len)
    spans = split_ranks(plan, world)
    transfers = kv_transfer_plan(spans, context)
    owner = {ref: span.rank for span in spans for ref in span.refs}

    for span in spans:
        received = sorted(ref for transfer in transfers if transfer.dst == span.rank for ref in transfer.refs)
        expected = list(range(max(0, span.refs.start - (context - 1)), span.refs.start)) if span.active else []
        assert received == expected, f"rank {span.rank} must see the {context - 1} references before its own"

    for transfer in transfers:
        assert transfer.src != transfer.dst
        assert all(owner[ref] == transfer.src for ref in transfer.refs), "a rank only sends its own K/V"
        assert owner[transfer.refs.start] == transfer.src

    active = {span.rank for span in spans if span.active}
    assert all(transfer.src in active and transfer.dst in active for transfer in transfers)

    # One message per boundary: a rank never has two separate runs from the same source.
    pairs = [(transfer.src, transfer.dst) for transfer in transfers]
    assert len(pairs) == len(set(pairs))


@pytest.mark.parametrize("clip_len", CLIP_LENGTHS)
def test_short_video_leaves_idle_ranks_out(clip_len: int) -> None:
    plan = build_plan(30, clip_len)
    spans = split_ranks(plan, 8)
    assert plan.windows < 8
    assert [span.active for span in spans].count(False) == 8 - plan.windows
    transfers = kv_transfer_plan(spans, 4)
    assert all(not span.active for span in spans if span.rank >= plan.windows)
    assert all(transfer.dst < plan.windows for transfer in transfers)


def test_plan_rejects_degenerate_input() -> None:
    with pytest.raises(ValueError):
        build_plan(0, 16)
    with pytest.raises(ValueError):
        build_plan(10, 1)
    with pytest.raises(ValueError):
        split_ranks(build_plan(10, 16), 0)
    with pytest.raises(ValueError):
        kv_transfer_plan(split_ranks(build_plan(10, 16), 2), 0)
