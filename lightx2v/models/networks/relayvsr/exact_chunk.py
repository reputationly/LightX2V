"""Exact block parallelism for RelayVSR: window plan, rank spans, layer-major KV exchange.

The serial relay chain looks reference-major, but its only cross-reference state is each
layer's K/V ring: a reference's activations never feed another reference, and a
reference's layer-ℓ K/V is a pure function of its own layer-ℓ input.  So the chain can be
run layer-major — every rank stages its own references, exchanges the ``context - 1``
prefix references' K/V with its predecessor, then finishes the layer — and the result is
mathematically identical to the serial run (no warm-up, no seam, no approximation).

``RelayPlan`` / ``split_ranks`` mirror ``RelayVSRRunner.restore_frames`` exactly;
``kv_transfer_plan`` is a pure function of the spans, so every rank derives the same
schedule without extra communication.
"""

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class RelayPlan:
    """Window grid of the serial relay chain for one video."""

    total: int
    clip_len: int
    step: int
    windows: int
    refs: int

    def window_frames(self, window: int) -> range:
        start = window * self.step
        return range(start, start + min(self.clip_len, self.total - start))

    def emitted_frames(self, window: int) -> range:
        frames = self.window_frames(window)
        # Window 0 emits its first frame; later windows share it with the window before.
        return frames if window == 0 else range(frames.start + 1, frames.stop)

    def ref_frame(self, ref: int) -> int:
        return min(ref * self.step, self.total - 1)

    def decode_frames(self, first: int, last: int) -> range:
        """Frames a rank owning windows ``[first, last)`` has to decode."""
        if last <= first:
            return range(0)
        return range(first * self.step, min((last - 1) * self.step + self.clip_len, self.total))

    def emit_range(self, first: int, last: int) -> range:
        """Frames a rank owning windows ``[first, last)`` has to write."""
        if last <= first:
            return range(0)
        start = 0 if first == 0 else first * self.step + 1
        return range(start, min((last - 1) * self.step + self.clip_len, self.total))


def build_plan(total: int, clip_len: int) -> RelayPlan:
    if total <= 0:
        raise ValueError(f"RelayVSR needs at least one frame, got total={total}")
    if clip_len < 2:
        raise ValueError(f"RelayVSR clip_len must be >= 2, got {clip_len}")
    step = clip_len - 1
    windows = 1 if total <= clip_len else 1 + -(-(total - clip_len) // step)
    return RelayPlan(total=total, clip_len=clip_len, step=step, windows=windows, refs=1 if total == 1 else windows + 1)


@dataclass(frozen=True)
class RankSpan:
    rank: int
    windows: range
    refs: range
    decode: range
    emit: range

    @property
    def active(self) -> bool:
        return len(self.windows) > 0


def split_ranks(plan: RelayPlan, world: int) -> list[RankSpan]:
    """Contiguous window split; the remainder goes to the earlier ranks.

    Frames are the cheap part here (measured: 1628 fps demux-only), while the last
    active rank also owns reference ``W`` and has to decode the whole prefix, so the
    extra windows are handed out at the front.
    """
    if world < 1:
        raise ValueError(f"RelayVSR needs at least one rank, got world={world}")
    base, extra = divmod(plan.windows, world)
    bounds = []
    cursor = 0
    for rank in range(world):
        count = base + (1 if rank < extra else 0)
        bounds.append((rank, cursor, cursor + count))
        cursor += count
    last_active = max((rank for rank, first, last in bounds if last > first), default=None)
    spans = []
    for rank, first, last in bounds:
        if last <= first:
            spans.append(RankSpan(rank=rank, windows=range(first, last), refs=range(first, first), decode=range(0), emit=range(0)))
            continue
        ref_stop = plan.refs if rank == last_active else last
        spans.append(
            RankSpan(
                rank=rank,
                windows=range(first, last),
                refs=range(first, ref_stop),
                decode=plan.decode_frames(first, last),
                emit=plan.emit_range(first, last),
            )
        )
    return spans


@dataclass(frozen=True)
class KVTransfer:
    src: int
    dst: int
    refs: range


def kv_transfer_plan(spans: list[RankSpan], context: int) -> list[KVTransfer]:
    """Per-layer K/V movement: each rank receives the ``context - 1`` references before its own.

    A reference is only ever sent by the rank that owns it, so every transfer is a single
    point-to-point message and no rank forwards somebody else's K/V.
    """
    if context < 1:
        raise ValueError(f"RelayVSR temporal_context_frames must be >= 1, got {context}")
    owner = {ref: span.rank for span in spans for ref in span.refs}
    transfers = []
    for span in spans:
        if not span.active:
            continue
        prefix = range(max(0, span.refs.start - (context - 1)), span.refs.start)
        run_start = previous = None
        for ref in prefix:
            src = owner[ref]
            if src != previous:
                if previous is not None:
                    transfers.append(KVTransfer(src=previous, dst=span.rank, refs=range(run_start, ref)))
                run_start, previous = ref, src
        if previous is not None:
            transfers.append(KVTransfer(src=previous, dst=span.rank, refs=range(run_start, prefix.stop)))
    return transfers


class KVExchange(Protocol):
    """Layer-scoped K/V exchange; ``rank`` is the logical rank owning ``own``."""

    def post(self, layer: int, rank: int, own: dict[int, tuple[torch.Tensor, torch.Tensor]]) -> None: ...

    def collect(self, layer: int, rank: int) -> dict[int, tuple[torch.Tensor, torch.Tensor]]: ...


class LocalKVExchange:
    """Single-process stand-in for the NCCL exchange (parity tests, one rank per process absent)."""

    def __init__(self, transfers: list[KVTransfer]):
        self.store: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self.requests: dict[int, list[int]] = {}
        for transfer in transfers:
            self.requests.setdefault(transfer.dst, []).extend(transfer.refs)

    def post(self, layer, rank, own):
        for ref, kv in own.items():
            self.store[(layer, ref)] = kv

    def collect(self, layer, rank):
        return {ref: self.store[(layer, ref)] for ref in self.requests.get(rank, ())}


class NcclKVExchange:
    """Point-to-point K/V exchange over the chunk group (one message per rank boundary per layer)."""

    def __init__(self, transfers: list[KVTransfer], rank: int, group):
        self.rank, self.group = rank, group
        self.outgoing = [(transfer.dst, transfer.refs) for transfer in transfers if transfer.src == rank]
        self.incoming = [(transfer.src, transfer.refs) for transfer in transfers if transfer.dst == rank]
        self.handles: list = []
        self.keepalive: list[torch.Tensor] = []
        self.received: list[tuple[range, torch.Tensor]] = []

    def post(self, layer, rank, own):
        if rank != self.rank:
            raise ValueError(f"NcclKVExchange is bound to rank {self.rank}, got {rank}")
        self.handles, self.keepalive, self.received = [], [], []
        if not self.outgoing and not self.incoming:
            return
        sample = own[self.incoming[0][1].start if self.incoming else self.outgoing[0][1].start][0]
        self.tokens, dtype, device = sample.shape[0], sample.dtype, sample.device
        requests = []
        for dst, refs in self.outgoing:
            keys = torch.cat([own[ref][0] for ref in refs], 0)
            values = torch.cat([own[ref][1] for ref in refs], 0)
            payload = torch.cat((keys, values), 0)
            self.keepalive.append(payload)
            requests.append(dist.P2POp(dist.isend, payload, dst, group=self.group))
        for src, refs in self.incoming:
            buffer = torch.empty((2 * len(refs) * self.tokens, sample.shape[1], sample.shape[2]), dtype=dtype, device=device)
            self.keepalive.append(buffer)
            self.received.append((refs, buffer))
            requests.append(dist.P2POp(dist.irecv, buffer, src, group=self.group))
        self.handles = dist.batch_isend_irecv(requests)

    def collect(self, layer, rank):
        if not self.received:
            return {}
        for handle in self.handles:
            handle.wait()
        collected = {}
        for refs, buffer in self.received:
            split = len(refs) * self.tokens
            keys, values = buffer[:split], buffer[split:]
            for index, ref in enumerate(refs):
                collected[ref] = (keys[index * self.tokens : (index + 1) * self.tokens], values[index * self.tokens : (index + 1) * self.tokens])
        self.handles, self.keepalive, self.received = [], [], []
        return collected
