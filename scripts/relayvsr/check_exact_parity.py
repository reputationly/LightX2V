"""A2 parity check: exact block parallelism against the serial relay chain, bitwise, in one process.

Runs the serial reference/flash chain, then the layer-major driver with virtual rank layouts
(``LocalKVExchange``), and asserts every reference latent and every emitted frame is
``torch.equal``.  Use ``--use_compile false`` for the strictly bitwise comparison; with
compilation on the two paths fuse differently and the comparison is informational.

    python scripts/relayvsr/check_exact_parity.py --frames 96 --worlds 1,2,3,4 \
        --model_cls relayvsr_x2 --task sr --model_path ... --config_json ... \
        --video_path ... --sr_ratio 2 --seed 42
"""

import argparse
import hashlib
import sys

import av
import torch

from lightx2v.infer import build_parser
from lightx2v.models.networks.relayvsr.exact_chunk import LocalKVExchange, build_plan, kv_transfer_plan, split_ranks
from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.envs import GET_DTYPE
from lightx2v.utils.set_config import build_cli_inputs, init_parallel
from lightx2v.utils.utils import validate_config_paths


def digest(tensor):
    """Stable per-frame fingerprint; keeps long comparisons off the GPU and out of host RAM."""
    tensor = tensor.detach().cpu().contiguous()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.view(torch.int16)
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def read_frames(path, limit):
    frames = []
    with av.open(str(path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for frame in container.decode(video=0):
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(2, 0, 1).contiguous())
            if len(frames) == limit:
                break
    return frames


def serial_chain(runner, frames, seed):
    """Serial reference chain: per-reference latents plus every emitted frame, before postprocess."""
    recorded = []
    original = runner.reference

    def reference(frame):
        condition = original(frame)
        recorded.append(runner.scheduler.latents[:, :, 0].to(GET_DTYPE()).clone())
        return condition

    runner.reference = reference
    try:
        outputs = [digest(output) for output in runner.restore_frames(iter(frames), seed)]
    finally:
        runner.reference = original
    torch.cuda.empty_cache()
    return recorded, outputs


def exact_chain(runner, frames, seed, plan, world, context):
    """Layer-major reference chain plus window decoding, laid out as ``world`` virtual ranks."""
    spans = split_ranks(plan, world)
    padded = runner.geometry.padded_size(*frames[0].shape[-2:])
    height, width = frames[0].shape[-2:]

    print(f"[parity]   exact world={world}: allocated before phase A {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    runner.init_run(*padded, seed, with_wan_cache=False)
    batches = []
    for span in spans:
        runner.scheduler.prepare(seed, runner.device)
        noises = []
        for ref in range(span.refs.stop):
            runner.scheduler.prepare_latents(*padded, runner.device)
            if ref >= span.refs.start:
                noises.append(runner.scheduler.latents.clone())
        batches.append((span.rank, span.refs.start, noises, [runner.preprocess(frames[plan.ref_frame(ref)]) for ref in span.refs]))
    exchange = LocalKVExchange(kv_transfer_plan(spans, context))
    latents = [None] * plan.refs
    for (rank, first_ref, _, _), produced in zip(batches, runner.model.infer_references(batches, exchange, context)):
        for offset, latent in enumerate(produced):
            latents[first_ref + offset] = latent
    print(f"[parity]   exact world={world}: allocated after phase A {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    runner.end_run()

    runner.init_run(*padded, seed, with_wan_cache=False)
    conditions = {ref: runner.model.flash_model_prepare_condition(latent) for ref, latent in enumerate(latents)}
    outputs = []
    for span in spans:
        for window in span.windows:
            window_frames = [frames[index] for index in plan.window_frames(window)]
            start_condition, end_condition = conditions[window], conditions[min(window + 1, plan.refs - 1)]
            pair = [(torch.cat((a[0], b[0]), 0), torch.cat((a[1], b[1]), 0)) for a, b in zip(start_condition, end_condition)]
            runner.flash_model_history.reset()
            for offset, frame in enumerate(window_frames):
                output = runner.model.flash_model_decode(runner.preprocess(frame), pair, runner.flash_model_history, offset)
                if window == 0 or offset > 0:
                    outputs.append(digest(output[..., : height * runner.geometry.scale, : width * runner.geometry.scale]))
    runner.end_run()
    torch.cuda.empty_cache()
    return latents, outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=96)
    parser.add_argument("--worlds", type=str, default="1,2,3,4")
    mine, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = build_parser().parse_args()
    startup_config, request_data = build_cli_inputs(args)
    if startup_config["parallel"]:
        init_parallel(startup_config)
    validate_config_paths(startup_config)

    runner = build_runner(startup_config)
    input_info = runner.prepare_request(request_data)
    frames = read_frames(input_info.video_path, mine.frames)
    plan = build_plan(len(frames), int(runner.config.get("clip_len", 16)))
    context = int(runner.model.wan_config["temporal_context_frames"])
    seed = int(input_info.seed)
    print(f"[parity] frames={plan.total} clip_len={plan.clip_len} windows={plan.windows} refs={plan.refs} context={context} seed={seed}", flush=True)

    reference_latents, serial_outputs = serial_chain(runner, frames, seed)
    print(f"[parity] serial allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)
    print(f"[parity] serial: {len(reference_latents)} refs, {len(serial_outputs)} frames", flush=True)

    failures = 0
    for world in [int(value) for value in mine.worlds.split(",")]:
        latents, outputs = exact_chain(runner, frames, seed, plan, world, context)
        latent_mismatch = [ref for ref in range(plan.refs) if not torch.equal(latents[ref], reference_latents[ref])]
        count = min(len(outputs), len(serial_outputs))
        frame_mismatch = [index for index in range(count) if outputs[index] != serial_outputs[index]]
        same_frames = len(outputs) == len(serial_outputs)
        status = "PASS" if not latent_mismatch and not frame_mismatch and same_frames else "FAIL"
        print(
            f"[parity] world={world}: {status} latents ok={plan.refs - len(latent_mismatch)}/{plan.refs} "
            f"frames ok={count - len(frame_mismatch)}/{count} same_count={same_frames} "
            f"first_latent_mismatch={latent_mismatch[:3]} first_frame_mismatch={frame_mismatch[:3]}",
            flush=True,
        )
        failures += status != "PASS"

    serial_outputs = reference_latents = frames = None
    sys.exit(1 if failures else 0)


main()
