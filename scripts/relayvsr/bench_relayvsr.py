"""Steady-state latency and memory for RelayVSR in resident mode.

Loads the model once, warms up once, then submits the same request ``--repeat`` times and
reports every run so the first (compile-heavy) request can be dropped from the average.

    torchrun --nproc_per_node=4 scripts/relayvsr/bench_relayvsr.py --repeat 3 \
        --model_cls relayvsr_x2 --task sr --model_path ... --config_json ... \
        --video_path ... --sr_ratio 2 --seed 42 --save_result_path /tmp/out.mp4
"""

import argparse
import os
import sys
import threading
import time

import torch
import torch.distributed as dist

from lightx2v.infer import build_parser
from lightx2v.models.runners.runner_factory import build_runner
from lightx2v.utils.set_config import build_cli_inputs, init_parallel
from lightx2v.utils.utils import validate_config_paths
from lightx2v_platform.registry_factory import PLATFORM_DEVICE_REGISTER


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--stop-after", type=float, default=None, help="raise stop_signal after N seconds of the first request")
    mine, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    args = build_parser().parse_args()
    startup_config, request_data = build_cli_inputs(args)
    if startup_config["parallel"]:
        PLATFORM_DEVICE_REGISTER.get(os.getenv("PLATFORM", "cuda"), None).init_parallel_env()
        init_parallel(startup_config)
    validate_config_paths(startup_config)

    rank = dist.get_rank() if dist.is_initialized() else 0
    runner = build_runner(startup_config)
    input_info = runner.prepare_request(request_data)
    runner.warmup()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    timings = []
    for attempt in range(mine.repeat):
        if mine.stop_after is not None and attempt == 0:
            threading.Timer(mine.stop_after, setattr, (runner, "stop_signal", True)).start()
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            runner.run_request(input_info)
        except Exception as error:  # noqa: BLE001 - the cancellation test wants the type name
            if rank == 0:
                print(f"[bench] run {attempt + 1}: cancelled after {time.perf_counter() - started:.2f}s with {type(error).__name__}", flush=True)
            if mine.stop_after is None:
                raise
            runner.stop_signal = False
            timings.append(float("nan"))
            continue
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - started)
        if rank == 0:
            print(f"[bench] run {attempt + 1}: {timings[-1]:.2f}s", flush=True)

    peak = torch.cuda.max_memory_allocated() / 2**30
    if rank == 0:
        steady = [value for value in (timings[1:] or timings) if value == value]
        if steady:
            print(f"[bench] runs={['%.2f' % value for value in timings]} steady={sum(steady) / len(steady):.2f}s peak_alloc={peak:.2f}GiB", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


main()
