#!/usr/bin/env python3
"""
gpustack-lx2v-launcher — front proxy + profile-driven engine spawner.

Purpose (see docs/lightx2v-backend-design.md §8.1 / §10 and
docs/lightx2v-builtin-backend-plan.md M1):

GPUStack's LightX2V built-in backend (worker/backends/lightx2v.py) starts an
instance with:

    gpustack-lx2v-launcher --model <model_path> --port <PORT> --host <HOST>

This launcher, running INSIDE the engine image, then:

  1. Counts the GPUs the container was given (CUDA/NVIDIA_VISIBLE_DEVICES).
  2. Picks a profile from profiles.yaml keyed by (model_cls, gpu_count) — the
     profile fixes model_cls / task / config_json / parallelism. Profiles live
     in the image, so calibrating a profile = rebuild the engine image, never
     GPUStack.
  3. Validates parallel_product == gpu_count and spawns the engine on an
     INTERNAL port:
        N == 1 : python -m lightx2v.server ...
        N  > 1 : torchrun --nproc_per_node=N -m lightx2v.server ...   (rank0 = HTTP)
  4. Binds {{PORT}} itself as a thin reverse proxy that:
        - answers  GET /ready  -> 503 until the engine is up (and warmup done),
                                   200 afterwards  (this is what GPUStack's
                                   health_check_path=/ready polls, so scheduling
                                   never routes to a still-loading instance);
        - forwards GET /metrics -> the engine's Prometheus server (on its own
                                   ephemeral port), so GPUStack/Grafana can
                                   scrape worker_ip:{{port}}/metrics without
                                   knowing the internal port;
        - forwards everything else to the internal engine.

Stdlib only (argparse/http.server/urllib/subprocess/threading) plus PyYAML,
which the LightX2V image already ships. No extra runtime deps.
"""

import argparse
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

try:
    import yaml
except Exception:  # pragma: no cover - image is expected to ship PyYAML
    yaml = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [lx2v-launcher] %(levelname)s %(message)s")
logger = logging.getLogger("gpustack-lx2v-launcher")

# Internal endpoint the launcher polls to decide the engine is up. LightX2V
# registers a liveness route at /health on the app root (server.py) that returns
# 200 once the rank-0 FastAPI app is serving — and per main.py that happens after
# the model has finished loading (start_distributed_inference runs before
# uvicorn.run). NOTE: the service-status route is mounted at /v1/service/status
# (router.py prefix="/v1/service"), NOT /status — polling /status 404s forever.
# Override via LX2V_ENGINE_UP_PATH only if the engine image changes this.
_ENGINE_UP_PATH = os.environ.get("LX2V_ENGINE_UP_PATH", "/health")
_DEFAULT_PROFILES = os.path.join(os.path.dirname(__file__), "profiles.yaml")


def _count_gpus() -> int:
    """Count assigned GPUs from the container's visible-devices env."""
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        val = os.environ.get(key)
        if val is None:
            continue
        val = val.strip()
        if val == "" or val.lower() == "void":
            return 0
        if val.lower() == "all":
            break  # fall through to nvidia-smi
        return len([x for x in val.split(",") if x.strip() != ""])
    # "all" or unset -> ask nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            text=True,
        )
        return len([ln for ln in out.splitlines() if ln.strip() != ""])
    except Exception:
        return 1


def _infer_model_cls(model_path: str, override: str) -> str:
    if override:
        return override
    name = os.path.basename(os.path.normpath(model_path)).lower()
    # Coarse inference; deploys should pass --model-cls (or set it in the
    # GPUStack backend parameters) when the path is ambiguous.
    if "z_image" in name or "z-image" in name or "zimage" in name:
        return "z_image"
    # Checked BEFORE the "wan" branch: these model dirs usually contain "wan"
    # too (e.g. Wan2.2-VACE-Fun-*, InfiniteTalk rides the Wan2.1 distill DiT)
    # and would otherwise be misclassified as plain wan2.2_moe.
    if "infinitetalk" in name:
        return "infinitetalk"
    if "seedvr" in name:
        return "seedvr2"
    if "vace" in name:
        return "wan2.2_moe_vace"
    if "wan" in name:
        # I2V/FLF2V is the distill cls (Wan2.2-I2V experiment report:
        # model_cls=wan2.2_moe_distill); T2V is the plain MoE cls (§12.2).
        if "i2v" in name or "flf2v" in name:
            return "wan2.2_moe_distill"
        return "wan2.2_moe"
    if "qwen" in name:
        return "qwen_image"
    return name


def _infer_task_hint(model_path: str, override: str) -> str:
    """Coarse task inference from the model dir name, used only to pick between
    same-GPU-count variants of one model_cls (e.g. qwen_image t2i vs i2i).
    Deploys pass --task as a backend parameter when the path is ambiguous."""
    if override:
        return override
    name = os.path.basename(os.path.normpath(model_path)).lower()
    # Single-task model classes first — infer their task before the generic
    # heuristics below (mirrors _infer_model_cls checking these before "wan").
    # A VACE dir named e.g. "wan2.2-vace-video-editing" must NOT fall to the
    # "edit"->i2i case, or _load_profile rejects its only (task=vace) variant.
    if "infinitetalk" in name:
        return "s2v"
    if "seedvr" in name:
        return "sr"
    if "vace" in name:
        return "vace"
    if "edit" in name:
        return "i2i"
    for task in ("flf2v", "i2v", "t2v", "s2v", "i2i", "t2i"):
        if task in name:
            return task
    return ""


def _load_profile(profiles_file: str, model_cls: str, gpu_count: int, task_hint: str = "", profile_name: str = "") -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the launcher profiles file")
    with open(profiles_file, "r") as f:
        profiles = yaml.safe_load(f) or {}
    model_profiles = profiles.get(model_cls)
    if not model_profiles:
        raise RuntimeError(f"No profile for model_cls '{model_cls}' in {profiles_file}. Known: {sorted(profiles.keys())}")
    variants = model_profiles.get("variants", [])
    if profile_name:
        # Explicit variant pin (--profile backend parameter). Needed when
        # neither GPU count nor task disambiguates — e.g. the two InfiniteTalk
        # archives are both (infinitetalk, 4 GPUs, s2v) and differ only in
        # resolution. Still validated against the actual GPU count below.
        named = [v for v in variants if str(v.get("name", "")) == profile_name]
        if len(named) != 1:
            names = [v.get("name", "?") for v in variants]
            raise RuntimeError(
                f"--profile '{profile_name}' does not match exactly one variant of model_cls '{model_cls}' in {profiles_file}. Known variants: {names}"
            )
        variant = named[0]
        if int(variant.get("gpus", 0)) != gpu_count:
            raise RuntimeError(
                f"--profile '{profile_name}' needs {variant.get('gpus', '?')} GPU(s) but the container has {gpu_count}"
            )
        candidates = named
    else:
        candidates = [v for v in variants if int(v.get("gpus", 0)) == gpu_count]
        if not candidates:
            raise RuntimeError(f"No {gpu_count}-GPU variant for model_cls '{model_cls}' in {profiles_file}")
        if len(candidates) > 1:
            # Same GPU count, different tasks (e.g. qwen_image t2i vs i2i):
            # disambiguate by task, never guess.
            matched = [v for v in candidates if task_hint and str(v.get("task", "")) == task_hint]
            if len(matched) != 1:
                names = [f"{v.get('name', '?')} (task={v.get('task', '?')})" for v in candidates]
                raise RuntimeError(
                    f"Ambiguous {gpu_count}-GPU variants for model_cls '{model_cls}': {names}. Pass --task <task> (or --profile <name> when tasks tie, e.g. the two InfiniteTalk resolutions) as a backend parameter to pick one (inferred task hint: '{task_hint or 'none'}')."
                )
            candidates = matched
    variant = candidates[0]
    if not profile_name and task_hint and variant.get("task") and str(variant["task"]) != task_hint:
        raise RuntimeError(
            f"Model path suggests task '{task_hint}' but the only {gpu_count}-GPU "
            f"variant for '{model_cls}' is task '{variant['task']}' "
            f"({variant.get('name', '?')}). Pass --task to override if intended."
        )
    # If the profile documents the parallel mesh, sanity-check it against
    # the GPU count. These fields are ADVISORY — the engine reads the
    # real mesh from config["parallel"] in the config JSON (top-level CLI
    # args do not configure parallelism), so they must mirror it.
    has_parallel = any(k in variant for k in ("cfg_p_size", "seq_p_size", "tensor_p_size"))
    prod = int(variant.get("cfg_p_size", 1)) * int(variant.get("seq_p_size", 1))
    tp = int(variant.get("tensor_p_size", 0))
    expected = tp if tp else prod
    if has_parallel and expected != gpu_count:
        raise RuntimeError(f"Profile '{model_cls}'/{gpu_count}-gpu is inconsistent: parallel product {expected} != gpu_count {gpu_count}")
    return variant


def _free_ports(count: int, exclude=()) -> list:
    """
    Return `count` distinct free ephemeral loopback ports, none in `exclude`.

    Ephemeral (bind :0) rather than derived (public+1) on purpose: LightX2V's
    Prometheus metrics server binds a FIXED port (LIGHTX2V_METRIC_PORT, default
    8001) before uvicorn, so a derived engine port collides with it when the
    public port is 8000, and co-located instances (z_image runs 4 replicas per
    node) would all collide on 8001. Distinct ephemeral ports for both the
    engine and its metrics avoid every case. Sockets are held open together so
    the returned ports are guaranteed distinct, then closed just before use.
    """
    exclude = set(exclude)
    socks = []
    ports = []
    try:
        while len(ports) < count:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
            if port in exclude:
                s.close()
                continue
            socks.append(s)
            ports.append(port)
    finally:
        for s in socks:
            s.close()
    return ports


def _build_engine_command(
    model_path: str,
    model_cls: str,
    internal_port: int,
    metric_port: int,
    master_port: int,
    profile: dict,
    passthrough: list,
) -> list:
    gpus = int(profile.get("gpus", 1))
    engine = [
        sys.executable,
        "-m",
        "lightx2v.server",
        "--model_path",
        model_path,
        "--model_cls",
        model_cls,
        "--host",
        "127.0.0.1",
        "--port",
        str(internal_port),
        # Unique metrics port per instance; the engine defaults it to a FIXED
        # 8001 which would collide with the engine port and across co-located
        # instances.
        "--metric_port",
        str(metric_port),
    ]
    if profile.get("task"):
        engine += ["--task", str(profile["task"])]
    if profile.get("config_json"):
        engine += ["--config_json", str(profile["config_json"])]
    # Extra engine args declared in the profile (dict -> --key value).
    for key, value in (profile.get("engine_args") or {}).items():
        engine += [f"--{key}", str(value)]
    # Anything GPUStack forwarded verbatim (user backend parameters).
    engine += passthrough

    if gpus <= 1:
        return engine

    # Multi-GPU: torchrun spawns N ranks (sets WORLD_SIZE / LOCAL_RANK, which the
    # engine reads). The parallel MESH (seq_p_size / cfg_p_size / attn type) is
    # NOT set here: the engine reads it from config["parallel"] in the config
    # JSON — top-level CLI args become plain config keys that set_parallel_config
    # ignores. So the profile's config_json MUST carry a matching "parallel"
    # block; the launcher only validates the GPU count against it.
    # --master_addr/--master_port make the c10d rendezvous unique per instance:
    # torchrun otherwise defaults to 29500, which collides across co-located
    # multi-GPU instances (host network; e.g. two 4-GPU wan replicas on an 8-GPU
    # node) — the second would fail/hang before /ready ever goes healthy.
    torchrun = [
        "torchrun",
        f"--nproc_per_node={gpus}",
        "--master_addr=127.0.0.1",
        f"--master_port={master_port}",
        "-m",
        "lightx2v.server",
    ]
    return torchrun + engine[3:]  # engine[3:] drops "python -m lightx2v.server"


class _ReadyState:
    def __init__(self):
        self.ready = False
        self.lock = threading.Lock()

    def set_ready(self):
        with self.lock:
            self.ready = True

    def is_ready(self) -> bool:
        with self.lock:
            return self.ready


def _make_handler(internal_base: str, metrics_base: str, state: _ReadyState):  # noqa: C901
    class _ProxyHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quieter logs
            return

        def _handle_ready(self):
            code = 200 if state.is_ready() else 503
            body = b'{"ready": true}' if code == 200 else b'{"ready": false}'
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _proxy(self, base):
            length = int(self.headers.get("Content-Length", 0) or 0)
            payload = self.rfile.read(length) if length else None
            url = base + self.path
            req = urlrequest.Request(url, data=payload, method=self.command)
            incoming_host = self.headers.get("Host")
            for k, v in self.headers.items():
                if k.lower() in ("host", "content-length", "connection"):
                    continue
                req.add_header(k, v)
            # Preserve the caller-facing Host (and add forwarded headers) so
            # upstreams that build absolute URLs from request.base_url — e.g.
            # /v1/images/edits, whose default response_format is "url" — return a
            # URL reachable back through this launcher, not the internal
            # 127.0.0.1:<engine_port> loopback that urllib would otherwise set.
            if incoming_host:
                req.add_header("Host", incoming_host)
                req.add_header("X-Forwarded-Host", incoming_host)
            req.add_header("X-Forwarded-Proto", "http")
            try:
                with urlrequest.urlopen(req, timeout=3600) as resp:
                    self.send_response(resp.status)
                    for k, v in resp.headers.items():
                        if k.lower() in ("transfer-encoding", "connection"):
                            continue
                        self.send_header(k, v)
                    self.end_headers()
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            except HTTPError as e:
                body = e.read()
                self.send_response(e.code)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except URLError as e:
                msg = f'{{"error": "upstream unavailable: {e}"}}'.encode()
                self.send_response(502)
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)

        def _path_only(self):
            return self.path.split("?", 1)[0].rstrip("/")

        def do_GET(self):
            path = self._path_only()
            if path == "/ready":
                return self._handle_ready()
            # Expose the engine's Prometheus metrics (served on a separate,
            # ephemeral port) through the known public port so GPUStack/Grafana
            # can scrape worker_ip:{{port}}/metrics without knowing the internal
            # port.
            if path == "/metrics":
                return self._proxy(metrics_base)
            return self._proxy(internal_base)

        def do_POST(self):
            return self._proxy(internal_base)

        def do_PUT(self):
            return self._proxy(internal_base)

        def do_DELETE(self):
            return self._proxy(internal_base)

    return _ProxyHandler


def _wait_engine_up(internal_base: str, warmup: dict, state: _ReadyState):
    """Poll the engine until it serves, run optional warmup, then flip /ready."""
    up_url = internal_base + _ENGINE_UP_PATH
    while True:
        try:
            with urlrequest.urlopen(up_url, timeout=2) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(2)
    logger.info("engine is serving (%s 200)", _ENGINE_UP_PATH)

    if warmup and warmup.get("endpoint"):
        try:
            import json as _json

            data = _json.dumps(warmup.get("body", {})).encode()
            req = urlrequest.Request(
                internal_base + warmup["endpoint"],
                data=data,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            urlrequest.urlopen(req, timeout=warmup.get("timeout", 600)).read()
            logger.info("warmup request completed")
        except Exception as e:
            logger.warning("warmup request failed (continuing): %s", e)

    state.set_ready()
    logger.info("/ready now returns 200")


def main():
    parser = argparse.ArgumentParser(description="gpustack-lx2v-launcher")
    parser.add_argument("--model", required=True, help="Model path (GPUStack {{model_path}})")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True, help="Public port ({{port}})")
    parser.add_argument("--model-cls", default="", help="Override inferred model_cls")
    parser.add_argument(
        "--task",
        default="",
        help="Override the task hint used to pick between same-GPU-count variants (e.g. qwen_image t2i vs i2i); the engine still receives the selected variant's task",
    )
    parser.add_argument(
        "--profile",
        default="",
        help="Pin a specific variant by its profile `name` (e.g. infinitetalk-720p/int8-4card); required when GPU count + task cannot disambiguate",
    )
    parser.add_argument("--profiles-file", default=_DEFAULT_PROFILES)
    parser.add_argument("--internal-port", type=int, default=0)
    args, passthrough = parser.parse_known_args()

    gpu_count = _count_gpus()
    model_cls = _infer_model_cls(args.model, args.model_cls)
    task_hint = _infer_task_hint(args.model, args.task)
    profile = _load_profile(args.profiles_file, model_cls, gpu_count, task_hint, args.profile)
    # Distinct ephemeral ports, all unique per instance and != public port:
    #  - engine HTTP  (the engine otherwise takes the public --port)
    #  - metrics      (the engine otherwise pins a fixed 8001)
    #  - torchrun master/rendezvous (torchrun otherwise pins 29500)
    # All must differ across the multiple instances co-located on one node
    # (host network). The master port is only used for multi-GPU profiles.
    if args.internal_port:
        internal_port = args.internal_port
        metric_port, master_port = _free_ports(2, exclude={args.port, internal_port})
    else:
        internal_port, metric_port, master_port = _free_ports(3, exclude={args.port})

    logger.info(
        "model_cls=%s gpus=%s profile=%s internal_port=%s metric_port=%s master_port=%s public=%s:%s",
        model_cls,
        gpu_count,
        profile.get("name", "?"),
        internal_port,
        metric_port,
        master_port,
        args.host,
        args.port,
    )

    engine_cmd = _build_engine_command(
        args.model,
        model_cls,
        internal_port,
        metric_port,
        master_port,
        profile,
        passthrough,
    )
    logger.info("engine command: %s", " ".join(engine_cmd))
    engine = subprocess.Popen(engine_cmd)

    internal_base = f"http://127.0.0.1:{internal_port}"
    metrics_base = f"http://127.0.0.1:{metric_port}"
    state = _ReadyState()
    threading.Thread(
        target=_wait_engine_up,
        args=(internal_base, profile.get("warmup") or {}, state),
        daemon=True,
    ).start()

    httpd = ThreadingHTTPServer((args.host, args.port), _make_handler(internal_base, metrics_base, state))

    def _shutdown(signum, frame):
        logger.info("received signal %s, shutting down", signum)
        try:
            engine.terminate()
        except Exception:
            pass
        # Do NOT call httpd.shutdown() here: this handler runs on the main thread
        # where serve_forever() is blocked, and shutdown() waits for it to exit —
        # calling it from the same thread deadlocks (stops/restarts would hang
        # until a force kill). Exit directly; the engine already got SIGTERM
        # above. Mirrors the engine's own os._exit signal pattern (main.py).
        os._exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # If the engine dies, take the proxy down so the instance is marked failed.
    def _watch_engine():
        rc = engine.wait()
        logger.error("engine exited with code %s; stopping launcher", rc)
        httpd.shutdown()
        os._exit(rc if rc else 1)

    threading.Thread(target=_watch_engine, daemon=True).start()

    logger.info("proxy listening on %s:%s (/ready gated)", args.host, args.port)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
