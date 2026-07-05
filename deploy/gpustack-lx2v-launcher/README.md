# gpustack-lx2v-launcher

Front proxy + profile-driven engine spawner, built **into this engine image**.
It is what the GPUStack LightX2V built-in backend execs per instance:

```
gpustack-lx2v-launcher --model <model_path> --port <PORT> --host <HOST> [user params...]
```

It counts the container's GPUs, picks a profile from `profiles.yaml`, spawns
`python -m lightx2v.server` (1 GPU) or `torchrun --nproc_per_node=N -m
lightx2v.server` (N GPUs) on an **internal** port, and binds `{{PORT}}` as a
thin reverse proxy that self-answers `GET /ready` (503 until the engine is up +
warmed, 200 after), routes `GET /metrics` to the engine's Prometheus server, and
forwards everything else. GPUStack's `health_check_path=/ready` polls this, so
scheduling never routes to a still-loading instance. Readiness is derived by
polling the engine's own `/health` liveness route (LightX2V has no `/ready`; its
service status is at `/v1/service/status`, not `/status`).

## Ports

Only the public `{{port}}` is known to GPUStack (it assigns it, health-checks
and proxies to it). The launcher owns it and relays to two **ephemeral** private
ports it picks per instance: the engine's HTTP server and its Prometheus metrics
server (the engine otherwise pins metrics to a fixed 8001, which collides with
the engine port and across the 4 co-located z_image replicas). `/metrics` is
proxied out through `{{port}}` so GPUStack/Grafana can scrape
`worker_ip:{{port}}/metrics` without knowing the internal port.

## Image layout

`dockerfiles/Dockerfile_aarch64_app` does `COPY . /opt/LightX2V`, so at runtime:

- launcher: `/opt/LightX2V/deploy/gpustack-lx2v-launcher/gpustack_lx2v_launcher.py`
- profiles: `/opt/LightX2V/deploy/gpustack-lx2v-launcher/profiles.yaml`
- configs:  `/opt/LightX2V/configs/...` (what `profiles.yaml` points at — baked in,
  no NFS dependency)

The Dockerfile installs a `/usr/local/bin/gpustack-lx2v-launcher` wrapper (see
the `gpustack-lx2v-launcher` block added to `Dockerfile_aarch64_app`).

## Status

- **z_image** — ready: `profiles.yaml` points at
  `configs/z_image/z_image_a100_sage.json` (infer_steps=9, sage_attn2,
  enable_cfg=false, rope=torch — A100-safe).
- **wan2.2_moe (4-card int8)** — `profiles.yaml` has a **TODO placeholder**: the
  A100 int8 4-card ulysses config is not yet in `configs/`. Create it and update
  the `config_json` path before deploying wan. The parallel mesh **must** live in
  that config JSON's `"parallel"` block (e.g.
  `{"seq_p_size": 4, "cfg_p_size": 1, "seq_p_attn_type": "ulysses"}`) — the engine
  reads `config["parallel"]`, and top-level CLI args do NOT configure
  parallelism. The launcher only runs `torchrun --nproc_per_node=N` and validates
  the GPU count; it does not (and cannot) inject the mesh via CLI.

## Smoke test (standalone, no GPUStack)

```
CUDA_VISIBLE_DEVICES=0 python deploy/gpustack-lx2v-launcher/gpustack_lx2v_launcher.py \
  --model /path/to/z_image --port 8080 --host 0.0.0.0
# GET  http://localhost:8080/ready        -> 503 then 200 after the engine serves
# POST http://localhost:8080/v1/tasks/image/  (forwarded to the engine)
```

## Notes

- `model_cls` is inferred from the model path (`z_image` / `wan2.2_moe`); pass
  `--model-cls` (via GPUStack backend parameters) if paths are ambiguous.
- GPU count comes from `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` injected
  by GPUStack. z_image = 1-card replicas, wan = one 4-card replica; the launcher
  asserts `parallel_product == gpu_count`.
- PyYAML must be present in the image (it already is for LightX2V).
