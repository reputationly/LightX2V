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

`profiles.yaml` 只提供两个 SR 能力（2026-11 上游同步后的范围）：

- **seedvr2** — `seedvr2-3b/seg121-1card`（单卡，`configs/seedvr/a100/seedvr2_3b_seg121.json`）
  与 `seedvr2-3b/seg121-4card`（4 卡段级数据并行，`seg_p_size: 4`）。
- **swiftvr** — `swiftvr/1card`（反相双跑，`configs/swiftvr/a100/swiftvr_antiphase.json`）
  与 `swiftvr/segp4`（4 卡，上游 `chunk_p_size: 4`；变体名保持不变，GPUStack 的
  model 按 `--profile swiftvr/segp4` 引用它）。

其余模型（视频生成/编辑、图像、数字人、V2A）已迁到 vLLM-Omni：本引擎不再提供它们的
profile，按这些模型扩容会在 `_load_profile` 阶段明确报错退出，而不是静默起错服务。

Every multi-GPU config carries its real `"parallel"` mesh in JSON. The launcher
only runs `torchrun --nproc_per_node=N` and validates the GPU count; it does not
and cannot inject the mesh via top-level CLI arguments.

## Smoke test (standalone, no GPUStack)

```
CUDA_VISIBLE_DEVICES=0 python deploy/gpustack-lx2v-launcher/gpustack_lx2v_launcher.py \
  --model /path/to/SeedVR2-3B --port 8080 --host 0.0.0.0
# GET  http://localhost:8080/ready        -> 503 then 200 after the engine serves
# POST http://localhost:8080/v1/tasks/video/  (forwarded to the engine)
```

## Notes

- `model_cls` is inferred from the model path (`z_image` / `wan2.2_moe`); pass
  `--model-cls` (via GPUStack backend parameters) if paths are ambiguous.
- GPU count comes from `CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` injected
  by GPUStack. z_image = 1-card replicas, wan = one 4-card replica; the launcher
  asserts `parallel_product == gpu_count`.
- PyYAML must be present in the image (it already is for LightX2V).

## Lint before you push

The repo's `lint` GitHub workflow runs `pre-commit run --all-files`, which is a
**separate** workflow from `Build ARM64 Docker Image` — the build can go green
while `lint` fails, so check `lint` after pushing Python changes. The failing
hook is almost always `ruff-format`. It is pinned to **ruff v0.11.0** with
`--config=pyproject.toml` (line-length 200); pre-commit runs it in its own
isolated env, so a locally-installed newer ruff can disagree. Reproduce the CI
result with the exact version:

```
python3 -m venv /tmp/rufflint
/tmp/rufflint/bin/pip install ruff==0.11.0
/tmp/rufflint/bin/ruff format --check --config=pyproject.toml .   # lists files to reformat
/tmp/rufflint/bin/ruff format        --config=pyproject.toml <file>   # fix in place
/tmp/rufflint/bin/ruff check         --config=pyproject.toml .   # the `ruff` (lint) hook
```
