#!/usr/bin/env bash
set -euo pipefail

LIGHTX2V_PATH=${LIGHTX2V_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
ASSET_MANIFEST=${ASSET_MANIFEST:-${LIGHTX2V_PATH}/configs/wan22/bernini_s2v_assets_nfs.json}
CONFIG_JSON=${CONFIG_JSON:-${LIGHTX2V_PATH}/configs/wan22/a100/bernini_s2v_moe_4gpu.json}
HOST=${HOST:-0.0.0.0}
PORT=${PORT:-8000}
MASTER_PORT=${MASTER_PORT:-29533}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

eval "$(
python3 - "${ASSET_MANIFEST}" <<'PY'
import json
import shlex
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    assets = json.load(handle)
for env_name, key in (
    ("HIGH_NOISE_MODEL", "high_noise_model"),
    ("LOW_NOISE_MODEL", "low_noise_model"),
    ("SHARED_S2V_MODEL", "shared_s2v_model"),
    ("OUTPUT_ROOT", "output_root"),
):
    print(f"{env_name}={shlex.quote(assets[key])}")
PY
)"

python3 - "${ASSET_MANIFEST}" "${LOW_NOISE_MODEL}" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    assets = json.load(handle)
model_path = sys.argv[2]  # what torchrun passes as --model_path below

missing = []
for key in ("high_noise_model", "low_noise_model"):
    root = assets[key]
    required = ["config.json", "diffusion_pytorch_model.safetensors.index.json", "non_block.safetensors"]
    required += [f"block_{index}.safetensors" for index in range(40)]
    missing += [os.path.join(root, name) for name in required if not os.path.exists(os.path.join(root, name))]

shared_root = assets["shared_s2v_model"]
missing += [
    os.path.join(shared_root, name)
    for name in assets["required_shared_assets"]
    if not os.path.exists(os.path.join(shared_root, name))
]
if missing:
    raise SystemExit("Bernini-R-S2V assets are incomplete:\n  " + "\n  ".join(missing[:20]))

# The VAE, T5 checkpoint, T5 tokenizer and wav2vec front-end are all resolved by the
# runner relative to --model_path, NOT to shared_s2v_model. Checking only the shared
# root lets preflight pass and the service then die in load_audio_encoder / load_vae.
unreachable = [name for name in assets["runtime_probes"] if not os.path.exists(os.path.join(model_path, name))]
if unreachable:
    raise SystemExit(
        "Shared assets are not reachable from --model_path (%s):\n  %s\n\nLink them in:\n"
        "  for d in %s %s; do\n"
        "    for a in %s; do ln -sfn %s/$a $d/$a; done\n"
        "  done" % (
            model_path,
            "\n  ".join(os.path.join(model_path, name) for name in unreachable),
            assets["high_noise_model"],
            assets["low_noise_model"],
            " ".join(assets["runtime_linked_assets"]),
            shared_root,
        )
    )
print("Bernini-R-S2V asset preflight passed")
PY

mkdir -p "${OUTPUT_ROOT}"
cd "${LIGHTX2V_PATH}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    -m lightx2v.server \
    --model_cls wan2.2_s2v_moe \
    --task s2v \
    --model_path "${LOW_NOISE_MODEL}" \
    --config_json "${CONFIG_JSON}" \
    --host "${HOST}" \
    --port "${PORT}"
