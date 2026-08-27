#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
离线把 LoRA 焊进 base DiT —— 复刻 LightX2V 运行时 LoRALoader 的合并语义。

为什么不用 tools/extract/lora_merger.py:那个只认 .lora_up/.lora_down 老格式,
对 Qwen/diffusers 格式(.lora_B/.lora_A、.lora_B.default.weight 等)会 0 命中。
这里直接复用 lightx2v.utils.lora_loader.LoRALoader,支持全部格式 + 正确的
transformer_blocks.N key 映射,和运行时 apply_lora 完全一致,只是一次性做完、
不走 cpu_offload 逐 block 搬运(运行时慢的根源)。

在 lightx2v 容器内跑(CPU-only,不占卡),PYTHONPATH=/opt/LightX2V:
  python /data/merge_qwen_lora.py \
    --base-transformer /data/models/Qwen-Image-2512/transformer \
    --lora /data/models/loras/Qwen-Image-2512-Lightning/Qwen-Image-2512-Lightning-8steps-V1.0-bf16.safetensors \
    --out  /data/models/merged/qwen_2512_lightning_8step_merged.safetensors \
    --strength 1.0

输出单文件 safetensors → 配置里用 "dit_original_ckpt" 指向它即可。
"""

import argparse
import glob
import os
import sys

import torch
from safetensors.torch import load_file, save_file

from lightx2v.utils.lora_loader import LoRALoader


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-transformer", required=True, help="base 的 transformer 目录(含分片 safetensors)或单文件")
    ap.add_argument("--lora", required=True, help="LoRA safetensors 路径")
    ap.add_argument("--out", required=True, help="输出合并后的单文件 safetensors")
    ap.add_argument("--strength", type=float, default=1.0, help="LoRA 强度(等价运行时 lora_configs.strength)")
    ap.add_argument("--compute-dtype", default="bf16", choices=["bf16", "fp32"], help="合并计算精度;fp32 更准但 RAM 翻倍")
    ap.add_argument("--out-dtype", default="bf16", choices=["bf16", "fp32"], help="落盘精度(默认 bf16,与 base 一致)")
    args = ap.parse_args()

    cdt = torch.bfloat16 if args.compute_dtype == "bf16" else torch.float32
    odt = torch.bfloat16 if args.out_dtype == "bf16" else torch.float32

    # 1) 读 base(union 所有分片)
    if os.path.isdir(args.base_transformer):
        files = sorted(glob.glob(os.path.join(args.base_transformer, "*.safetensors")))
    else:
        files = [args.base_transformer]
    if not files:
        print(f"!! 没找到 safetensors: {args.base_transformer}")
        sys.exit(2)
    print(f"[base] {len(files)} 个文件,读取中(compute={args.compute_dtype})...")
    wd = {}
    for f in files:
        print("  load", os.path.basename(f))
        d = load_file(f, device="cpu")
        for k, v in d.items():
            wd[k] = v.to(cdt) if v.dtype.is_floating_point else v
    print(f"[base] 权重张量 {len(wd)}")

    # 2) 读 lora
    lora = load_file(args.lora, device="cpu")
    print(f"[lora] 张量 {len(lora)}")

    # 3) 预检 key 映射命中率(0 命中就别继续)
    loader = LoRALoader()
    pairs = loader.extract_lora_pairs(lora)
    diffs = loader.extract_lora_diffs(lora)
    matched = sum(1 for mk in pairs if mk in wd) + sum(1 for mk in diffs if mk in wd)
    print(f"[map] lora pairs={len(pairs)} diffs={len(diffs)} | 命中 base={matched}")
    if matched == 0:
        print("!! 一个都没命中,key 映射对不上。示例 lora key:")
        for k in list(lora.keys())[:8]:
            print("    L:", k)
        print("示例 base key:")
        for k in list(wd.keys())[:8]:
            print("    B:", k)
        sys.exit(3)

    # 4) 复刻运行时合并(in-place 改 wd)
    n = loader.apply_lora(weight_dict=wd, lora_weights=lora, strength=args.strength)
    print(f"[merge] applied={n}(应接近命中数 {matched})")

    # 5) 落盘单文件
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    for k, v in wd.items():
        out[k] = (v.to(odt) if v.dtype.is_floating_point else v).contiguous()
    save_file(out, args.out)
    sz = os.path.getsize(args.out) / (1024**3)
    print(f"[done] -> {args.out}  ({sz:.1f} GB, {len(out)} 张量, dtype={args.out_dtype})")


if __name__ == "__main__":
    main()
