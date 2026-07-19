#!/usr/bin/env python3
# =============================================================================
# HunyuanImage-3.0-Instruct-Distil (NF4) —— 流水线并发 POC
#   验证群里 5090 的打法:模型按层摊多卡后,同时喂 N 张图让各卡同时干活。
#   目标:单流 87s/张(3卡利用率~1/3)→ 3 并发吞吐逼近 3×。
#   风险:generate_image 线程不安全 → 崩溃或花图,本脚本就是来验这个的。
#   跑法(容器内):
#     python hy3_concurrent_test.py                     # 默认 3 并发 × 2 轮
#     HY3_CONC=2 python hy3_concurrent_test.py          # 保守先试 2 并发
#   产物:hy3_conc_out-<n>.png,逐张人工核验有没有串台/花图。
# =============================================================================
import os
import json
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL = os.environ.get(
    "HY3_MODEL",
    "/nfs-models/wuhanjisuan894/models/HunyuanImage-3.0-Instruct-Distil-NF4-v2",
)
OUT_DIR = os.environ.get("HY3_OUT_DIR", "/nfs-models/wuhanjisuan894")
CONC    = int(os.environ.get("HY3_CONC", "3"))     # 并发数
ROUNDS  = int(os.environ.get("HY3_ROUNDS", "2"))   # 每个并发槽连发几张
STEPS   = int(os.environ.get("HY3_STEPS", "8"))
GPU_CAP = os.environ.get("HY3_GPU_CAP", "30GiB")
STAGGER = float(os.environ.get("HY3_STAGGER", "5"))  # 错峰提交间隔(秒)
TOPO    = os.environ.get("HY3_TOPO", "auto")   # auto | group(群友拓扑:base钉cuda:0,层均分其余卡)
MODE    = os.environ.get("HY3_MODE", "batch")  # batch=原生批量(安全,推荐) | thread=多线程(已审计:不安全,仅验证用)


def build_group_device_map(model_dir, ndev):
    """复刻群友拓扑:VAE/encoder/embedding 等 base 模块钉 cuda:0,
    transformer 层均分到 cuda:1..N。模块名从权重索引自动识别。"""
    idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    names = list(idx["weight_map"].keys())
    layer_ids = set()
    base_modules = set()
    for n in names:
        if n.startswith("model.layers."):
            layer_ids.add(int(n.split(".")[2]))
        else:
            parts = n.split(".")
            # model.wte / model.ln_f 取两段,vae / vision_model 等取一段
            base_modules.add(".".join(parts[:2]) if parts[0] == "model" else parts[0])
    n_layers = max(layer_ids) + 1
    dm = {m: 0 for m in base_modules}
    gpus = list(range(1, ndev))
    per = math.ceil(n_layers / len(gpus))
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = gpus[min(i // per, len(gpus) - 1)]
    print(f">>> group topo: {len(base_modules)} base modules -> cuda:0 | "
          f"{n_layers} layers -> cuda:1..{ndev - 1} (~{per}/gpu)", flush=True)
    return dm

# 每个槽不同 prompt,串台(输出对不上 prompt)一眼可见
PROMPTS = [
    "A photorealistic close-up portrait of a red fox in a snowy forest, soft morning light",
    "A blue vintage car parked on a rainy Tokyo street at night, neon reflections",
    "A wooden sailboat on a calm turquoise sea at golden hour, aerial view",
    "A steaming cup of coffee on a rustic table, macro shot, shallow depth of field",
]

def vram(tag):
    parts = []
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        parts.append(f"cuda:{i} {(total - free) / 2**30:.1f}G")
    print(f"  [{tag}] " + " | ".join(parts), flush=True)

def find_image(o):
    if isinstance(o, Image.Image):
        return o
    if isinstance(o, dict):
        o = list(o.values())
    if isinstance(o, (list, tuple)):
        for x in o:
            r = find_image(x)
            if r is not None:
                return r
    return None

def main():
    print(f">>> conc={CONC} rounds={ROUNDS} stagger={STAGGER}s steps={STEPS}", flush=True)
    t0 = time.time()
    ndev = torch.cuda.device_count()
    device_map = "auto"
    if TOPO == "group" and ndev >= 2:
        try:
            device_map = build_group_device_map(MODEL, ndev)
        except Exception as e:
            print(f">>> group topo failed ({e}), fallback to auto", flush=True)
    max_mem = {i: GPU_CAP for i in range(ndev)}
    max_mem["cpu"] = "200GiB"
    load_kwargs = dict(max_memory=max_mem) if device_map == "auto" else {}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
        moe_impl="eager",
        moe_drop_tokens=True,
        **load_kwargs,
    )
    model.eval()
    model.load_tokenizer(MODEL)
    print(f">>> loaded in {time.time() - t0:.0f}s", flush=True)
    vram("after-load")

    # 单流热身一张,顺便拿单流基准
    print(">>> warmup (serial baseline) ...", flush=True)
    t = time.time()
    model.generate_image(prompt=PROMPTS[0], seed=1, image_size="1024x1024",
                         bot_task="image", use_system_prompt="en_vanilla",
                         diff_infer_steps=STEPS, verbose=0)
    serial = time.time() - t
    print(f">>> serial baseline: {serial:.1f}s", flush=True)

    if MODE == "batch":
        # 原生批量:prompt 传 list,一次调用出 N 张(prepare_model_inputs 支持 str|list[str])
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(CONC)]
        for rnd in range(ROUNDS):
            print(f">>> batch round {rnd + 1}/{ROUNDS}: {CONC} prompts in one call ...", flush=True)
            t = time.time()
            out = model.generate_image(
                prompt=prompts, seed=100 + rnd, image_size="1024x1024",
                bot_task="image", use_system_prompt="en_vanilla",
                diff_infer_steps=STEPS, verbose=1,
            )
            dt = time.time() - t
            imgs = out if isinstance(out, (list, tuple)) else [out]
            imgs = [find_image(x) for x in imgs]
            imgs = [x for x in imgs if x is not None]
            for i, im in enumerate(imgs):
                im.save(f"{OUT_DIR}/hy3_batch_out-r{rnd}-{i}.png")
            vram(f"after-batch-{rnd + 1}")
            print(f">>> batch {CONC}: {dt:.1f}s 总 | {dt / max(len(imgs), 1):.1f}s/张 "
                  f"| 吞吐 {len(imgs) / dt * 60:.2f} 张/分 "
                  f"({(len(imgs) / dt * 60) / (60 / serial):.2f}x 单流) | 出图 {len(imgs)}/{CONC}",
                  flush=True)
        print(">>> 逐张检查 hy3_batch_out-*.png:内容须与 prompt 对应、无花图", flush=True)
        return

    lock = threading.Lock()
    results = []

    def worker(slot, rnd):
        idx = slot + rnd * CONC
        time.sleep(slot * STAGGER)  # 错峰进流水线
        t = time.time()
        try:
            out = model.generate_image(
                prompt=PROMPTS[slot % len(PROMPTS)], seed=100 + idx,
                image_size="1024x1024", bot_task="image",
                use_system_prompt="en_vanilla",
                diff_infer_steps=STEPS, verbose=0,
            )
            dt = time.time() - t
            img = find_image(out)
            ok = img is not None
            if ok:
                img.save(f"{OUT_DIR}/hy3_conc_out-{idx}.png")
            with lock:
                results.append((idx, dt, ok))
                print(f"  [slot{slot} r{rnd}] {'OK' if ok else 'NO-IMG'} {dt:.1f}s", flush=True)
        except Exception as e:
            with lock:
                results.append((idx, time.time() - t, False))
                print(f"  [slot{slot} r{rnd}] FAILED {type(e).__name__}: {e}", flush=True)

    print(f">>> concurrent: {CONC} slots x {ROUNDS} rounds ...", flush=True)
    t_all = time.time()
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        futs = [ex.submit(worker, s, r) for r in range(ROUNDS) for s in range(CONC)]
        for f in futs:
            f.result()
    wall = time.time() - t_all
    vram("after-concurrent")

    n_ok = sum(1 for _, _, ok in results if ok)
    n = len(results)
    lats = [dt for _, dt, ok in results if ok]
    print("=" * 50, flush=True)
    print(f">>> 单流基准: {serial:.1f}s/张  (吞吐 {60 / serial:.2f} 张/分)", flush=True)
    print(f">>> 并发 {CONC}: 成功 {n_ok}/{n}, 总墙钟 {wall:.1f}s, "
          f"吞吐 {n_ok / wall * 60:.2f} 张/分 "
          f"({(n_ok / wall * 60) / (60 / serial):.2f}x 单流)", flush=True)
    if lats:
        print(f">>> 并发下单张时延: 均值 {sum(lats) / len(lats):.1f}s, "
              f"min {min(lats):.1f}s, max {max(lats):.1f}s", flush=True)
    print(">>> 逐张检查 hy3_conc_out-*.png:内容必须和各自 prompt 对上、无花图", flush=True)

if __name__ == "__main__":
    main()
