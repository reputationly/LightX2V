#!/usr/bin/env python3
# =============================================================================
# HunyuanImage-3.0-Instruct-Distil (NF4) —— 单节点 A100×3 POC 冒烟脚本
#   目标:只验证"能不能加载 + 出一张图",不追求速度/画质。
#   容器内跑(需挂 /nfs-models、GPU device=1,2,3):
#     python /nfs-models/wuhanjisuan894/hy3_t2i_test.py
#   环境要点:
#     - NF4 权重已 bnb 量化,from_pretrained 不需再传 quantization_config
#     - moe_impl=eager 避开 ARM 上装不了的 flashinfer
#     - attn=sdpa(Instruct 官方只支持 sdpa)
#     - device_map=auto + max_memory 每卡留头,激活溢出自动落 CPU(233G 兜底)
#   关键(群里 5090 实测):必须关掉模型自带 enhance prompt(CoT think/recaption),
#     否则慢到离谱。用 bot_task="image" + use_system_prompt="en_vanilla" 走直生快路。
#   环境变量可覆盖:HY3_MODEL / HY3_OUT / HY3_PROMPT / HY3_STEPS / HY3_GPU_CAP
#                    HY3_SIZE / HY3_BOT_TASK / HY3_SYS
#   热态稳态:HY3_RUNS=4 → 同进程连发 4 张,丢首张取均值(测试纪律)
#   i2i:HY3_IMAGE=/path/a.png[,b.png] → 图生图编辑(此时 prompt 写编辑指令)
# =============================================================================
import os
import time
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

MODEL  = os.environ.get(
    "HY3_MODEL",
    "/nfs-models/wuhanjisuan894/models/HunyuanImage-3.0-Instruct-Distil-NF4-v2",
)
OUT    = os.environ.get("HY3_OUT", "/nfs-models/wuhanjisuan894/hy3_t2i_out.png")
PROMPT = os.environ.get(
    "HY3_PROMPT",
    "A photorealistic close-up portrait of a red fox in a snowy forest, "
    "soft morning light, 85mm lens, sharp focus",
)
STEPS   = int(os.environ.get("HY3_STEPS", "8"))       # 蒸馏版吃 8 步
GPU_CAP = os.environ.get("HY3_GPU_CAP", "30GiB")      # 每卡权重上限,留头给 KV/MoE 激活
SIZE     = os.environ.get("HY3_SIZE", "1024x1024")    # 显式尺寸(比 auto 稳)
BOT_TASK = os.environ.get("HY3_BOT_TASK", "image")    # image=直生不增强 | think_recaption=慢
SYS_PROMPT = os.environ.get("HY3_SYS", "en_vanilla")  # en_vanilla=纯生成无改写
RUNS     = int(os.environ.get("HY3_RUNS", "1"))       # >1 = 热态稳态(丢首张取均值)
IMAGES   = os.environ.get("HY3_IMAGE", "")            # i2i 输入图,逗号分隔多图(≤3)
TOPO     = os.environ.get("HY3_TOPO", "auto")         # auto | group(1+N:base钉cuda:0,层均分其余卡,需≥4卡才有意义)
MOE_IMPL = os.environ.get("HY3_MOE", "eager")         # eager | flashinfer(消掉eager路由的显存爆炸+提速,需flashinfer可用)


def vram(tag):
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"  [{tag}] cuda:{i} used {(total - free) / 2**30:5.1f}/{total / 2**30:.0f} GiB",
              flush=True)


def build_group_device_map(model_dir, ndev):
    """群友 1+N 拓扑:base(VAE/encoder/embedding)钉 cuda:0,transformer 层均分 cuda:1..N。
    模块名从权重索引自动识别。多图融合的路由尖峰只落在层卡上,每卡权重降到 ~15.5G。"""
    import json as _json
    import math as _math
    idx = _json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    layer_ids, base_modules = set(), set()
    for n in idx["weight_map"]:
        if n.startswith("model.layers."):
            layer_ids.add(int(n.split(".")[2]))
        else:
            parts = n.split(".")
            base_modules.add(".".join(parts[:2]) if parts[0] == "model" else parts[0])
    n_layers = max(layer_ids) + 1
    dm = {m: 0 for m in base_modules}
    gpus = list(range(1, ndev))
    per = _math.ceil(n_layers / len(gpus))
    for i in range(n_layers):
        dm[f"model.layers.{i}"] = gpus[min(i // per, len(gpus) - 1)]
    print(f">>> group topo: {len(base_modules)} base modules -> cuda:0 | "
          f"{n_layers} layers -> cuda:1..{ndev - 1} (~{per}/gpu)", flush=True)
    return dm


def find_image(o):
    """generate_image 返回结构不确定,递归找出第一张 PIL.Image。"""
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
    print(">>> torch", torch.__version__, "| cuda_ok", torch.cuda.is_available(),
          "| ndev", torch.cuda.device_count(), flush=True)
    print(">>> model =", MODEL, flush=True)
    vram("start")

    print(">>> loading model ...", flush=True)
    t0 = time.time()
    ndev = torch.cuda.device_count()
    if TOPO == "group" and ndev >= 2:
        device_map = build_group_device_map(MODEL, ndev)
        max_mem = None
    else:
        device_map = "auto"
        caps = [c.strip() for c in GPU_CAP.split(",")]  # 支持 "24GiB,33GiB" 每卡不对称配额
        max_mem = {i: caps[i % len(caps)] for i in range(ndev)}
        max_mem["cpu"] = "200GiB"
        print(">>> max_memory =", max_mem, flush=True)
    # 与作者量化配置一致(load_quantized_instruct_distil_nf4.py)
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    extra = {"max_memory": max_mem} if max_mem is not None else {}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=quant_config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        attn_implementation="sdpa",
        moe_impl=MOE_IMPL,
        moe_drop_tokens=True,
        **extra,
    )
    print(">>> moe_impl =", MOE_IMPL, flush=True)
    model.eval()
    model.load_tokenizer(MODEL)   # 官方要求:generate_image 前必须挂 tokenizer
    print(f">>> loaded in {time.time() - t0:.0f}s", flush=True)
    vram("after-load")

    gen_kwargs = dict(
        prompt=PROMPT,
        image_size=SIZE,
        bot_task=BOT_TASK,            # image = 直生,不 think/不改写(快路)
        use_system_prompt=SYS_PROMPT, # en_vanilla = 纯生成无增强
        diff_infer_steps=STEPS,
        verbose=2,
    )
    if IMAGES:                        # i2i:传入参考图 + 输出对齐原图尺寸
        gen_kwargs["image"] = [p.strip() for p in IMAGES.split(",") if p.strip()]
        # 40G 卡实测红线:≥3 参考图 = eager MoE 路由尖峰必 OOM(7 种摆法全灭),直接拒绝
        max_ref = int(os.environ.get("HY3_MAX_REF", "2"))
        if len(gen_kwargs["image"]) > max_ref:
            print(f"!! 拒绝:{len(gen_kwargs['image'])} 张参考图超过 40G 卡安全上限 "
                  f"{max_ref} 张(≥3 图实测必 OOM,见实验报告)。", flush=True)
            raise SystemExit(2)
        gen_kwargs["image_size"] = "auto"
        gen_kwargs["infer_align_image_size"] = True
        print(">>> i2i mode, input =", gen_kwargs["image"], flush=True)

    times = []
    for r in range(RUNS):
        print(f">>> generating run {r + 1}/{RUNS} ({STEPS} steps) | "
              f"bot_task={BOT_TASK} sys={SYS_PROMPT}", flush=True)
        t1 = time.time()
        out = model.generate_image(seed=42 + r, **gen_kwargs)
        dt = time.time() - t1
        times.append(dt)
        print(f">>> run {r + 1} generated in {dt:.1f}s", flush=True)
        vram(f"after-gen-{r + 1}")

        img = find_image(out)
        if img is None:
            print("!! no PIL image in output, type =", type(out), out, flush=True)
            raise SystemExit(1)
        path = OUT if RUNS == 1 else OUT.replace(".png", f"-{r + 1}.png")
        img.save(path)
        print(">>> SAVED", path, img.size, flush=True)

    if RUNS > 1:
        hot = times[1:]
        print(f">>> 热态稳态(丢首张): {sum(hot) / len(hot):.1f}s 均值 "
              f"| 各次 {[f'{t:.1f}' for t in times]}", flush=True)


if __name__ == "__main__":
    main()
