"""验收用:检查「我们自己的配置」里有没有上游代码已不再读取的键。

用法(在仓库根目录):
    python docs/上游同步-2026-11/audit_our_config_keys.py                      # 合并分支自检: 配置@HEAD vs 代码@HEAD
    python docs/上游同步-2026-11/audit_our_config_keys.py --code-rev upstream/main  # 同步前摸底: 配置@HEAD vs 上游代码

「我们的配置」= merge-base..config-rev 之间新增/修改过的 configs/**/*.json,外加
deploy/gpustack-lx2v-launcher/profiles.yaml 里 config_json 指向的文件。

退出码: 出现 LEGACY_KEYS 中任一键 -> 1;否则 0。其余「代码里搜不到的键」只打印,由人判断
(模型结构键、我们自己的新特性键在代码@HEAD 里应当能搜到)。
"""

import argparse
import collections
import json
import re
import subprocess
import sys

LEGACY_KEYS = {
    "target_video_length",
    "target_height",
    "target_width",
    "target_fps",
    "parallel.seq_p_fp8_comm",
    "parallel.seq_p_fp4_comm",
    "parallel.seg_p_size@swiftvr",
    "seg_warmup_chunks@swiftvr",
    "resize_mode@seedvr",
    "video_frame_interpolation",
}
NESTED = {"parallel", "video_frame_interpolation", "dynamic_sparse_attn_setting", "general_sparse_attn_setting", "video_super_resolution", "fls", "ar_config"}


def git(*args):
    return subprocess.check_output(["git", *args]).decode(errors="ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-rev", default="HEAD")
    ap.add_argument("--code-rev", default="HEAD")
    ap.add_argument("--base", default=None, help="默认 merge-base(config-rev, upstream/main)")
    args = ap.parse_args()

    base = args.base or git("merge-base", args.config_rev, "upstream/main").strip()
    cfgs = [f for f in git("diff", "--name-only", "--diff-filter=AM", base, args.config_rev, "--", "configs").split() if f.endswith(".json")]
    try:
        profiles = git("show", f"{args.config_rev}:deploy/gpustack-lx2v-launcher/profiles.yaml")
        cfgs += [p.replace("/opt/LightX2V/", "") for p in re.findall(r"config_json:\s*(\S+)", profiles)]
    except subprocess.CalledProcessError:
        pass
    cfgs = sorted(set(cfgs))

    code_files = [f for f in git("ls-tree", "-r", "--name-only", args.code_rev, "--", "lightx2v", "lightx2v_platform").split() if f.endswith(".py")]
    corpus = "\n".join(git("show", f"{args.code_rev}:{f}") for f in code_files)

    usage = collections.defaultdict(list)
    legacy_hits = collections.defaultdict(list)
    for f in cfgs:
        try:
            data = json.loads(git("show", f"{args.config_rev}:{f}"))
        except Exception as e:  # noqa: BLE001
            print(f"BAD/MISSING {f}: {e}")
            continue
        family = next((name for name in ("swiftvr", "seedvr") if f"/{name}/" in f), None)

        def walk(d, prefix):
            for k, v in d.items():
                full = f"{prefix}.{k}" if prefix else k
                usage[(full, k)].append(f)
                tag = f"{full}@{family}" if family else full
                if full in LEGACY_KEYS or tag in LEGACY_KEYS:
                    legacy_hits[tag].append(f)
                if isinstance(v, dict) and k in NESTED:
                    walk(v, full)

        if isinstance(data, dict):
            walk(data, "")

    print(f"checked {len(cfgs)} configs against code@{args.code_rev}\n")
    print("== 代码里搜不到的键(人工判断:死键可删 / 漏迁移) ==")
    for (full, k), fs in sorted(usage.items(), key=lambda kv: -len(kv[1])):
        if not re.search(r"[\"']" + re.escape(k) + r"[\"']", corpus):
            print(f"{len(fs):3d} {full:45s} e.g. {sorted(set(fs))[:2]}")
    print("\n== LEGACY 键(必须为空) ==")
    for k, fs in legacy_hits.items():
        print(f"{len(fs):3d} {k:45s} e.g. {sorted(set(fs))[:3]}")
    sys.exit(1 if legacy_hits else 0)


if __name__ == "__main__":
    main()
