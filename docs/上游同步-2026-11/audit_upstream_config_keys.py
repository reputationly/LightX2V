"""列出上一个同步周期里,上游相对 merge-base 在 configs/ 下删掉和新增的配置键。

一次性审计工具(2026-11 同步周期用),不参与运行路径。用法:

    git fetch upstream
    python docs/上游同步-2026-11/audit_upstream_config_keys.py
"""

import collections
import json
import subprocess

BASE = subprocess.check_output(["git", "merge-base", "HEAD", "upstream/main"]).decode().strip()
UPSTREAM = "upstream/main"


def load(rev, path):
    try:
        return json.loads(subprocess.check_output(["git", "show", f"{rev}:{path}"], stderr=subprocess.DEVNULL))
    except Exception:
        return None


def flat(config, prefix=""):
    out = {}
    if isinstance(config, dict):
        for key, value in config.items():
            dotted = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict) and key not in ("lora_configs",):
                out.update(flat(value, dotted))
            else:
                out[dotted] = value
    return out


def changed_config_files():
    diff = subprocess.check_output(["git", "diff", "--name-status", "-M", BASE, UPSTREAM, "--", "configs"]).decode()
    for line in diff.splitlines():
        parts = line.split("\t")
        status = parts[0]
        if status.startswith("R"):
            yield parts[1], parts[2]
        elif status == "M":
            yield parts[1], parts[1]


def main():
    removed = collections.Counter()
    added = collections.Counter()
    pairs = collections.Counter()
    examples = {}

    for old_path, new_path in changed_config_files():
        if not old_path.endswith(".json"):
            continue
        before, after = load(BASE, old_path), load(UPSTREAM, new_path)
        if before is None or after is None:
            continue
        left, right = flat(before), flat(after)
        gone, new = set(left) - set(right), set(right) - set(left)
        for key in gone:
            removed[key] += 1
        for key in new:
            added[key] += 1
        for key in gone:
            for other in new:
                pairs[(key, other)] += 1
                examples.setdefault((key, other), (old_path, left[key], right[other]))

    print("=== REMOVED keys (count)")
    for key, count in removed.most_common(80):
        print(f"{count:4d} {key}")
    print("=== ADDED keys (count)")
    for key, count in added.most_common(60):
        print(f"{count:4d} {key}")
    print(f"=== pairs: {len(pairs)} ===")


if __name__ == "__main__":
    main()
