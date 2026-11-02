#!/bin/bash
# §8.1 第三关的**配置键粒度**版本(字面版见 ws8-grep-literal.txt:它对上游自己的代码恒误报,
# hunyuan/wan/ltx2/worldmirror 等文件里的 target_height 只是普通局部变量名)。
# 判据:上游已删除的旧键(带引号的键名形态)不得出现在「我们改过的文件」里,唯一例外是兼容层。
set -u
cd "$(git rev-parse --show-toplevel)"
KEYS='target_height|target_width|target_shape|target_video_length|target_fps|video_frame_interpolation|seg_warmup_chunks|rife'
rc=0

echo "== 1) configs/ deploy/ 里不得出现旧键(JSON 键形态) =="
hits=$(grep -rnE "\"($KEYS)\"[[:space:]]*:" configs deploy 2>/dev/null || true)
if [ -n "$hits" ]; then echo "$hits"; rc=1; else echo "   (none)"; fi

echo "== 2) 我们改过的 .py 里,旧键(带引号)只允许出现在 legacy_request_compat.py =="
hits=$(git diff --name-only upstream/main HEAD \
  | grep -E '^(lightx2v|lightx2v_platform)/.*\.py$' \
  | grep -v '^lightx2v/server/legacy_request_compat\.py$' \
  | while read -r f; do [ -f "$f" ] && grep -nE "\"($KEYS)\"" "$f" | sed "s#^#$f:#"; done || true)
if [ -n "$hits" ]; then echo "$hits"; rc=1; else echo "   (none)"; fi

echo "== 3) 代码不得再读 config 的旧键(config.get(\"x\") / config[\"x\"]) =="
hits=$(grep -rnE "config(\.get\(|\[)['\"]($KEYS)['\"]" --include=*.py lightx2v | grep -v legacy_request_compat.py || true)
if [ -n "$hits" ]; then echo "$hits"; rc=1; else echo "   (none)"; fi

echo "rc=$rc"
exit $rc
