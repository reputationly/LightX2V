#!/usr/bin/env python3
"""§8.3 的画质度量:尺寸/帧数、逐帧 PSNR、锐度、judder(2beat)。

    measure.py info    <a.mp4> [...]
    measure.py psnr    <a.mp4> <b.mp4>            # 逐帧 PSNR(mean/min),要求尺寸一致
    measure.py sharp   <a.mp4> [...]
    measure.py judder  <a.mp4> [...]              # 震荡度 = mean|d_{t+1}-d_t| / mean(d)

PSNR/尺寸用 ffmpeg,锐度与 judder 的公式与 /nfs-data/lx2v_test 下现网分析脚本同源
(cmp_metrics.py 的拉普拉斯方差、judder.py 的震荡度)。
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np


def probe(path):
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=width,height,nb_read_frames,r_frame_rate", "-of", "json", path],
        capture_output=True,
        text=True,
    ).stdout
    stream = json.loads(raw)["streams"][0]
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames": int(stream["nb_read_frames"]),
        "fps": stream["r_frame_rate"],
    }


def gray(path, width=480, height=270, limit=400):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-vf", f"scale={width}:{height}", "-frames:v", str(limit), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True,
    ).stdout
    array = np.frombuffer(raw, dtype=np.uint8)
    count = array.size // (width * height)
    if count == 0:
        raise RuntimeError(f"cannot decode {path}")
    return array[: count * width * height].reshape(count, height, width).astype(np.float32)


def sharpness(frames):
    return float(np.mean([(f[:-2, 1:-1] + f[1:-1, :-2] - 4 * f[1:-1, 1:-1] + f[1:-1, 2:] + f[2:, 1:-1]).var() for f in frames[::10]]))


def judder(frames):
    d = np.abs(frames[1:] - frames[:-1]).reshape(len(frames) - 1, -1).mean(axis=1)
    if len(d) < 3:
        return None
    return float(np.abs(d[1:] - d[:-1]).mean() / d.mean())


def psnr(a, b):
    """逐帧 PSNR(RGB,逐帧独立),返回 mean/min 与帧数。"""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", a, "-i", b, "-lavfi", "[0:v][1:v]psnr=stats_file=-", "-f", "null", "-"],
        capture_output=True,
        text=True,
    ).stdout
    values = []
    for line in out.splitlines():
        for token in line.split():
            if token.startswith("psnr_avg:"):
                values.append(float(token.split(":", 1)[1]))
    if not values:
        # ffmpeg 把 psnr 打到 stderr;重试一次只取汇总行
        text = subprocess.run(
            ["ffmpeg", "-v", "info", "-i", a, "-i", b, "-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"],
            capture_output=True,
            text=True,
        ).stderr
        for line in text.splitlines():
            if "PSNR" in line and "average:" in line:
                mean = float(line.split("average:")[1].split()[0])
                return {"frames": None, "psnr_mean": mean, "psnr_min": None}
        raise RuntimeError("psnr filter produced no stats (size mismatch?)")
    return {"frames": len(values), "psnr_mean": round(float(np.mean(values)), 3), "psnr_min": round(float(np.min(values)), 3)}


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    command, paths = sys.argv[1], sys.argv[2:]
    if command == "info":
        for path in paths:
            print(path, probe(path))
        return 0
    if command == "psnr":
        print(psnr(paths[0], paths[1]))
        return 0
    if command == "sharp":
        for path in paths:
            print(f"{path}: sharpness={sharpness(gray(path)):.1f}")
        return 0
    if command == "judder":
        for path in paths:
            print(f"{path}: judder={judder(gray(path)):.5f}")
        return 0
    print(f"unknown command {command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
