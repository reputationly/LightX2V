"""Compare two videos frame by frame: PSNR, identical frames, and per-boundary detail.

python scripts/relayvsr/compare_outputs.py a.mp4 b.mp4 --boundaries 187 374 561
"""

import argparse
import json
import math

import av
import numpy as np


def read_frames(path):
    frames = []
    with av.open(str(path)) as container:
        container.streams.video[0].thread_type = "AUTO"
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))
    return frames


def probe(path):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        report = {
            "frames": int(stream.frames) if stream.frames else None,
            "width": stream.width,
            "height": stream.height,
            "fps": float(stream.average_rate) if stream.average_rate else None,
        }
        for candidate in container.streams:
            if candidate.type == "audio":
                report["audio_duration"] = float(candidate.duration * candidate.time_base) if candidate.duration else None
        report["duration"] = report["frames"] / report["fps"] if report["frames"] and report["fps"] else None
    return report


def psnr(left, right):
    if left.shape != right.shape:
        return None
    difference = left.astype(np.float64) - right.astype(np.float64)
    mse = float((difference * difference).mean())
    return math.inf if mse == 0 else 10 * math.log10(255.0**2 / mse)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--boundaries", type=int, nargs="*", default=[])
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=50.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    left, right = read_frames(args.left), read_frames(args.right)
    count = min(len(left), len(right))
    scores = [psnr(left[index], right[index]) for index in range(count)]
    for index in range(count, max(len(left), len(right))):
        scores.append(None)

    identical = sum(1 for score in scores if score == math.inf)
    finite = [score for score in scores if score is not None and score != math.inf]
    below = [index for index, score in enumerate(scores) if score is not None and score < args.threshold]

    report = {
        "left": probe(args.left),
        "right": probe(args.right),
        "compared_frames": count,
        "identical_frames": identical,
        "min_psnr": min(scores, key=lambda score: math.inf if score is None else score) if scores else None,
        "mean_psnr": (math.inf if not finite else float(np.mean(finite))) if identical + len(finite) == count else None,
        "frames_below_threshold": below[:50],
        "boundaries": {},
    }
    for boundary in args.boundaries:
        span = range(max(0, boundary - args.window), min(count, boundary + args.window + 1))
        report["boundaries"][boundary] = {index: scores[index] for index in span}

    print(f"frames: {len(left)} vs {len(right)} (compared {count}), identical {identical}")
    print(f"min PSNR: {report['min_psnr']:.2f} dB, mean (finite): {report['mean_psnr'] if report['mean_psnr'] is None else round(report['mean_psnr'], 2)}")
    print(f"below {args.threshold} dB: {len(below)} frames {below[:20]}")
    for boundary, detail in report["boundaries"].items():
        rendered = " ".join(f"{index}:{'inf' if score == math.inf else f'{score:.1f}'}" for index, score in detail.items())
        print(f"boundary {boundary}: {rendered}")
    print(f"left  {report['left']}")
    print(f"right {report['right']}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(report, handle, indent=2, default=str)


main()
