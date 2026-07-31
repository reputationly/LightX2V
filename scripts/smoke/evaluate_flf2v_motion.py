#!/usr/bin/env python3
"""Measure motion distribution and endpoint fidelity for FLF2V videos.

The metrics are intentionally lightweight: OpenCV Farneback flow for motion,
normalized L1 and SSIM for the first/last-frame constraints.  They are useful
for controlled A/B tests, not as a perceptual quality score across unrelated
videos.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def _read_video(path: Path) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if len(frames) < 2:
        raise ValueError(f"video must contain at least 2 frames: {path}")
    return frames, fps


def _read_reference(path: Path, shape: tuple[int, int], resize_mode: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot open image: {path}")
    height, width = shape
    if resize_mode == "cover":
        source_height, source_width = image.shape[:2]
        scale = max(width / source_width, height / source_height)
        resized_width = max(width, round(source_width * scale))
        resized_height = max(height, round(source_height * scale))
        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        left = (resized_width - width) // 2
        top = (resized_height - height) // 2
        return image[top : top + height, left : left + width]
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def _normalized_l1(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))) / 255.0)


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean channel-wise SSIM using the standard 11x11 Gaussian window."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    channel_scores = []
    for channel in range(a.shape[2]):
        x = a[:, :, channel]
        y = b[:, :, channel]
        mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y
        sigma_x2 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x2
        sigma_y2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y2
        sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_xy
        score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2))
        channel_scores.append(float(np.mean(score)))
    return float(np.mean(channel_scores))


def _motion_curve(frames: list[np.ndarray], flow_width: int) -> list[float]:
    height, width = frames[0].shape[:2]
    scale = min(1.0, flow_width / width)
    flow_size = (max(8, round(width * scale)), max(8, round(height * scale)))

    def gray(frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, flow_size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    previous = gray(frames[0])
    curve = []
    for frame in frames[1:]:
        current = gray(frame)
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        # Normalize to full-resolution pixels so runs using the same content at
        # different flow_width values remain approximately comparable.
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1]) / scale
        curve.append(float(np.mean(magnitude)))
        previous = current
    return curve


def _third_means(values: np.ndarray) -> list[float]:
    chunks = np.array_split(values, 3)
    return [float(np.mean(chunk)) if len(chunk) else 0.0 for chunk in chunks]


def evaluate(
    video_path: Path,
    first_frame_path: Path | None,
    last_frame_path: Path | None,
    flow_width: int,
    reference_resize: str,
) -> tuple[dict[str, object], list[float]]:
    frames, fps = _read_video(video_path)
    motion = np.asarray(_motion_curve(frames, flow_width), dtype=np.float64)
    thirds = _third_means(motion)
    frame_width = frames[0].shape[1]
    motion_mean_width_pct = float(np.mean(motion) / frame_width * 100)
    result: dict[str, object] = {
        "video": str(video_path),
        "width": frame_width,
        "height": frames[0].shape[0],
        "frames": len(frames),
        "fps": fps,
        "duration_seconds": len(frames) / fps if fps else None,
        "motion_mean_px": float(np.mean(motion)),
        "motion_mean_width_pct": motion_mean_width_pct,
        "motion_speed_width_pct_per_second": motion_mean_width_pct * fps if fps else None,
        "motion_median_px": float(np.median(motion)),
        "motion_p95_px": float(np.percentile(motion, 95)),
        "motion_peak_px": float(np.max(motion)),
        "motion_peak_transition": int(np.argmax(motion) + 1),
        "motion_first_third_px": thirds[0],
        "motion_middle_third_px": thirds[1],
        "motion_last_third_px": thirds[2],
        "motion_last_to_first_ratio": thirds[2] / thirds[0] if thirds[0] else None,
    }
    shape = frames[0].shape[:2]
    if first_frame_path is not None:
        reference = _read_reference(first_frame_path, shape, reference_resize)
        result["first_frame_l1"] = _normalized_l1(frames[0], reference)
        result["first_frame_ssim"] = _ssim(frames[0], reference)
    if last_frame_path is not None:
        reference = _read_reference(last_frame_path, shape, reference_resize)
        result["last_frame_l1"] = _normalized_l1(frames[-1], reference)
        result["last_frame_ssim"] = _ssim(frames[-1], reference)
    return result, motion.tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--first-frame", type=Path)
    parser.add_argument("--last-frame", type=Path)
    parser.add_argument("--flow-width", type=int, default=480)
    parser.add_argument("--reference-resize", choices=("stretch", "cover"), default="stretch")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--curve-csv", type=Path)
    args = parser.parse_args()
    if args.flow_width <= 0:
        parser.error("--flow-width must be positive")

    results = []
    curves = {}
    for video in args.videos:
        result, curve = evaluate(
            video,
            args.first_frame,
            args.last_frame,
            args.flow_width,
            args.reference_resize,
        )
        results.append(result)
        curves[str(video)] = curve

    payload = {"results": results}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    if args.curve_csv:
        args.curve_csv.parent.mkdir(parents=True, exist_ok=True)
        max_length = max(map(len, curves.values()))
        with args.curve_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            names = list(curves)
            writer.writerow(["transition", *names])
            for index in range(max_length):
                writer.writerow([index + 1, *[curves[name][index] if index < len(curves[name]) else "" for name in names]])


if __name__ == "__main__":
    main()
