# VACE 模式测试输入构造: 以 vace_r2v_int8.mp4(832x480,81f,16fps) 为素材
# 产出 /data/vace_inputs/{inpaint,outpaint,extend}_{src,mask}.mp4
# 在 lightx2v 容器内运行: python /data/smoke/vace_prep_inputs.py
import os

import av
import numpy as np

SRC = "/data/outputs/vace_r2v_int8.mp4"
OUTD = "/data/vace_inputs"
os.makedirs(OUTD, exist_ok=True)


def read_frames(path):
    c = av.open(path)
    s = c.streams.video[0]
    frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    c.close()
    return frames


def write_video(path, frames, fps=16):
    c = av.open(path, "w")
    s = c.add_stream("libx264", rate=fps)
    s.height, s.width = frames[0].shape[:2]
    s.pix_fmt = "yuv420p"
    s.options = {"crf": "12"}
    for arr in frames:
        for pkt in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            c.mux(pkt)
    for pkt in s.encode():
        c.mux(pkt)
    c.close()
    print("写出", path, len(frames), "帧")


frames = read_frames(SRC)
H, W = frames[0].shape[:2]
print(f"素材 {W}x{H} {len(frames)}帧")
white = np.full((H, W, 3), 255, dtype=np.uint8)
black = np.zeros((H, W, 3), dtype=np.uint8)
gray = np.full((H, W, 3), 128, dtype=np.uint8)

# 1) inpaint: 原片 + 中央矩形mask(白=重画)
y1, y2, x1, x2 = int(H * 0.15), int(H * 0.65), int(W * 0.30), int(W * 0.70)
m = black.copy()
m[y1:y2, x1:x2] = 255
write_video(f"{OUTD}/inpaint_src.mp4", frames)
write_video(f"{OUTD}/inpaint_mask.mp4", [m] * len(frames))

# 2) outpaint: 画面缩到60%居中灰底 + 周边mask白(重画)中心黑(保留)
sh, sw = int(H * 0.6) // 2 * 2, int(W * 0.6) // 2 * 2
top, left = (H - sh) // 2, (W - sw) // 2
import torch
import torch.nn.functional as F

small = [F.interpolate(torch.from_numpy(f).permute(2, 0, 1)[None].float(), size=(sh, sw), mode="bilinear")[0].permute(1, 2, 0).byte().numpy() for f in frames]
osrc, om = [], white.copy()
om[top : top + sh, left : left + sw] = 0
for f in small:
    canvas = gray.copy()
    canvas[top : top + sh, left : left + sw] = f
    osrc.append(canvas)
write_video(f"{OUTD}/outpaint_src.mp4", osrc)
write_video(f"{OUTD}/outpaint_mask.mp4", [om] * len(frames))

# 3) extend 续写: 首帧真实+其余灰; mask首帧黑(保留)其余白(生成)
write_video(f"{OUTD}/extend_src.mp4", [frames[0]] + [gray] * (len(frames) - 1))
write_video(f"{OUTD}/extend_mask.mp4", [black] + [white] * (len(frames) - 1))
print("全部完成")
