# inpaint 修正版输入: mask 区域灰底填充(模型看不见原内容, 才会按提示词重画)
# 容器内运行: python /data/smoke/vace_prep_inpaint2.py
import av

SRC = "/data/outputs/vace_r2v_int8.mp4"
c = av.open(SRC)
s = c.streams.video[0]
frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
c.close()

H, W = frames[0].shape[:2]
y1, y2, x1, x2 = int(H * 0.15), int(H * 0.65), int(W * 0.30), int(W * 0.70)  # 与 inpaint_mask.mp4 同区域

out = av.open("/data/vace_inputs/inpaint2_src.mp4", "w")
st = out.add_stream("libx264", rate=16)
st.height, st.width = H, W
st.pix_fmt = "yuv420p"
st.options = {"crf": "12"}
for f in frames:
    g = f.copy()
    g[y1:y2, x1:x2] = 128
    for pkt in st.encode(av.VideoFrame.from_ndarray(g, format="rgb24")):
        out.mux(pkt)
for pkt in st.encode():
    out.mux(pkt)
out.close()
print("写出 inpaint2_src.mp4", len(frames), "帧 (mask区域已灰填)")
