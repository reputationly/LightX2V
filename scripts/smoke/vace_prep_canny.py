# 生成 canny 边缘控制视频: /data/outputs/vace_r2v_int8.mp4 → /data/vace_inputs/canny_src.mp4
# 容器内运行: python /data/smoke/vace_prep_canny.py
import av
import cv2
import numpy as np

c = av.open("/data/outputs/vace_r2v_int8.mp4")
s = c.streams.video[0]
frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
c.close()

out = av.open("/data/vace_inputs/canny_src.mp4", "w")
st = out.add_stream("libx264", rate=16)
st.height, st.width = frames[0].shape[:2]
st.pix_fmt = "yuv420p"
st.options = {"crf": "12"}
for f in frames:
    edge = cv2.Canny(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), 100, 200)
    rgb = np.stack([edge] * 3, axis=-1)
    for pkt in st.encode(av.VideoFrame.from_ndarray(rgb, format="rgb24")):
        out.mux(pkt)
for pkt in st.encode():
    out.mux(pkt)
out.close()
print("写出 canny_src.mp4", len(frames), "帧")
