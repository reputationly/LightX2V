"""decord 垫片:用 PyAV 实现 LightX2V 用到的最小 API 面(aarch64 无 decord wheel)。"""

import av
import numpy as np

try:
    import torch
except Exception:  # torch 理论上一定在,防御性处理
    torch = None

_BRIDGE = "native"


class _NDArray:
    """native 桥接下的返回对象,提供 .asnumpy();torch 桥接下直接返回 tensor。"""

    def __init__(self, arr):
        self._arr = np.ascontiguousarray(arr)

    def asnumpy(self):
        return self._arr

    # --- 以下为补齐项 ---
    # 真 decord 返回的是 tvm NDArray,带 .shape/.dtype/.ndim。原垫片只给了 asnumpy(),
    # 凡是直接读这些属性的调用方都会 AttributeError —— SwiftVR 的 run_video_pipeline
    # 就是这么挂的(first_frame.shape[0] // 8 * 8)。
    @property
    def shape(self):
        return self._arr.shape

    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def ndim(self):
        return self._arr.ndim

    def __getattr__(self, name):
        # 兜住其余 ndarray 只读属性/方法(astype、mean、transpose ...)。
        # 只在常规查找失败后才走到这里,不会遮蔽上面显式定义的成员。
        return getattr(self.__dict__["_arr"], name)

    def __array__(self):
        return self._arr

    def __len__(self):
        return len(self._arr)

    def __getitem__(self, i):
        return self._arr[i]


def _wrap(arr):
    if _BRIDGE == "torch" and torch is not None:
        return torch.from_numpy(np.ascontiguousarray(arr))
    return _NDArray(arr)


class VideoReader:
    def __init__(self, video_path, num_threads=1, ctx=None, fault_tol=1, width=-1, height=-1, **kwargs):
        with av.open(video_path) as container:
            stream = container.streams.video[0]
            rate = stream.average_rate
            self._fps = float(rate) if rate else 16.0
            self._frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(video=0)])
        self._idx = 0

    def get_avg_fps(self):
        return self._fps

    def __len__(self):
        return self._frames.shape[0]

    def seek(self, pos):
        self._idx = int(pos)
        return None

    def next(self):
        frame = self._frames[self._idx]
        self._idx += 1
        return _wrap(frame)

    def get_frame_timestamp(self, i):
        return np.array([i / self._fps, (i + 1) / self._fps], dtype=np.float32)

    def get_batch(self, frame_ids):
        idx = np.clip(np.asarray(list(frame_ids), dtype=np.int64), 0, len(self) - 1)
        return _wrap(self._frames[idx])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return _wrap(self._frames[i])
        return _wrap(self._frames[int(i)])


class _Bridge:
    @staticmethod
    def set_bridge(name):
        global _BRIDGE
        _BRIDGE = name


bridge = _Bridge()


def cpu(index=0):
    return None


def gpu(index=0):
    return None


__version__ = "0.6.0+pyav-shim"
