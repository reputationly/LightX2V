"""init_parallel 的段并行分支:与 mesh 轴互斥,且必须等于 world size。"""

import pytest

from lightx2v.utils import set_config as sc


class _FakeDist:
    """只实现 init_parallel 用到的那几个入口。"""

    def __init__(self, world_size):
        self._world_size = world_size
        self.collectives = 0

    def is_initialized(self):
        return True

    def get_world_size(self):
        return self._world_size

    def get_rank(self):
        return 0

    def all_reduce(self, *args, **kwargs):
        self.collectives += 1


@pytest.fixture()
def env(monkeypatch):
    """离线环境:CPU 设备 + 假 dist + 任何 mesh 初始化都算失败。"""

    def _apply(world_size):
        fake = _FakeDist(world_size)
        monkeypatch.setattr(sc, "dist", fake)
        monkeypatch.setattr(sc, "AI_DEVICE", "cpu", raising=False)
        monkeypatch.setattr(sc, "init_device_mesh", lambda *a, **k: (_ for _ in ()).throw(AssertionError("seg_p_size must not build a device mesh")))
        return fake

    return _apply


def _config(parallel, **extra):
    config = {"parallel": parallel, "model_cls": "seedvr2", "task": "sr"}
    config.update(extra)
    return config


def test_seg_p_size_sets_pure_segment_parallelism(env):
    fake = env(4)
    config = _config({"seg_p_size": 4})

    sc.init_parallel(config)

    assert config["seg_parallel"] is True
    assert config["tensor_parallel"] is False
    assert config["seq_parallel"] is False
    assert config["cfg_parallel"] is False
    assert config["pipefusion_parallel"] is False
    assert "device_mesh" not in config  # 段并行不建 mesh
    assert fake.collectives == 1  # 末尾的通信 warmup 照样执行


def test_seg_p_size_rejects_mesh_axes(env):
    env(4)
    config = _config({"seg_p_size": 4, "seq_p_size": 2})

    with pytest.raises(ValueError, match="exclusive"):
        sc.init_parallel(config)


def test_seg_p_size_must_equal_world_size(env):
    env(8)
    config = _config({"seg_p_size": 4})

    with pytest.raises(ValueError, match="must equal world_size"):
        sc.init_parallel(config)


def test_swiftvr_returns_early(monkeypatch):
    """SwiftVR 自己管 chunk 组,init_parallel 不该碰它。"""

    def _boom(*args, **kwargs):
        raise AssertionError("init_parallel must not touch swiftvr")

    monkeypatch.setattr(sc, "dist", type("D", (), {"is_initialized": staticmethod(_boom), "get_world_size": staticmethod(_boom)})())
    sc.init_parallel({"parallel": {"chunk_p_size": 4}, "model_cls": "swiftvr"})
