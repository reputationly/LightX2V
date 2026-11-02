"""SwiftVR 输出尺寸三条路:size / target_short_edge / sr_ratio(含封顶与显存护栏)。"""

from types import SimpleNamespace

import pytest

from lightx2v.models.runners.swiftvr.swiftvr_runner import SwiftVRRunner

TIER_1080 = [1080, 1920]
CEIL_4K = {"max_target_height": 2160, "max_target_width": 3840}


def _runner(**config):
    runner = SwiftVRRunner.__new__(SwiftVRRunner)
    runner.config = {"sr_tier_size": TIER_1080, **CEIL_4K, **config}
    return runner


def _input(size=None, short_edge=0, ratio=2.0):
    """真 SRInputInfo 的字段姿势(不带 torch 侧状态)。"""
    return SimpleNamespace(size=list(size or []), target_short_edge=short_edge, sr_ratio=ratio)


def test_size_is_exact_and_wins():
    """显式 size:调用方完全指定,不吃档位、也不按面积缩。"""
    out = _runner().resolve_output_size(_input(size=[1440, 2560], short_edge=1080, ratio=4.0), 768, 1344)
    assert out == (1440, 2560)


def test_target_short_edge_keeps_source_aspect():
    """短边命中 2K,长边按源画幅推出来(1344x768 = 1.75),不是 16:9 的 2560。"""
    height, width = _runner().resolve_output_size(_input(short_edge=1440, ratio=4.0), 768, 1344)
    assert height == 1440
    assert width == round(1344 * (1440 / 768))  # 2520
    assert abs(width / height - 1344 / 768) < 1e-6


def test_sr_ratio_below_tier_is_not_capped():
    """832x480 乘 1.5 只有 720x1248,没到 1080P 档,原样放大。"""
    assert _runner().resolve_output_size(_input(ratio=1.5), 480, 832) == (720, 1248)


def test_sr_ratio_is_capped_to_the_tier():
    """网关固定发 sr_ratio=4.0:768p 乘 4 会到 5.3K,必须落到 1080P 档的精确尺寸。"""
    assert _runner().resolve_output_size(_input(ratio=4.0), 768, 1344) == (1080, 1920)


def test_sr_ratio_tier_follows_source_orientation():
    """竖版源落在 1080x1920 这一档的竖版形态上。"""
    assert _runner().resolve_output_size(_input(ratio=4.0), 1344, 768) == (1920, 1080)


def test_size_is_clamped_by_the_vram_ceiling():
    """8K 的 size 必须按面积压到 4K 以内,且保持画幅比。"""
    height, width = _runner().resolve_output_size(_input(size=[4320, 7680]), 768, 1344)
    assert height * width <= 2160 * 3840
    assert height < 4320 and width < 7680
    assert abs((width / height) - (7680 / 4320)) < 1e-3


def test_ceiling_can_be_disabled():
    """max_target_* 配 0 = 关闭护栏(自建/离线跑大图)。"""
    runner = _runner(max_target_height=0, max_target_width=0)
    assert runner.resolve_output_size(_input(size=[4320, 7680]), 768, 1344) == (4320, 7680)


def test_require_even_rounds_to_even_edges():
    height, width = _runner().resolve_output_size(_input(short_edge=1081, ratio=4.0), 480, 832, require_even=True)
    assert height % 2 == 0 and width % 2 == 0
    assert height == 1080


def test_bad_size_length_is_rejected():
    with pytest.raises(ValueError, match=r"size must be \[height, width\]"):
        _runner().resolve_output_size(_input(size=[1080]), 768, 1344)
