"""网关兼容层:旧 payload 打新 schema 必须能过,且结果与"引擎本来就会忽略它"一致。"""

from lightx2v.server.schema import ImageTaskRequest, VideoTaskRequest


def test_legacy_payload_survives_and_keeps_contract_fields():
    """现网网关的真实形态:旧键 + size 字符串 + 一堆自有键。"""
    request = VideoTaskRequest(
        **{
            "video_path": "/a.mp4",
            "sr_ratio": 4.0,
            "target_short_edge": 1080,
            "size": "1920x1080",  # 网关会发字符串,引擎读不懂 -> 按没传处理
            "model": "seedvr2",
            "user_id": "1",
            "task_type": "sr",
        }
    )

    assert request.video_path == "/a.mp4"
    assert request.sr_ratio == 4.0
    assert request.target_short_edge == 1080
    assert request.size == []
    # 未声明的键不该出现在模型上(它们已被兼容层丢掉,而不是走 extra="forbid" 报 4xx)
    for dropped in ("model", "user_id", "task_type"):
        assert not hasattr(request, dropped)


def test_target_shape_renamed_to_size():
    request = VideoTaskRequest(**{"video_path": "/a.mp4", "target_shape": [2160, 3840]})
    assert request.size == [2160, 3840]


def test_new_name_wins_when_both_present():
    request = VideoTaskRequest(**{"video_path": "/a.mp4", "size": [1080, 1920], "target_shape": [720, 1280]})
    assert request.size == [1080, 1920]


def test_image_request_accepts_legacy_shape():
    request = ImageTaskRequest(**{"prompt": "x", "target_shape": [512, 512]})
    assert request.size == [512, 512]


def test_target_video_length_renamed_to_num_frames():
    request = VideoTaskRequest(**{"prompt": "x", "target_video_length": 81})
    assert request.num_frames == 81


def test_infer_steps_dropped_silently():
    """上游删了这个字段;网关还在发,丢掉即可,不能变成 4xx。"""
    request = VideoTaskRequest(**{"prompt": "x", "infer_steps": 4})
    assert not hasattr(request, "infer_steps")
