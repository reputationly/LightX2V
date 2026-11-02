"""worker 侧宽松字段过滤:网关多带的键要被丢掉 + warning,而不是让 runner 抛错。"""

import pytest

from lightx2v.server.legacy_request_compat import normalize_worker_request

SR_FIELDS = frozenset({"task", "video_path", "image_path", "sr_ratio", "size", "match_target_size", "seed", "save_result_path", "prompt"})


class _FakeRunner:
    """只实现过滤要用到的那两个属性。

    ``get_supported_request_fields`` 对未注册的 task 抛 KeyError —— 与上游
    ``supported_request_fields_by_task[task]`` 的行为一致。
    """

    def __init__(self, task="sr", enable_cfg=False):
        self.config = {"task": task, "enable_cfg": enable_cfg}

    def get_supported_request_fields(self, task):
        if task not in ("sr",):
            raise KeyError(task)
        # 与上游一致:关 CFG 时 negative_prompt 不在白名单里
        if self.config.get("enable_cfg", False):
            return SR_FIELDS | {"negative_prompt"}
        return SR_FIELDS


@pytest.fixture()
def warnings():
    """收集 loguru 的 WARNING 文本(兼容层"丢字段"的审计证据)。"""
    from loguru import logger

    records = []
    handler_id = logger.add(lambda message: records.append(str(message)), level="WARNING")
    try:
        yield records
    finally:
        logger.remove(handler_id)


def test_drops_fields_the_runner_cannot_explain(warnings):
    task_data = {
        "task": "sr",
        "video_path": "/a.mp4",
        "sr_ratio": 4.0,
        "seed": 42,
        "target_short_edge": 1440,  # 网关对 SeedVR 也发,但 SeedVR 不读
        "target_fps": 32,  # RIFE 删除后的历史字段
        "negative_prompt": "",  # 关 CFG 时网关照样发
    }

    filtered = normalize_worker_request(_FakeRunner(), task_data)

    assert filtered == {"task": "sr", "video_path": "/a.mp4", "sr_ratio": 4.0, "seed": 42}
    assert any("dropping request fields" in line for line in warnings)
    assert any("target_short_edge" in line and "target_fps" in line and "negative_prompt" in line for line in warnings)


def test_negative_prompt_survives_when_cfg_is_enabled():
    """开 CFG 时 negative_prompt 是白名单字段,不能被丢。"""
    task_data = {"task": "sr", "video_path": "/a.mp4", "negative_prompt": "blurry"}
    assert normalize_worker_request(_FakeRunner(enable_cfg=True), task_data) == task_data


def test_keeps_everything_when_all_fields_are_supported():
    task_data = {"task": "sr", "video_path": "/a.mp4", "sr_ratio": 2.0}
    assert normalize_worker_request(_FakeRunner(), task_data) == task_data


def test_defaults_task_from_runner_config():
    task_data = {"video_path": "/a.mp4", "sr_ratio": 2.0, "target_fps": 32}
    assert normalize_worker_request(_FakeRunner(), task_data) == {"video_path": "/a.mp4", "sr_ratio": 2.0}


def test_unknown_task_is_left_to_prepare_request():
    """未注册的 task 不在这里吞掉:让 runner.prepare_request 抛它自己的错。"""
    task_data = {"task": "not-a-task", "video_path": "/a.mp4"}
    assert normalize_worker_request(_FakeRunner(), task_data) == task_data
