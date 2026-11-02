"""HTTP 兼容层:让网关零改动地打上游新 schema。

上游 `68d0ece5`/`fabad304`/`ec13e359` 把请求参数统一了(`target_shape→size`、
`target_video_length→num_frames`),并给 `BaseTaskRequest` 加了 `extra="forbid"`。
网关(现网 new-api)仍在发旧键和一堆自有键(`model`/`user_id`/`task_type`/`infer_steps`),
直接打过来就是 422/400。这一层的职责只有两个:

1. `normalize_request_payload`:schema 校验**之前**把旧键改名、把无法解释的值丢掉、
   把 schema 不认识的键丢掉(一次 warning 列全),让 `extra="forbid"` 继续生效;
2. `normalize_worker_request`:worker 侧按 runner 的 `supported_request_fields` 再过滤
   一遍 —— 网关对 SeedVR 也会带 `target_short_edge`、关 CFG 时的 `negative_prompt` 等,
   现网这些本来就被忽略,不该变成 4xx。

注意:CLI 路径(`lightx2v.infer`)不走这里,保持上游的严格校验。
"""

from __future__ import annotations

from typing import Any

from loguru import logger

# 旧名 -> 新名。只在**新名缺省**时改名;新名已存在则以新名为准并 warning。
LEGACY_RENAMES = {
    "target_shape": "size",
    "target_video_length": "num_frames",
}

# 上游已删除的请求键。网关还会发,但语义在新代码里没有对应物,丢弃且只记 info。
DROP_SILENTLY = {"infer_steps"}


def normalize_request_payload(data: dict, model_fields: set[str]) -> dict:
    """把网关的旧 payload 归一到当前 schema。

    - `LEGACY_RENAMES`:旧名改成新名;新名已存在时保留新名 + warning。
    - `size` 为字符串(网关会发 `"1920x1080"` 之类)一律丢弃:现网引擎本就忽略它,
      解析成 `[h, w]` 会改变行为。
    - 不在 `model_fields` 里的键丢弃,一次 warning 列全(否则 `extra="forbid"` 报 4xx)。
    - `DROP_SILENTLY` 里的键只记 info。
    """
    if not isinstance(data, dict):
        return data

    payload = dict(data)

    # 1) 旧名 -> 新名
    for old, new in LEGACY_RENAMES.items():
        if old not in payload:
            continue
        if new in payload:
            logger.warning(f"legacy request field {old!r} ignored: {new!r} is already set ({payload[new]!r})")
            payload.pop(old)
            continue
        payload[new] = payload.pop(old)

    # 2) size 字符串:引擎读不懂,按"没传"处理
    size = payload.get("size")
    if isinstance(size, str):
        logger.warning(f"request field 'size' is a string ({size!r}), ignoring it: expected [height, width]")
        payload.pop("size", None)

    # 3) 丢掉 schema 不认识的键
    unknown = sorted(set(payload) - model_fields)
    if unknown:
        quiet = [key for key in unknown if key in DROP_SILENTLY]
        loud = [key for key in unknown if key not in DROP_SILENTLY]
        if loud:
            logger.warning(f"dropping request fields unknown to this service: {', '.join(loud)}")
        if quiet:
            logger.info(f"dropping request fields removed upstream: {', '.join(quiet)}")
        payload = {key: value for key, value in payload.items() if key in model_fields}

    return payload


def normalize_worker_request(runner: Any, task_data: dict) -> dict:
    """按 runner 的字段白名单过滤请求(宽松),只留它能解释的键。

    `runner.prepare_request` 对多余字段是严格抛错的(`多一个就 ValueError`),而现网
    网关对 SeedVR 这种 SR 能力会顺手带上 `target_short_edge`、`target_fps`、
    关 CFG 时的 `negative_prompt` —— 这些在旧引擎里都被静默忽略,行为上等价于"没传"。
    在这里丢掉并 warning,保住"网关零改动"这条契约。
    """
    if not isinstance(task_data, dict):
        return task_data

    task = task_data.get("task") or runner.config.get("task")
    try:
        allowed = set(runner.get_supported_request_fields(task))
    except Exception:  # 未注册的 task:交给 prepare_request 抛它自己的错
        return task_data

    dropped = sorted(set(task_data) - allowed - {"task"})
    if not dropped:
        return task_data

    logger.warning(f"{type(runner).__name__} ({task}) dropping request fields it does not support: {', '.join(dropped)}")
    return {key: value for key, value in task_data.items() if key in allowed or key == "task"}
