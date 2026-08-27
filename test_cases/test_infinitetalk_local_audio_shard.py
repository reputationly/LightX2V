import importlib.util
import math
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).parents[1] / "lightx2v/models/networks/wan/infer/infinitetalk/audio_shard.py"
_SPEC = importlib.util.spec_from_file_location("infinitetalk_audio_shard", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
local_audio_shard_indices = _MODULE.local_audio_shard_indices


@pytest.mark.parametrize(
    ("grid_t", "grid_h", "grid_w", "world_size"),
    [
        (1, 2, 3, 4),
        (5, 2, 3, 4),
        (7, 3, 5, 4),
        (21, 11, 17, 4),
        (21, 11, 17, 3),
    ],
)
def test_local_audio_shard_repack_matches_full_frame_local_result(grid_t, grid_h, grid_w, world_size):
    spatial_tokens = grid_h * grid_w
    token_count = grid_t * spatial_tokens
    local_len = math.ceil(token_count / world_size)

    # A fake frame-local operation: each query receives an offset determined only
    # by its frame. Repacking shards must reproduce the full-sequence result exactly.
    full = torch.arange(token_count, dtype=torch.int64)
    expected = full + torch.div(full, spatial_tokens, rounding_mode="floor") * 10_000
    actual_parts = []

    for rank in range(world_size):
        valid_len, frame_start, frame_end, frame_slots, spatial_offsets = local_audio_shard_indices(
            token_count,
            spatial_tokens,
            local_len,
            rank,
        )
        assert 0 <= frame_start <= frame_end <= grid_t
        if valid_len == 0:
            continue

        global_start = rank * local_len
        local_values = full[global_start : global_start + valid_len]
        frame_buffer = torch.zeros(frame_end - frame_start, spatial_tokens, dtype=torch.int64)
        frame_buffer[frame_slots, spatial_offsets] = local_values
        frame_offsets = torch.arange(frame_start, frame_end, dtype=torch.int64).unsqueeze(1) * 10_000
        local_result = (frame_buffer + frame_offsets)[frame_slots, spatial_offsets]
        actual_parts.append(local_result)

    assert torch.equal(torch.cat(actual_parts), expected)


def test_local_audio_shard_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        local_audio_shard_indices(token_count=1, spatial_tokens=0, local_len=1, rank=0)
