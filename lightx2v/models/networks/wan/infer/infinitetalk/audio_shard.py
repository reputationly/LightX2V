import torch


def local_audio_shard_indices(token_count, spatial_tokens, local_len, rank, device=None):
    """Map one contiguous padded sequence-parallel shard into video-frame slots."""
    if token_count < 0 or spatial_tokens <= 0 or local_len < 0 or rank < 0:
        raise ValueError("Invalid local audio shard dimensions")

    global_start = rank * local_len
    valid_len = max(0, min(local_len, token_count - global_start))
    if valid_len == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return 0, 0, 0, empty, empty

    global_indices = torch.arange(global_start, global_start + valid_len, device=device)
    frame_ids = torch.div(global_indices, spatial_tokens, rounding_mode="floor")
    frame_start = global_start // spatial_tokens
    frame_end = (global_start + valid_len - 1) // spatial_tokens + 1
    frame_slots = frame_ids - frame_start
    spatial_offsets = global_indices.remainder(spatial_tokens)
    return valid_len, frame_start, frame_end, frame_slots, spatial_offsets
