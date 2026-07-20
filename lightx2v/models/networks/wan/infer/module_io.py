from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class GridOutput:
    tensor: torch.Tensor
    tuple: tuple


@dataclass
class WanPreInferModuleOutput:
    embed: torch.Tensor
    grid_sizes: GridOutput
    x: torch.Tensor
    embed0: torch.Tensor
    context: torch.Tensor
    # 3D RoPE / position related
    cos_sin: Optional[torch.Tensor] = None
    rope_positions: Optional[torch.Tensor] = None
    valid_token_len: int = 0
    valid_latent_num: int = 0
    # extra
    adapter_args: Dict[str, Any] = field(default_factory=dict)
    conditional_dict: Dict[str, Any] = field(default_factory=dict)
    # bernini v2v: full-sequence boolean mask (True=target tokens, False=context).
    # Sliced AFTER the blocks (and after ulysses gather) so downstream sees only
    # target tokens. None for every non-v2v task. See wan_diffusion.py:504.
    v2v_target_mask: Optional[torch.Tensor] = None
