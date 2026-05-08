from __future__ import annotations

import math
from dataclasses import dataclass

STATE_LAYOUT_PER_LAYER = "per_layer_text_speaker_context_v1"
BRANCH_LAYOUT_COND1 = "cond1"
BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3 = "independent_text_speaker3"

DEFAULT_NUM_LAYERS = 12
DEFAULT_NUM_HEADS = 20
DEFAULT_HEAD_DIM = 64
FP16_BYTES = 2


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int")


@dataclass(frozen=True)
class CoreMLConditionBucket:
    sequence_length: int
    text_len: int
    speaker_context_len_bucket: int

    def __post_init__(self) -> None:
        _validate_positive_int("sequence_length", self.sequence_length)
        _validate_positive_int("text_len", self.text_len)
        _validate_positive_int(
            "speaker_context_len_bucket",
            self.speaker_context_len_bucket,
        )

    @property
    def c_ctx_bucket(self) -> int:
        return self.text_len + self.speaker_context_len_bucket

    def bucket_id(self, branch_layout: str) -> str:
        branch_count_for_layout(branch_layout)
        return (
            f"S{self.sequence_length}_T{self.text_len}_"
            f"R{self.speaker_context_len_bucket}_{branch_layout}"
        )


def expected_per_layer_state_names(num_layers: int = DEFAULT_NUM_LAYERS) -> tuple[str, ...]:
    _validate_positive_int("num_layers", num_layers)

    state_names: list[str] = []
    for layer_index in range(num_layers):
        state_names.append(f"context_k_l{layer_index:02d}")
        state_names.append(f"context_v_l{layer_index:02d}")
    state_names.append("valid_mask_state")
    return tuple(state_names)


def per_layer_state_shapes(
    bucket: CoreMLConditionBucket,
    branch_layout: str = BRANCH_LAYOUT_COND1,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
) -> dict[str, tuple[int, ...]]:
    batch_size = branch_count_for_layout(branch_layout)
    _validate_positive_int("num_layers", num_layers)
    _validate_positive_int("num_heads", num_heads)
    _validate_positive_int("head_dim", head_dim)

    kv_shape = (batch_size, bucket.c_ctx_bucket, num_heads, head_dim)
    shapes: dict[str, tuple[int, ...]] = {}
    for layer_index in range(num_layers):
        shapes[f"context_k_l{layer_index:02d}"] = kv_shape
        shapes[f"context_v_l{layer_index:02d}"] = kv_shape
    shapes["valid_mask_state"] = (batch_size, bucket.c_ctx_bucket)
    return shapes


def kv_bytes_per_branch(
    bucket: CoreMLConditionBucket,
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    _validate_positive_int("num_layers", num_layers)
    _validate_positive_int("num_heads", num_heads)
    _validate_positive_int("head_dim", head_dim)
    _validate_positive_int("dtype_bytes", dtype_bytes)

    return 2 * num_layers * bucket.c_ctx_bucket * num_heads * head_dim * dtype_bytes


def branch_count_for_layout(branch_layout: str) -> int:
    if branch_layout == BRANCH_LAYOUT_COND1:
        return 1
    if branch_layout == BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3:
        return 3
    raise ValueError(f"unknown branch_layout: {branch_layout}")


def kv_memory_bytes(
    bucket: CoreMLConditionBucket,
    branch_layouts: tuple[str, ...] | list[str],
    num_layers: int = DEFAULT_NUM_LAYERS,
    num_heads: int = DEFAULT_NUM_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    dtype_bytes: int = FP16_BYTES,
) -> int:
    return sum(
        branch_count_for_layout(branch_layout)
        * kv_bytes_per_branch(
            bucket,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype_bytes=dtype_bytes,
        )
        for branch_layout in branch_layouts
    )


def estimate_cfg_active_steps(
    num_steps: int,
    cfg_min_t: float = 0.5,
    cfg_max_t: float = 1.0,
    init_scale: float = 0.999,
) -> int:
    _validate_positive_int("num_steps", num_steps)
    if (
        not math.isfinite(cfg_min_t)
        or not math.isfinite(cfg_max_t)
        or cfg_min_t < 0.0
        or cfg_max_t > 1.0
        or cfg_min_t > cfg_max_t
    ):
        raise ValueError("cfg_min_t and cfg_max_t must satisfy 0.0 <= min <= max <= 1.0")
    if not math.isfinite(init_scale) or init_scale <= 0.0:
        raise ValueError("init_scale must be positive")

    active_steps = 0
    for step_index in range(num_steps):
        t = (1.0 - step_index / num_steps) * init_scale
        if cfg_min_t <= t <= cfg_max_t:
            active_steps += 1
    return active_steps
