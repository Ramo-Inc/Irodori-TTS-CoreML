from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "irodori_tts" / "coreml_cache.py"
MODULE_SPEC = importlib.util.spec_from_file_location("coreml_cache_contract", MODULE_PATH)
assert MODULE_SPEC is not None
coreml_cache = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = coreml_cache
MODULE_SPEC.loader.exec_module(coreml_cache)


def test_expected_per_layer_state_names_default() -> None:
    assert coreml_cache.expected_per_layer_state_names() == (
        "context_k_l00",
        "context_v_l00",
        "context_k_l01",
        "context_v_l01",
        "context_k_l02",
        "context_v_l02",
        "context_k_l03",
        "context_v_l03",
        "context_k_l04",
        "context_v_l04",
        "context_k_l05",
        "context_v_l05",
        "context_k_l06",
        "context_v_l06",
        "context_k_l07",
        "context_v_l07",
        "context_k_l08",
        "context_v_l08",
        "context_k_l09",
        "context_v_l09",
        "context_k_l10",
        "context_v_l10",
        "context_k_l11",
        "context_v_l11",
        "valid_mask_state",
    )


def test_expected_per_layer_state_names_two_layers() -> None:
    assert coreml_cache.expected_per_layer_state_names(num_layers=2) == (
        "context_k_l00",
        "context_v_l00",
        "context_k_l01",
        "context_v_l01",
        "valid_mask_state",
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"sequence_length": 0, "text_len": 256, "speaker_context_len_bucket": 160},
            "sequence_length",
        ),
        (
            {"sequence_length": True, "text_len": 256, "speaker_context_len_bucket": 160},
            "sequence_length",
        ),
        (
            {"sequence_length": 100, "text_len": -1, "speaker_context_len_bucket": 160},
            "text_len",
        ),
        (
            {"sequence_length": 100, "text_len": True, "speaker_context_len_bucket": 160},
            "text_len",
        ),
        (
            {"sequence_length": 100, "text_len": 256, "speaker_context_len_bucket": 0},
            "speaker_context_len_bucket",
        ),
        (
            {"sequence_length": 100, "text_len": 256, "speaker_context_len_bucket": False},
            "speaker_context_len_bucket",
        ),
    ],
)
def test_coreml_condition_bucket_rejects_non_positive_or_bool_dimensions(
    kwargs: dict[str, int | bool],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        coreml_cache.CoreMLConditionBucket(**kwargs)


def test_per_layer_state_shapes_for_default_bucket() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )

    shapes = coreml_cache.per_layer_state_shapes(bucket)

    assert tuple(shapes) == coreml_cache.expected_per_layer_state_names()
    for state_name in coreml_cache.expected_per_layer_state_names()[:-1]:
        assert shapes[state_name] == (1, 416, 20, 64)
    assert shapes["valid_mask_state"] == (1, 416)


def test_per_layer_state_shapes_for_independent_text_speaker3_bucket() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )

    shapes = coreml_cache.per_layer_state_shapes(
        bucket,
        branch_layout=coreml_cache.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
    )

    assert tuple(shapes) == coreml_cache.expected_per_layer_state_names()
    for state_name in coreml_cache.expected_per_layer_state_names()[:-1]:
        assert shapes[state_name] == (3, 416, 20, 64)
    assert shapes["valid_mask_state"] == (3, 416)


def test_kv_bytes_per_branch_for_default_bucket() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )

    assert coreml_cache.kv_bytes_per_branch(bucket) == 25_559_040


def test_kv_memory_bytes_for_cond_and_independent_layouts() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )

    assert (
        coreml_cache.kv_memory_bytes(
            bucket,
            (
                coreml_cache.BRANCH_LAYOUT_COND1,
                coreml_cache.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
            ),
        )
        == 102_236_160
    )


def test_bucket_id_includes_dimensions_and_branch_layout() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )

    assert (
        bucket.bucket_id(coreml_cache.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3)
        == "S100_T256_R160_independent_text_speaker3"
    )


def test_estimate_cfg_active_steps_matches_rf_schedule() -> None:
    assert (
        coreml_cache.estimate_cfg_active_steps(
            num_steps=40,
            cfg_min_t=0.5,
            cfg_max_t=1.0,
        )
        == 20
    )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_steps": 0}, "num_steps"),
        ({"num_steps": 40, "cfg_min_t": -0.1}, "cfg_min_t"),
        ({"num_steps": 40, "cfg_max_t": 1.1}, "cfg_min_t"),
        ({"num_steps": 40, "cfg_min_t": 0.8, "cfg_max_t": 0.2}, "cfg_min_t"),
        ({"num_steps": 40, "cfg_min_t": float("nan")}, "cfg_min_t"),
    ],
)
def test_estimate_cfg_active_steps_rejects_invalid_inputs(
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        coreml_cache.estimate_cfg_active_steps(**kwargs)


def test_unknown_branch_layout_rejected() -> None:
    with pytest.raises(ValueError, match="unknown branch_layout"):
        coreml_cache.branch_count_for_layout("split")


def test_per_layer_state_shapes_rejects_unknown_branch_layout() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )

    with pytest.raises(ValueError, match="unknown branch_layout"):
        coreml_cache.per_layer_state_shapes(bucket, branch_layout="split")


@pytest.mark.parametrize(
    "branch_layout",
    [
        coreml_cache.BRANCH_LAYOUT_JOINT2,
        coreml_cache.BRANCH_LAYOUT_ALTERNATING_TEXT2,
        coreml_cache.BRANCH_LAYOUT_ALTERNATING_SPEAKER2,
    ],
)
def test_branch_count_for_two_branch_layouts(branch_layout: str) -> None:
    assert coreml_cache.branch_count_for_layout(branch_layout) == 2


def test_per_layer_state_shapes_for_joint2_layout() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )
    shapes = coreml_cache.per_layer_state_shapes(
        bucket,
        branch_layout=coreml_cache.BRANCH_LAYOUT_JOINT2,
    )
    for state_name in coreml_cache.expected_per_layer_state_names()[:-1]:
        assert shapes[state_name] == (2, 416, 20, 64)
    assert shapes["valid_mask_state"] == (2, 416)


def test_kv_memory_bytes_for_joint_and_alternating_layouts() -> None:
    bucket = coreml_cache.CoreMLConditionBucket(
        sequence_length=100,
        text_len=256,
        speaker_context_len_bucket=160,
    )
    one_branch = coreml_cache.kv_bytes_per_branch(bucket)
    assert (
        coreml_cache.kv_memory_bytes(
            bucket,
            (coreml_cache.BRANCH_LAYOUT_COND1, coreml_cache.BRANCH_LAYOUT_JOINT2),
        )
        == 3 * one_branch
    )
    assert (
        coreml_cache.kv_memory_bytes(
            bucket,
            (
                coreml_cache.BRANCH_LAYOUT_COND1,
                coreml_cache.BRANCH_LAYOUT_ALTERNATING_TEXT2,
                coreml_cache.BRANCH_LAYOUT_ALTERNATING_SPEAKER2,
            ),
        )
        == 5 * one_branch
    )


@pytest.mark.parametrize(
    "branch_layouts",
    [
        (coreml_cache.BRANCH_LAYOUT_COND1, coreml_cache.BRANCH_LAYOUT_JOINT2),
        (coreml_cache.BRANCH_LAYOUT_COND1, coreml_cache.BRANCH_LAYOUT_ALTERNATING_TEXT2),
        (coreml_cache.BRANCH_LAYOUT_COND1, coreml_cache.BRANCH_LAYOUT_ALTERNATING_SPEAKER2),
        (
            coreml_cache.BRANCH_LAYOUT_COND1,
            coreml_cache.BRANCH_LAYOUT_ALTERNATING_TEXT2,
            coreml_cache.BRANCH_LAYOUT_ALTERNATING_SPEAKER2,
        ),
    ],
)
def test_canonical_branch_layout_tuples_accepted(branch_layouts: tuple[str, ...]) -> None:
    assert branch_layouts in coreml_cache.ALLOWED_CONDITION_BRANCH_LAYOUTS
