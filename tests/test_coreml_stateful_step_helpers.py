from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import coreml_stateful_step_benchmark as bench  # noqa: E402


def test_compute_state_kv_bytes() -> None:
    assert (
        bench.compute_state_kv_bytes(
            num_layers=12,
            c_ctx_bucket=416,
            num_heads=20,
            head_dim=64,
        )
        == 25_559_040
    )


def test_pack_context_kv_state_preserves_text_speaker_order() -> None:
    text_k = torch.full((1, 2, 1, 1), 1.0)
    text_v = torch.full((1, 2, 1, 1), 2.0)
    speaker_k = torch.full((1, 3, 1, 1), 3.0)
    speaker_v = torch.full((1, 3, 1, 1), 4.0)

    packed = bench.pack_context_kv_state(
        [(text_k, text_v, speaker_k, speaker_v)],
        text_mask=torch.tensor([[True, True]]),
        speaker_mask=torch.tensor([[True, True, True]]),
        speaker_context_bucket=4,
    )

    assert packed.context_k_state.dtype == np.float16
    assert packed.context_v_state.dtype == np.float16
    assert packed.context_k_state.shape == (1, 1, 6, 1, 1)
    assert packed.context_v_state.shape == (1, 1, 6, 1, 1)
    assert packed.context_k_state[0, 0, :, 0, 0].tolist() == [1.0, 1.0, 3.0, 3.0, 3.0, 0.0]
    assert packed.context_v_state[0, 0, :, 0, 0].tolist() == [2.0, 2.0, 4.0, 4.0, 4.0, 0.0]
    assert packed.valid_mask_state.tolist() == [[1.0, 1.0, 1.0, 1.0, 1.0, 0.0]]


def test_pack_context_kv_state_rejects_speaker_overflow() -> None:
    text_k = torch.zeros((1, 2, 1, 1))
    text_v = torch.zeros((1, 2, 1, 1))
    speaker_k = torch.zeros((1, 3, 1, 1))
    speaker_v = torch.zeros((1, 3, 1, 1))

    with pytest.raises(ValueError, match="speaker_context_len 3 exceeds speaker_context_bucket 2"):
        bench.pack_context_kv_state(
            [(text_k, text_v, speaker_k, speaker_v)],
            text_mask=torch.tensor([[True, True]]),
            speaker_mask=torch.tensor([[True, True, True]]),
            speaker_context_bucket=2,
        )


def test_valid_mask_state_excludes_latent_mask() -> None:
    text_mask = torch.tensor([[True, False, True]])
    speaker_mask = torch.tensor([[True, True]])
    latent_mask = torch.tensor([[True, True, True, True]])

    mask = bench.build_valid_mask_state(
        text_mask=text_mask,
        speaker_mask=speaker_mask,
        speaker_context_bucket=2,
        latent_mask=latent_mask,
    )

    assert mask.shape == (1, 5)
    assert mask.tolist() == [[1.0, 0.0, 1.0, 1.0, 1.0]]


def test_valid_mask_state_pads_invalid_to_zero() -> None:
    mask = bench.build_valid_mask_state(
        text_mask=torch.tensor([[True, False]]),
        speaker_mask=torch.tensor([[True, False, True]]),
        speaker_context_bucket=5,
    )

    assert mask.dtype == np.float16
    assert mask.tolist() == [[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]


def test_normalize_compute_plan_counts_accepts_ios16_and_ios18() -> None:
    raw = {
        "ios16.linear": {"total": 2, "ne_preferred": 1, "preferred_devices": {"CPU": 1, "MLNeuralEngineComputeDevice": 1}},
        "ios18.matmul": {"total": 3, "ne_preferred": 2, "preferred_devices": {"MLNeuralEngineComputeDevice": 2, "GPU": 1}},
        "softmax": {"total": 4, "ne_preferred": 3, "preferred_devices": {"MLNeuralEngineComputeDevice": 3}},
        "ios18.add": {"total": 9, "ne_preferred": 0, "preferred_devices": {"CPU": 9}},
    }

    normalized = bench.normalize_compute_plan_counts(raw)

    assert normalized["linear"]["total"] == 2
    assert normalized["linear"]["ne_preferred"] == 1
    assert normalized["matmul"]["total"] == 3
    assert normalized["matmul"]["ne_preferred"] == 2
    assert normalized["softmax"]["total"] == 4
    assert normalized["softmax"]["ne_preferred"] == 3


def _passing_status_kwargs() -> dict:
    return {
        "conversion_ok": True,
        "write_state_ok": True,
        "rel_diff": 0.01,
        "max_rel_diff": 0.05,
        "normalized_compute_plan_counts": {
            "linear": {"ne_preferred": 1},
            "matmul": {"ne_preferred": 1},
            "softmax": {"ne_preferred": 1},
        },
        "require_ne_placement": True,
        "steady_predict_ms": 10.0,
        "first_predict_ms": 12.0,
        "write_state_ms": 10.0,
        "read_state_op_count": 3,
        "read_state_counts_by_state": {
            "context_k_state": 1,
            "context_v_state": 1,
            "valid_mask_state": 1,
        },
    }


def test_status_warns_when_steady_predict_slower_than_no_state() -> None:
    kwargs = _passing_status_kwargs()
    kwargs["steady_predict_ms"] = 14.5
    kwargs["first_predict_ms"] = 15.0
    status, reasons = bench.evaluate_stateful_status(**kwargs)

    assert status == "WARN"
    assert any("steady_predict_ms 14.5 is not faster than no-state baseline 14.428" in reason for reason in reasons)
    assert any("read_state_op_count=3" in reason for reason in reasons)


def test_status_warns_on_large_read_state_ops_even_when_speed_is_good() -> None:
    kwargs = _passing_status_kwargs()
    kwargs["large_read_state_ops"] = [
        {
            "name": "read_state_0",
            "state": "context_k_state",
            "shape": [12, 1, 416, 20, 64],
            "bytes_fp16": 12_779_520,
        }
    ]
    kwargs["slice_placement_summary"] = {
        "ios18.slice_by_index": {"total": 72, "ne_preferred": 72}
    }

    status, reasons = bench.evaluate_stateful_status(**kwargs)

    assert status == "WARN"
    assert any("packed state layout produced large read_state ops" in reason for reason in reasons)
    assert any("large_read_state_ops=1" in reason for reason in reasons)
    assert any("slice_placement=ios18.slice_by_index(total=72,ne=72)" in reason for reason in reasons)


def test_status_warns_on_read_state_error_when_other_gates_pass() -> None:
    kwargs = _passing_status_kwargs()
    kwargs["read_state_error"] = {"context_k_state": "read_state unsupported"}

    status, reasons = bench.evaluate_stateful_status(**kwargs)

    assert status == "WARN"
    assert any("read_state_error present after write_state" in reason for reason in reasons)


def test_status_warns_on_readback_diff_above_tolerance() -> None:
    kwargs = _passing_status_kwargs()
    kwargs["read_state_max_abs_diff"] = {
        "context_k_state": 0.0,
        "context_v_state": 0.002,
        "valid_mask_state": 0.0,
    }

    status, reasons = bench.evaluate_stateful_status(**kwargs)

    assert status == "WARN"
    assert any("read_state_max_abs_diff 0.002 exceeds tolerance 0.001" in reason for reason in reasons)


def test_status_fails_on_rel_diff_or_missing_ne_when_required() -> None:
    status, reasons = bench.evaluate_stateful_status(
        conversion_ok=True,
        write_state_ok=True,
        rel_diff=0.06,
        max_rel_diff=0.05,
        normalized_compute_plan_counts={
            "linear": {"ne_preferred": 1},
            "matmul": {"ne_preferred": 0},
            "softmax": {"ne_preferred": 1},
        },
        require_ne_placement=True,
        steady_predict_ms=10.0,
        first_predict_ms=10.0,
        write_state_ms=10.0,
        read_state_op_count=3,
    )

    assert status == "FAIL"
    assert "rel_diff 0.06 exceeds max_rel_diff 0.05" in reasons
    assert "matmul has no NE-preferred operations" in reasons


def test_compute_output_diff_metrics_rejects_shape_mismatch_and_non_finite() -> None:
    with pytest.raises(ValueError, match="output shape"):
        bench.compute_output_diff_metrics(
            np.zeros((1, 2), dtype=np.float32),
            np.zeros((1, 3), dtype=np.float32),
        )

    with pytest.raises(ValueError, match="Core ML output contains non-finite values"):
        bench.compute_output_diff_metrics(
            np.array([[np.nan]], dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
        )
