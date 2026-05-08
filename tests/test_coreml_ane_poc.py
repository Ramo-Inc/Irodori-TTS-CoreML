from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

ct = pytest.importorskip("coremltools")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import coreml_ane_poc as poc  # noqa: E402


def test_probe_a_current_style_complex_rope_fails_conversion() -> None:
    result = poc.probe_complex_rope_conversion_failure(ct)
    assert result.passed, result.detail
    assert "view_as_complex" in result.detail


def test_probe_b_current_tiny_forward_fails_conversion() -> None:
    result = poc.probe_current_tiny_forward_not_coreml_ready(ct)
    assert result.passed, result.detail


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML prediction requires macOS")
def test_probe_c_coreml_rewrite_executes_with_all_compute_units() -> None:
    result = poc.probe_coreml_compatible_denoiser(ct, ct.ComputeUnit.ALL, iterations=3)
    assert result.passed, result.detail
    assert result.max_abs_diff is not None
    assert result.max_abs_diff < poc.DIFF_TOL


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML prediction requires macOS")
def test_probe_c_coreml_rewrite_attempts_cpu_and_ne() -> None:
    cpu_and_ne = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
    if cpu_and_ne is None:
        pytest.skip("coremltools does not expose ComputeUnit.CPU_AND_NE")

    result = poc.probe_coreml_compatible_denoiser(ct, cpu_and_ne, iterations=3)
    if result.status == "SKIP":
        pytest.skip(result.detail)

    assert result.passed, result.detail
    assert result.max_abs_diff is not None
    assert result.max_abs_diff < poc.DIFF_TOL


@pytest.mark.skipif(platform.system() != "Darwin", reason="MLComputePlan requires macOS")
def test_probe_d_irodori_like_attention_has_ane_preferred_ops() -> None:
    result = poc.probe_ane_compute_plan_placement(ct)
    if result.status == "SKIP":
        pytest.skip(result.detail)

    assert result.passed, result.detail
    assert "ios16.linear" in result.detail
    assert "ios16.matmul" in result.detail or "ios16.softmax" in result.detail
    assert "MLNeuralEngineComputeDevice" in result.detail
