from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from irodori_tts.config import ModelConfig  # noqa: E402
from irodori_tts.model import (  # noqa: E402
    TextToLatentRFDiT,
    apply_rotary_emb,
    precompute_freqs_cis,
)
from tools import coreml_real_step_benchmark as bench  # noqa: E402


class _FakePrecision:
    FLOAT16 = "fake-float16"
    FLOAT32 = "fake-float32"


class _FakeCoreMLTools:
    precision = _FakePrecision


def test_resolve_compute_precision_maps_cli_values() -> None:
    assert bench.resolve_compute_precision(_FakeCoreMLTools, "default") == (None, "default")
    assert bench.resolve_compute_precision(_FakeCoreMLTools, "float16") == (
        "fake-float16",
        "float16",
    )
    assert bench.resolve_compute_precision(_FakeCoreMLTools, "float32") == (
        "fake-float32",
        "float32",
    )


def test_evaluate_benchmark_status_checks_rel_diff_and_ne_placement() -> None:
    passing_counts = {
        "ios16.linear": {"ne_preferred": 1},
        "ios16.matmul": {"ne_preferred": 2},
        "ios16.softmax": {"ne_preferred": 3},
    }
    status, reasons = bench.evaluate_benchmark_status(
        ne_preferred_counts=passing_counts,
        rel_diff=0.01,
        max_rel_diff=0.05,
        require_ne_placement=True,
    )
    assert status == "PASS"
    assert reasons == []

    missing_counts = {
        "ios16.linear": {"ne_preferred": 1},
        "ios16.matmul": {"ne_preferred": 0},
        "ios16.softmax": {"ne_preferred": 3},
    }
    status, reasons = bench.evaluate_benchmark_status(
        ne_preferred_counts=missing_counts,
        rel_diff=0.01,
        max_rel_diff=0.05,
        require_ne_placement=True,
    )
    assert status == "FAIL"
    assert any("ios16.matmul" in reason for reason in reasons)

    status, reasons = bench.evaluate_benchmark_status(
        ne_preferred_counts=missing_counts,
        rel_diff=0.06,
        max_rel_diff=0.05,
        require_ne_placement=False,
    )
    assert status == "FAIL"
    assert reasons == ["rel_diff 0.06 exceeds max_rel_diff 0.05"]


def test_apply_real_rope_matches_complex_rope() -> None:
    torch.manual_seed(20260508)
    x = torch.randn(2, 5, 3, 4, dtype=torch.float32)
    rope_cos, rope_sin = bench.build_rope_cache(sequence_length=5, head_dim=4)

    actual = bench.apply_real_rope(x, rope_cos, rope_sin)
    expected = apply_rotary_emb(x, precompute_freqs_cis(4, 5))

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_explicit_attention_matches_sdpa_with_bool_mask() -> None:
    torch.manual_seed(20260508)
    q = torch.randn(1, 2, 3, 4, dtype=torch.float32)
    k = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    v = torch.randn(1, 2, 5, 4, dtype=torch.float32)
    key_mask = torch.tensor([[True, True, False, True, False]])
    key_mask_f = key_mask.to(dtype=torch.float32)

    actual = bench.explicit_scaled_dot_product_attention(q, k, v, key_mask_f)
    expected = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=key_mask[:, None, None, :],
        is_causal=False,
    )

    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_real_coreml_denoiser_wrapper_matches_tiny_original_step() -> None:
    torch.manual_seed(20260508)
    cfg = ModelConfig(
        latent_dim=4,
        latent_patch_size=1,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=1.0,
        text_mlp_ratio=1.0,
        speaker_mlp_ratio=1.0,
        text_vocab_size=16,
        text_dim=8,
        text_layers=1,
        text_heads=2,
        use_caption_condition=False,
        speaker_dim=8,
        speaker_layers=1,
        speaker_heads=2,
        timestep_embed_dim=8,
        adaln_rank=4,
        norm_eps=1e-5,
    )
    model = TextToLatentRFDiT(cfg).eval()
    with torch.no_grad():
        model.out_proj.weight.normal_(mean=0.0, std=0.02)
        if model.out_proj.bias is not None:
            model.out_proj.bias.zero_()

    x_t = torch.randn(1, 4, cfg.patched_latent_dim)
    t = torch.tensor([0.999], dtype=torch.float32)
    text_state = torch.randn(1, 3, cfg.text_dim)
    text_mask = torch.tensor([[True, True, False]])
    speaker_state = torch.randn(1, 2, cfg.speaker_dim)
    speaker_mask = torch.tensor([[True, True]])
    latent_mask = torch.tensor([[True, True, True, True]])

    wrapper = bench.RealCoreMLDenoiserStep(model, sequence_length=4).eval()
    with torch.inference_mode():
        expected = model.forward_with_encoded_conditions(
            x_t=x_t,
            t=t,
            text_state=text_state,
            text_mask=text_mask,
            speaker_state=speaker_state,
            speaker_mask=speaker_mask,
            latent_mask=latent_mask,
        )
        actual = wrapper(
            x_t,
            t,
            text_state,
            text_mask.to(dtype=torch.float32),
            speaker_state,
            speaker_mask.to(dtype=torch.float32),
            latent_mask.to(dtype=torch.float32),
        )

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
