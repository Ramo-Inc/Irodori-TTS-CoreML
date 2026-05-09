from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "openai_api_server.py"
MODULE_SPEC = importlib.util.spec_from_file_location(
    "openai_api_server_auto_bucket",
    MODULE_PATH,
)
assert MODULE_SPEC is not None
openai_api_server = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = openai_api_server
MODULE_SPEC.loader.exec_module(openai_api_server)

from irodori_tts.inference_runtime import (  # noqa: E402
    InferenceRuntime,
    _patched_steps_for_seconds,
)


def planning_context(
    *,
    patched_steps: int,
    token_len: int = 12,
) -> openai_api_server.AutoSpeechPlanningContext:
    return openai_api_server.AutoSpeechPlanningContext(
        segment_text="segment",
        normalized_text="segment",
        seconds=1.0,
        sample_rate=24_000,
        hop_length=512,
        latent_patch_size=2,
        tokenizer_fingerprint="tokenizer:test",
        token_len=token_len,
        token_ids_hash="sha256:test",
        patched_steps=patched_steps,
    )


@pytest.mark.parametrize(
    ("patched_steps", "token_len", "expected_sequence_length", "expected_text_len"),
    [
        (1, 1, 256, 32),
        (256, 32, 256, 32),
        (257, 32, 512, 64),
        (512, 64, 512, 64),
        (513, 64, 1024, 128),
        (1024, 128, 1024, 128),
        (1025, 128, 1536, 192),
        (1536, 192, 1536, 192),
        (1537, 192, 2048, 256),
        (2048, 256, 2048, 256),
        (256, 33, 512, 64),
        (256, 65, 1024, 128),
        (256, 129, 1536, 192),
        (256, 193, 2048, 256),
    ],
)
def test_auto_select_bucket_picks_first_fitting_preset(
    patched_steps: int,
    token_len: int,
    expected_sequence_length: int,
    expected_text_len: int,
) -> None:
    resolution = openai_api_server._auto_select_bucket(
        planning_context(patched_steps=patched_steps, token_len=token_len),
    )

    assert resolution.reason is None
    assert resolution.attempted is None
    assert resolution.bucket is not None
    assert resolution.bucket.sequence_length == expected_sequence_length
    assert resolution.bucket.text_len == expected_text_len
    assert resolution.bucket.speaker_context_len_bucket == 160


def test_auto_bucket_oversize_t_returns_reason() -> None:
    resolution = openai_api_server._auto_select_bucket(
        planning_context(patched_steps=2048, token_len=257),
    )

    assert resolution.bucket is None
    assert resolution.reason == "oversize_t"
    assert resolution.attempted == "S2048_T>256_R160"


def test_auto_bucket_oversize_s_returns_reason() -> None:
    resolution = openai_api_server._auto_select_bucket(
        planning_context(patched_steps=2049, token_len=256),
    )

    assert resolution.bucket is None
    assert resolution.reason == "oversize_s"
    assert resolution.attempted == "S>2048_T256_R160"


def test_tokenize_for_bucket_uses_untruncated_encode_and_stable_hash() -> None:
    token_ids = list(range(300))

    class FakeTokenizer:
        def __init__(self) -> None:
            self.encode_calls: list[str] = []

        def encode(self, text: str) -> torch.Tensor:
            self.encode_calls.append(text)
            return torch.tensor(token_ids, dtype=torch.long)

        def batch_encode(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("tokenize_for_bucket must not call batch_encode")

    runtime = object.__new__(InferenceRuntime)
    runtime.tokenizer = FakeTokenizer()

    token_len, token_ids_hash = runtime.tokenize_for_bucket("normalized text")
    second_token_len, second_token_ids_hash = runtime.tokenize_for_bucket("normalized text")

    expected_hash = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()
    )
    assert token_len == 300
    assert second_token_len == 300
    assert token_ids_hash == expected_hash
    assert second_token_ids_hash == expected_hash
    assert runtime.tokenizer.encode_calls == ["normalized text", "normalized text"]


def test_estimate_patched_steps_matches_runtime_codec_and_patch_size() -> None:
    runtime = object.__new__(InferenceRuntime)
    runtime.codec = SimpleNamespace(
        sample_rate=24_000,
        model=SimpleNamespace(hop_length=512),
    )
    runtime.model_cfg = SimpleNamespace(latent_patch_size=2)

    assert runtime.estimate_patched_steps(5.0) == _patched_steps_for_seconds(
        5.0,
        sample_rate=24_000,
        hop_length=512,
        latent_patch_size=2,
    )


def _v2_fingerprint(**overrides: object) -> str:
    base: dict[str, object] = {
        "normalized_text": "hello world",
        "token_ids_hash": "sha256:abc",
        "tokenizer_fingerprint": "tokenizer:test",
        "model_fingerprint": "model:test",
        "reference_fingerprint": "server_default:/path:sha256:xyz",
        "caption": "neutral",
        "bucket": openai_api_server.CoreMLConditionBucket(
            sequence_length=256,
            text_len=32,
            speaker_context_len_bucket=160,
        ),
        "cfg": {
            "mode": "independent",
            "scale_text": 3.0,
            "scale_speaker": 5.0,
            "scale_caption": 0.0,
            "min_t": 0.5,
            "max_t": 1.0,
            "speaker_kv_scale": None,
            "speaker_kv_min_t": None,
            "speaker_kv_max_layers": None,
        },
        "branch_layouts": (
            openai_api_server.BRANCH_LAYOUT_COND1,
            openai_api_server.BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3,
        ),
        "speaker_context_len": 3,
        "state_copies": 1,
    }
    base.update(overrides)
    return openai_api_server._internal_auto_condition_fingerprint(**base)


def test_internal_auto_condition_fingerprint_excludes_raw_seconds() -> None:
    fp = _v2_fingerprint()
    assert fp.startswith("condition:auto-v2:")
    assert _v2_fingerprint() == fp


def test_internal_auto_condition_fingerprint_changes_with_text_caption_bucket_cfg_tokenizer_reference() -> None:
    base = _v2_fingerprint()
    assert _v2_fingerprint(normalized_text="hello world!!") != base
    assert _v2_fingerprint(token_ids_hash="sha256:zzz") != base
    assert _v2_fingerprint(tokenizer_fingerprint="tokenizer:other") != base
    assert _v2_fingerprint(model_fingerprint="model:other") != base
    assert _v2_fingerprint(reference_fingerprint="server_default:/other") != base
    assert _v2_fingerprint(caption="excited") != base
    assert (
        _v2_fingerprint(
            bucket=openai_api_server.CoreMLConditionBucket(
                sequence_length=512,
                text_len=64,
                speaker_context_len_bucket=160,
            ),
        )
        != base
    )
    assert (
        _v2_fingerprint(
            cfg={
                "mode": "independent",
                "scale_text": 4.0,
                "scale_speaker": 5.0,
                "scale_caption": 0.0,
                "min_t": 0.5,
                "max_t": 1.0,
                "speaker_kv_scale": None,
                "speaker_kv_min_t": None,
                "speaker_kv_max_layers": None,
            },
        )
        != base
    )


def test_explicit_condition_endpoint_canonical_fingerprint_unchanged() -> None:
    bucket = openai_api_server.CoreMLConditionBucket(
        sequence_length=256,
        text_len=32,
        speaker_context_len_bucket=160,
    )
    cfg = {
        "mode": "independent",
        "scale_text": 3.0,
        "scale_speaker": 5.0,
        "scale_caption": 0.0,
        "min_t": 0.5,
        "max_t": 1.0,
        "speaker_kv_scale": None,
        "speaker_kv_min_t": None,
        "speaker_kv_max_layers": None,
    }
    expected_payload = {
        "input": "input text",
        "caption": "neutral",
        "seconds": 2.0,
        "bucket": {
            "sequence_length": bucket.sequence_length,
            "text_len": bucket.text_len,
            "speaker_context_len": bucket.speaker_context_len_bucket,
        },
        "cfg": cfg,
    }
    expected = (
        "condition:"
        + hashlib.sha256(
            json.dumps(expected_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        ).hexdigest()
    )
    assert (
        openai_api_server._condition_fingerprint(
            {},
            bucket,
            cfg,
            "input text",
            "neutral",
            2.0,
        )
        == expected
    )


def test_explicit_irodori_bucket_bypasses_auto_select(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_auto_select(planning: object) -> None:
        del planning
        raise AssertionError("explicit bucket should bypass auto selection")

    monkeypatch.setattr(openai_api_server, "_auto_select_bucket", fail_auto_select)

    resolution = openai_api_server._resolve_auto_bucket_resolution(
        {
            "bucket": {
                "sequence_length": 512,
                "text_len": 128,
                "speaker_context_len": 160,
            },
        },
        openai_api_server.SpeechSegment(text="segment", seconds=5.0),
        runtime=object(),
    )

    assert resolution.reason is None
    assert resolution.bucket is not None
    assert resolution.bucket.sequence_length == 512
    assert resolution.bucket.text_len == 128
    assert resolution.bucket.speaker_context_len_bucket == 160
