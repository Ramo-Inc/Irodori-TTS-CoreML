from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MODULE_PATH = PROJECT_ROOT / "openai_api_server.py"
MODULE_SPEC = importlib.util.spec_from_file_location("openai_api_server", MODULE_PATH)
assert MODULE_SPEC is not None
openai_api_server = importlib.util.module_from_spec(MODULE_SPEC)
assert MODULE_SPEC.loader is not None
sys.modules[MODULE_SPEC.name] = openai_api_server
MODULE_SPEC.loader.exec_module(openai_api_server)

build_arg_parser = openai_api_server.build_arg_parser
build_settings_from_args = openai_api_server.build_settings_from_args
parse_args = openai_api_server.parse_args


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "IRODORI_AUTO_PREPARE_DEFAULT_CACHE",
        "IRODORI_STRICT_COREML",
        "IRODORI_MAX_RESIDENT_SPEAKER_KV_BUCKETS",
        "IRODORI_MAX_RESIDENT_CONDITION_CACHE_ENTRIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_cli_defaults_match_server_settings_defaults() -> None:
    args = parse_args([])
    settings = build_settings_from_args(args)
    assert settings.auto_prepare_default_cache is True
    assert settings.strict_coreml is False
    assert settings.default_reference_cache_prepare is True
    assert settings.default_condition_cache_prepare_text is None
    assert settings.condition_cache_default_ttl_seconds == pytest.approx(86400.0)
    assert settings.max_resident_speaker_kv_buckets == 3
    assert settings.max_resident_condition_cache_entries == 0
    assert settings.enable_resident_reference_cache is True
    assert settings.enable_resident_speaker_kv is True
    assert settings.enable_condition_packed_kv_cache is False
    assert settings.max_seconds == pytest.approx(70.0)
    assert settings.chars_per_second == pytest.approx(5.5)
    assert settings.seconds_padding == pytest.approx(1.5)
    assert settings.default_speed == pytest.approx(1.2)


def test_cli_default_speed_accepts_explicit_value() -> None:
    args = parse_args(["--default-speed", "1.5"])
    settings = build_settings_from_args(args)
    assert settings.default_speed == pytest.approx(1.5)


def test_cli_default_speed_below_minimum_rejected() -> None:
    args = parse_args(["--default-speed", "0.1"])
    with pytest.raises(ValueError, match="--default-speed"):
        build_settings_from_args(args)


def test_cli_default_speed_above_maximum_rejected() -> None:
    args = parse_args(["--default-speed", "5.0"])
    with pytest.raises(ValueError, match="--default-speed"):
        build_settings_from_args(args)


def test_cli_max_seconds_safe_cap_matches_default() -> None:
    assert openai_api_server.MAX_SAFE_SEGMENT_SECONDS == pytest.approx(70.0)
    args = parse_args([])
    settings = build_settings_from_args(args)
    assert openai_api_server._effective_segment_max_seconds(settings) == pytest.approx(70.0)


def test_cli_no_auto_prepare_default_cache_flag_disables_setting() -> None:
    args = parse_args(["--no-auto-prepare-default-cache"])
    settings = build_settings_from_args(args)
    assert settings.auto_prepare_default_cache is False


def test_cli_strict_coreml_flag_enables_setting() -> None:
    args = parse_args(["--strict-coreml"])
    settings = build_settings_from_args(args)
    assert settings.strict_coreml is True


def test_cli_no_enable_resident_speaker_kv_disables_setting() -> None:
    args = parse_args(["--no-enable-resident-speaker-kv"])
    settings = build_settings_from_args(args)
    assert settings.enable_resident_speaker_kv is False


def test_cli_no_enable_resident_reference_cache_disables_setting() -> None:
    args = parse_args(["--no-enable-resident-reference-cache"])
    settings = build_settings_from_args(args)
    assert settings.enable_resident_reference_cache is False


def test_cli_max_resident_speaker_kv_buckets_accepts_explicit_value() -> None:
    args = parse_args(["--max-resident-speaker-kv-buckets", "5"])
    settings = build_settings_from_args(args)
    assert settings.max_resident_speaker_kv_buckets == 5


def test_cli_negative_max_resident_speaker_kv_buckets_rejected() -> None:
    args = parse_args(["--max-resident-speaker-kv-buckets", "-1"])
    with pytest.raises(ValueError, match="--max-resident-speaker-kv-buckets"):
        build_settings_from_args(args)


def test_cli_negative_max_resident_condition_cache_entries_rejected() -> None:
    args = parse_args(["--max-resident-condition-cache-entries", "-2"])
    with pytest.raises(ValueError, match="--max-resident-condition-cache-entries"):
        build_settings_from_args(args)


def test_cli_zero_or_negative_ttl_disables_ttl() -> None:
    args = parse_args(["--condition-cache-default-ttl-seconds", "0"])
    settings = build_settings_from_args(args)
    assert settings.condition_cache_default_ttl_seconds is None

    args = parse_args(["--condition-cache-default-ttl-seconds", "-1"])
    settings = build_settings_from_args(args)
    assert settings.condition_cache_default_ttl_seconds is None


def test_cli_positive_ttl_kept() -> None:
    args = parse_args(["--condition-cache-default-ttl-seconds", "120"])
    settings = build_settings_from_args(args)
    assert settings.condition_cache_default_ttl_seconds == pytest.approx(120.0)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("true", True),
        ("1", True),
        ("yes", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_env_auto_prepare_default_cache_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
    env_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("IRODORI_AUTO_PREPARE_DEFAULT_CACHE", env_value)
    args = parse_args([])
    settings = build_settings_from_args(args)
    assert settings.auto_prepare_default_cache is expected


def test_env_strict_coreml_overrides_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRODORI_STRICT_COREML", "true")
    args = parse_args([])
    settings = build_settings_from_args(args)
    assert settings.strict_coreml is True


def test_env_max_resident_speaker_kv_buckets_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRODORI_MAX_RESIDENT_SPEAKER_KV_BUCKETS", "7")
    args = parse_args([])
    settings = build_settings_from_args(args)
    assert settings.max_resident_speaker_kv_buckets == 7


def test_env_max_resident_condition_cache_entries_overrides_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IRODORI_MAX_RESIDENT_CONDITION_CACHE_ENTRIES", "12")
    args = parse_args([])
    settings = build_settings_from_args(args)
    assert settings.max_resident_condition_cache_entries == 12


def test_explicit_flag_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRODORI_AUTO_PREPARE_DEFAULT_CACHE", "true")
    args = parse_args(["--no-auto-prepare-default-cache"])
    settings = build_settings_from_args(args)
    assert settings.auto_prepare_default_cache is False


def test_invalid_env_bool_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRODORI_STRICT_COREML", "maybe")
    with pytest.raises(ValueError, match="IRODORI_STRICT_COREML"):
        parse_args([])


def test_arg_parser_has_boolean_optional_action_pairs() -> None:
    parser = build_arg_parser()
    options = {
        action.option_strings[0]: action.option_strings
        for action in parser._actions
        if action.option_strings
    }
    for flag in (
        "--auto-prepare-default-cache",
        "--strict-coreml",
        "--enable-resident-speaker-kv",
        "--enable-resident-reference-cache",
        "--enable-condition-packed-kv-cache",
        "--default-reference-cache-prepare",
    ):
        assert flag in options
        assert any(opt.startswith("--no-") for opt in options[flag]), (
            f"{flag} should expose --no- variant via BooleanOptionalAction"
        )
