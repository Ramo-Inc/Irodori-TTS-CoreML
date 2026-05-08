#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import math
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Response
from huggingface_hub import hf_hub_download

from irodori_tts.inference_runtime import (
    RuntimeKey,
    SamplingRequest,
    clear_cached_runtime,
    default_runtime_device,
    get_cached_runtime,
    list_available_runtime_devices,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = "Aratako/Irodori-TTS-500M-v2"
DEFAULT_CODEC_REPO = "Aratako/Semantic-DACVAE-Japanese-32dim"
DEFAULT_API_MODEL_ID = "irodori-tts-500m-v2"
MODEL_CREATED = 1700000000
SUPPORTED_RESPONSE_FORMATS = {"mp3", "opus", "aac", "flac", "wav", "pcm"}
SECONDS_ROUND_INCREMENT = 0.5
MAX_SAFE_SEGMENT_SECONDS = 30.0


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    checkpoint: str
    reference_wav: Path
    api_model_id: str
    model_device: str
    codec_device: str
    model_precision: str
    codec_precision: str
    codec_repo: str
    default_num_steps: int
    max_num_steps: int
    seconds: float | None
    min_seconds: float
    max_seconds: float
    chars_per_second: float
    seconds_padding: float
    max_ref_seconds: float | None
    preload: bool
    log_timings: bool


@dataclass(frozen=True)
class SpeechSegment:
    text: str
    seconds: float


@dataclass(frozen=True)
class SpeechSegmentPlan:
    segments: tuple[SpeechSegment, ...]
    total_seconds: float
    seconds_mode: str


class RuntimeState:
    def __init__(self, settings: ServerSettings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._runtime_key: RuntimeKey | None = None

    def runtime_key(self) -> RuntimeKey:
        with self._lock:
            if self._runtime_key is None:
                self._runtime_key = RuntimeKey(
                    checkpoint=_resolve_checkpoint_path(self.settings.checkpoint),
                    model_device=self.settings.model_device,
                    codec_repo=self.settings.codec_repo,
                    model_precision=self.settings.model_precision,
                    codec_device=self.settings.codec_device,
                    codec_precision=self.settings.codec_precision,
                    codec_deterministic_encode=True,
                    codec_deterministic_decode=True,
                    enable_watermark=False,
                    compile_model=False,
                    compile_dynamic=False,
                )
            return self._runtime_key

    def get_runtime(self):
        runtime, _ = get_cached_runtime(self.runtime_key())
        return runtime

    @property
    def runtime_loaded(self) -> bool:
        return self._runtime_key is not None


def _prefer_mps_device() -> str:
    devices = list_available_runtime_devices()
    if "mps" in devices:
        return "mps"
    return default_runtime_device()


def _project_relative_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _validate_reference_wav(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Configured reference_wav is not a file: {path}")


def _resolve_checkpoint_path(raw_checkpoint: str) -> str:
    checkpoint = str(raw_checkpoint).strip()
    if checkpoint == "":
        raise ValueError("checkpoint must be non-empty.")

    candidate = Path(checkpoint).expanduser()
    local_candidates = [candidate]
    if not candidate.is_absolute():
        local_candidates.append(PROJECT_ROOT / candidate)
    for local_path in local_candidates:
        if local_path.is_file():
            return str(local_path)

    suffix = candidate.suffix.lower()
    if suffix in {".pt", ".safetensors"}:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    return str(hf_hub_download(repo_id=checkpoint, filename="model.safetensors"))


def _require_object(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    return payload


def _required_text(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a string.")
    text = value.strip()
    if text == "":
        raise HTTPException(status_code=400, detail=f"'{field}' must be non-empty.")
    return text


def _optional_text(payload: dict[str, Any], field: str, default: str | None = None) -> str | None:
    value = payload.get(field, default)
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a string.")
    return value


def _resolve_caption(payload: dict[str, Any]) -> str | None:
    """Resolve the style/control caption for SamplingRequest.

    Accepts the OpenAI-compatible ``instructions`` field, the singular alias
    ``instruction``, and the original Irodori ``caption`` extension. Whitespace-only
    values are treated as unspecified. When multiple non-empty values are provided
    they must be equal; otherwise HTTP 400 is raised.
    """

    def _normalized(field: str) -> str | None:
        raw = _optional_text(payload, field, None)
        if raw is None:
            return None
        stripped = raw.strip()
        return stripped or None

    instructions = _normalized("instructions")
    instruction = _normalized("instruction")
    caption = _normalized("caption")

    if (
        instructions is not None
        and instruction is not None
        and instructions != instruction
    ):
        raise HTTPException(
            status_code=400,
            detail="'instructions' and 'instruction' must match when both are provided.",
        )
    resolved_instruction = instructions if instructions is not None else instruction

    if (
        caption is not None
        and resolved_instruction is not None
        and caption != resolved_instruction
    ):
        raise HTTPException(
            status_code=400,
            detail="'caption' must match 'instructions' when both are provided.",
        )

    return resolved_instruction if resolved_instruction is not None else caption


def _optional_int(
    payload: dict[str, Any],
    field: str,
    default: int | None = None,
    *,
    min_value: int | None = None,
) -> int | None:
    value = payload.get(field, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"'{field}' must be an integer.")
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"'{field}' must be an integer.") from exc
    if min_value is not None and out < min_value:
        raise HTTPException(
            status_code=400,
            detail=f"'{field}' must be >= {min_value}.",
        )
    return out


def _optional_float(
    payload: dict[str, Any],
    field: str,
    default: float | None = None,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    value = payload.get(field, default)
    if value is None:
        return None
    if isinstance(value, bool):
        raise HTTPException(status_code=400, detail=f"'{field}' must be a number.")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"'{field}' must be a number.") from exc
    if not math.isfinite(out):
        raise HTTPException(status_code=400, detail=f"'{field}' must be finite.")
    if min_value is not None and out < min_value:
        raise HTTPException(status_code=400, detail=f"'{field}' must be >= {min_value}.")
    if max_value is not None and out > max_value:
        raise HTTPException(status_code=400, detail=f"'{field}' must be <= {max_value}.")
    return out


def _normalize_response_format(raw_format: str | None) -> tuple[str, str]:
    requested = (raw_format or "mp3").strip().lower()
    if requested not in SUPPORTED_RESPONSE_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_RESPONSE_FORMATS))
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format={requested!r}. Supported: {supported}.",
        )
    if requested in {"aac", "opus"}:
        return requested, "mp3"
    return requested, requested


def _format_seconds_header(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _count_non_whitespace_chars(text: str) -> int:
    return sum(1 for char in text if not char.isspace())


def _estimate_generation_seconds_uncapped(text: str, settings: ServerSettings) -> float:
    char_count = _count_non_whitespace_chars(text)
    raw_seconds = (float(char_count) / float(settings.chars_per_second)) + float(
        settings.seconds_padding
    )
    return math.ceil(raw_seconds / SECONDS_ROUND_INCREMENT) * SECONDS_ROUND_INCREMENT


def _effective_segment_max_seconds(settings: ServerSettings) -> float:
    return min(float(settings.max_seconds), MAX_SAFE_SEGMENT_SECONDS)


def _estimate_generation_seconds(text: str, settings: ServerSettings) -> float:
    rounded = _estimate_generation_seconds_uncapped(text, settings)
    return min(max(rounded, float(settings.min_seconds)), _effective_segment_max_seconds(settings))


def _speech_chunk_char_budget(settings: ServerSettings) -> int:
    segment_max_seconds = _effective_segment_max_seconds(settings)
    usable_seconds = max(0.0, segment_max_seconds - float(settings.seconds_padding))
    budget = int(math.floor(usable_seconds * float(settings.chars_per_second) * 0.9))
    return max(20, budget)


def _split_at_boundaries(text: str, boundaries: set[str]) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in boundaries:
            part = "".join(buffer)
            if part.strip():
                parts.append(part)
            buffer = []
    part = "".join(buffer)
    if part.strip():
        parts.append(part)
    return parts


def _split_at_secondary_boundaries(text: str) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    for char in text:
        buffer.append(char)
        if char in {"、", "，", ",", "；", ";", "：", ":"} or char.isspace():
            part = "".join(buffer)
            if part.strip():
                parts.append(part)
            buffer = []
    part = "".join(buffer)
    if part.strip():
        parts.append(part)
    return parts


def _hard_split_by_char_budget(text: str, char_budget: int) -> list[str]:
    parts: list[str] = []
    buffer: list[str] = []
    char_count = 0
    for char in text:
        char_weight = 0 if char.isspace() else 1
        if buffer and char_count + char_weight > char_budget:
            part = "".join(buffer).strip()
            if part:
                parts.append(part)
            buffer = []
            char_count = 0
        buffer.append(char)
        char_count += char_weight
    part = "".join(buffer).strip()
    if part:
        parts.append(part)
    return parts


def _pack_text_chunks(parts: list[str], char_budget: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for part in parts:
        if not part.strip():
            continue
        candidate = f"{current}{part}" if current else part
        if current and _count_non_whitespace_chars(candidate) > char_budget:
            chunk = current.strip()
            if chunk:
                chunks.append(chunk)
            current = part
        else:
            current = candidate

    chunk = current.strip()
    if chunk:
        chunks.append(chunk)
    return chunks


def _split_text_for_auto_chunks(text: str, settings: ServerSettings) -> list[str]:
    char_budget = _speech_chunk_char_budget(settings)
    sentence_parts = _split_at_boundaries(text, {"。", "！", "？", "!", "?", "\n", "\r"})
    small_parts: list[str] = []

    for sentence in sentence_parts:
        if _count_non_whitespace_chars(sentence) <= char_budget:
            small_parts.append(sentence)
            continue

        for part in _split_at_secondary_boundaries(sentence):
            if _count_non_whitespace_chars(part) <= char_budget:
                small_parts.append(part)
            else:
                small_parts.extend(_hard_split_by_char_budget(part, char_budget))

    return _pack_text_chunks(small_parts, char_budget)


def _build_speech_segment_plan(
    payload: dict[str, Any],
    text: str,
    settings: ServerSettings,
) -> SpeechSegmentPlan:
    segment_max_seconds = _effective_segment_max_seconds(settings)
    request_seconds = _optional_float(
        payload,
        "seconds",
        None,
        min_value=0.1,
        max_value=segment_max_seconds,
    )
    if request_seconds is not None:
        seconds = float(request_seconds)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "request")

    if settings.seconds is not None:
        seconds = min(float(settings.seconds), segment_max_seconds)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "fixed")

    uncapped_seconds = _estimate_generation_seconds_uncapped(text, settings)
    if uncapped_seconds <= segment_max_seconds:
        seconds = _estimate_generation_seconds(text, settings)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "auto")

    chunks = _split_text_for_auto_chunks(text, settings)
    segments = tuple(
        SpeechSegment(text=chunk, seconds=_estimate_generation_seconds(chunk, settings))
        for chunk in chunks
    )
    if not segments:
        seconds = _estimate_generation_seconds(text, settings)
        return SpeechSegmentPlan((SpeechSegment(text=text, seconds=seconds),), seconds, "auto")

    seconds_mode = "auto-chunked" if len(segments) > 1 else "auto"
    total_seconds = sum(segment.seconds for segment in segments)
    return SpeechSegmentPlan(segments, total_seconds, seconds_mode)


def _resolve_generation_seconds(
    payload: dict[str, Any],
    text: str,
    settings: ServerSettings,
) -> tuple[float, str]:
    segment_max_seconds = _effective_segment_max_seconds(settings)
    request_seconds = _optional_float(
        payload,
        "seconds",
        None,
        min_value=0.1,
        max_value=segment_max_seconds,
    )
    if request_seconds is not None:
        return float(request_seconds), "request"
    if settings.seconds is not None:
        return min(float(settings.seconds), segment_max_seconds), "fixed"
    return _estimate_generation_seconds(text, settings), "auto"


def _soundfile_format(format_name: str) -> tuple[str, str | None]:
    if format_name == "wav":
        return "WAV", "PCM_16"
    if format_name == "mp3":
        return "MP3", "MPEG_LAYER_III"
    if format_name == "flac":
        return "FLAC", "PCM_16"
    raise ValueError(f"soundfile format not available for {format_name!r}.")


def _content_type(format_name: str) -> str:
    return {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "flac": "audio/flac",
        "pcm": "audio/L16",
    }[format_name]


def _audio_array(audio):
    tensor = audio.detach().to("cpu").float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected audio tensor with shape (channels, samples), got {tuple(tensor.shape)}")
    return tensor.clamp(-1.0, 1.0).transpose(0, 1).contiguous().numpy()


def _serialize_audio(audio, sample_rate: int, format_name: str) -> bytes:
    array = _audio_array(audio)
    if format_name == "pcm":
        return (array * 32767.0).astype("<i2", copy=False).tobytes()

    import soundfile as sf

    sf_format, subtype = _soundfile_format(format_name)
    if sf_format not in sf.available_formats():
        raise HTTPException(
            status_code=400,
            detail=f"response_format={format_name!r} is not supported by this soundfile build.",
        )
    if subtype is not None and subtype not in sf.available_subtypes(sf_format):
        raise HTTPException(
            status_code=400,
            detail=(
                f"response_format={format_name!r} subtype {subtype!r} is not supported "
                "by this soundfile build."
            ),
        )

    buffer = BytesIO()
    sf.write(buffer, array, int(sample_rate), format=sf_format, subtype=subtype)
    return buffer.getvalue()


def _normalize_audio_segment(audio):
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)
    if audio.ndim != 2:
        raise ValueError(f"Expected audio tensor with shape (channels, samples), got {tuple(audio.shape)}")
    return audio


def _concatenate_audio_segments(audio_segments: list[Any]):
    if not audio_segments:
        raise ValueError("No audio segments were generated.")
    if len(audio_segments) == 1:
        return audio_segments[0]

    channel_count = int(audio_segments[0].shape[0])
    total_samples = sum(int(audio.shape[1]) for audio in audio_segments)
    combined = audio_segments[0].new_empty((channel_count, total_samples))
    offset = 0
    for audio in audio_segments:
        samples = int(audio.shape[1])
        combined[:, offset : offset + samples] = audio
        offset += samples
    return combined


def _health_payload(settings: ServerSettings, state: RuntimeState) -> dict[str, Any]:
    return {
        "status": "ok",
        "object": "health",
        "model": settings.api_model_id,
        "reference_wav": str(settings.reference_wav.relative_to(PROJECT_ROOT))
        if settings.reference_wav.is_relative_to(PROJECT_ROOT)
        else str(settings.reference_wav),
        "reference_wav_exists": settings.reference_wav.is_file(),
        "model_device": settings.model_device,
        "codec_device": settings.codec_device,
        "runtime_loaded": state.runtime_loaded,
    }


def create_app(settings: ServerSettings) -> FastAPI:
    state = RuntimeState(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _validate_reference_wav(settings.reference_wav)
        if settings.preload:
            await asyncio.to_thread(state.get_runtime)
        try:
            yield
        finally:
            clear_cached_runtime()

    app = FastAPI(title="Irodori-TTS OpenAI-compatible API", version="0.1.0", lifespan=lifespan)

    @app.api_route("/v1/health", methods=["GET", "POST"])
    async def health() -> dict[str, Any]:
        return _health_payload(settings, state)

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.api_model_id,
                    "object": "model",
                    "created": MODEL_CREATED,
                    "owned_by": "irodori-tts",
                }
            ],
        }

    @app.post("/v1/audio/speech")
    async def audio_speech(payload: Any = Body(...)) -> Response:
        data = _require_object(payload)
        text = _required_text(data, "input")
        # Compatibility-only fields such as model, voice, and reference_audio are
        # deliberately ignored. Runtime selection and speaker reference are server-owned.
        _optional_float(data, "speed", 1.0, min_value=0.25, max_value=4.0)
        requested_format, output_format = _normalize_response_format(
            _optional_text(data, "response_format", "mp3")
        )
        num_steps = _optional_int(
            data,
            "num_steps",
            settings.default_num_steps,
            min_value=1,
        )
        if int(num_steps) > settings.max_num_steps:
            raise HTTPException(
                status_code=400,
                detail=f"'num_steps' must be <= {settings.max_num_steps}.",
            )
        seed = _optional_int(data, "seed", None)
        caption = _resolve_caption(data)
        segment_plan = _build_speech_segment_plan(data, text, settings)

        if not settings.reference_wav.is_file():
            raise HTTPException(
                status_code=500,
                detail=f"Configured reference_wav not found: {settings.reference_wav}",
            )

        try:
            runtime = await asyncio.to_thread(state.get_runtime)
            audio_segments: list[Any] = []
            sample_rate: int | None = None
            channel_count: int | None = None
            for segment_index, segment in enumerate(segment_plan.segments):
                segment_seed = None if seed is None else int(seed) + segment_index
                result = await asyncio.to_thread(
                    runtime.synthesize,
                    SamplingRequest(
                        text=segment.text,
                        caption=caption,
                        ref_wav=str(settings.reference_wav),
                        ref_latent=None,
                        no_ref=False,
                        num_steps=int(num_steps),
                        seconds=float(segment.seconds),
                        max_ref_seconds=settings.max_ref_seconds,
                        seed=segment_seed,
                    ),
                    log_fn=print if settings.log_timings else None,
                )
                audio = _normalize_audio_segment(result.audio)
                result_sample_rate = int(result.sample_rate)
                result_channel_count = int(audio.shape[0])
                if sample_rate is None:
                    sample_rate = result_sample_rate
                    channel_count = result_channel_count
                elif result_sample_rate != sample_rate:
                    raise ValueError("Generated audio sample rates did not match across chunks.")
                elif result_channel_count != channel_count:
                    raise ValueError("Generated audio channel counts did not match across chunks.")
                audio_segments.append(audio)

            if sample_rate is None:
                raise ValueError("No audio segments were generated.")
            audio = _concatenate_audio_segments(audio_segments)
            audio_bytes = _serialize_audio(audio, sample_rate, output_format)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        headers = {
            "Content-Disposition": f'attachment; filename="speech.{output_format}"',
            "X-Irodori-Requested-Format": requested_format,
            "X-Irodori-Generation-Seconds": _format_seconds_header(segment_plan.total_seconds),
            "X-Irodori-Seconds-Mode": segment_plan.seconds_mode,
            "X-Irodori-Chunk-Count": str(len(segment_plan.segments)),
            "X-Irodori-Chunk-Seconds": ",".join(
                _format_seconds_header(segment.seconds) for segment in segment_plan.segments
            ),
            "X-Irodori-Num-Steps": str(int(num_steps)),
        }
        return Response(
            content=audio_bytes,
            media_type=_content_type(output_format),
            headers=headers,
        )

    return app


def parse_args() -> argparse.Namespace:
    default_device = _prefer_mps_device()
    parser = argparse.ArgumentParser(
        description="OpenAI-compatible HTTP TTS API for Irodori-TTS."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Local checkpoint file or Hugging Face repo id.",
    )
    parser.add_argument(
        "--reference-wav",
        default="rem.wav",
        help="Server-forced speaker reference wav. Client voice/reference fields are ignored.",
    )
    parser.add_argument("--api-model-id", default=DEFAULT_API_MODEL_ID)
    parser.add_argument("--model-device", default=default_device)
    parser.add_argument("--codec-device", default=default_device)
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--codec-repo", default=DEFAULT_CODEC_REPO)
    parser.add_argument("--default-num-steps", type=int, default=40)
    parser.add_argument("--max-num-steps", type=int, default=80)
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Fixed generation horizon for every request. If omitted, seconds are estimated from input length.",
    )
    parser.add_argument("--min-seconds", type=float, default=4.0)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--chars-per-second", type=float, default=4.0)
    parser.add_argument("--seconds-padding", type=float, default=1.5)
    parser.add_argument(
        "--max-ref-seconds",
        type=float,
        default=30.0,
        help="Maximum server reference duration. Set <=0 to disable the cap.",
    )
    parser.add_argument("--preload", action="store_true", help="Load the runtime during startup.")
    parser.add_argument(
        "--log-timings",
        action="store_true",
        help="Print runtime timing logs. Request payloads and secrets are never printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.default_num_steps) <= 0:
        raise ValueError("--default-num-steps must be > 0.")
    if int(args.max_num_steps) <= 0:
        raise ValueError("--max-num-steps must be > 0.")
    if int(args.default_num_steps) > int(args.max_num_steps):
        raise ValueError("--default-num-steps must be <= --max-num-steps.")
    if args.seconds is not None and float(args.seconds) <= 0:
        raise ValueError("--seconds must be > 0.")
    if float(args.min_seconds) <= 0:
        raise ValueError("--min-seconds must be > 0.")
    if float(args.max_seconds) < float(args.min_seconds):
        raise ValueError("--max-seconds must be >= --min-seconds.")
    if float(args.chars_per_second) <= 0:
        raise ValueError("--chars-per-second must be > 0.")
    if float(args.seconds_padding) < 0:
        raise ValueError("--seconds-padding must be >= 0.")
    max_ref_seconds = float(args.max_ref_seconds)
    settings = ServerSettings(
        host=str(args.host),
        port=int(args.port),
        checkpoint=str(args.checkpoint),
        reference_wav=_project_relative_path(str(args.reference_wav)),
        api_model_id=str(args.api_model_id),
        model_device=str(args.model_device),
        codec_device=str(args.codec_device),
        model_precision=str(args.model_precision),
        codec_precision=str(args.codec_precision),
        codec_repo=str(args.codec_repo),
        default_num_steps=int(args.default_num_steps),
        max_num_steps=int(args.max_num_steps),
        seconds=None if args.seconds is None else float(args.seconds),
        min_seconds=float(args.min_seconds),
        max_seconds=float(args.max_seconds),
        chars_per_second=float(args.chars_per_second),
        seconds_padding=float(args.seconds_padding),
        max_ref_seconds=None if max_ref_seconds <= 0 else max_ref_seconds,
        preload=bool(args.preload),
        log_timings=bool(args.log_timings),
    )
    _validate_reference_wav(settings.reference_wav)

    import uvicorn

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
