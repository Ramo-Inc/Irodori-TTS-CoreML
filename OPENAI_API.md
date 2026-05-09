# OpenAI-compatible TTS API

Start the LAN server:

```bash
./start_openai_server.sh
```

The start script defaults to `0.0.0.0:19841`. Override with env vars or trailing
arguments; later argparse options win:

```bash
IRODORI_TTS_HOST=127.0.0.1 IRODORI_TTS_PORT=19842 ./start_openai_server.sh
./start_openai_server.sh --host 127.0.0.1 --port 19842
```

Defaults:

- Listen address: `0.0.0.0:19841`
- Local URL: `http://127.0.0.1:19841`
- LAN URL: `http://192.168.0.111:19841`
- Checkpoint: `Aratako/Irodori-TTS-500M-v2`
- Model and codec device: `mps` when available
- Server-forced reference voice: `rem.wav`
- Default sampling steps: `30`
- Maximum request sampling steps: `80`
- Generation seconds: automatic estimate from input length

By default, `rem.wav` must exist next to `openai_api_server.py`. To use a different
server-owned reference file, start with `--reference-wav path/to/reference.wav`.

The server estimates the generation horizon from non-whitespace input length, then clamps
it between `--min-seconds` and `--max-seconds`. Defaults are tuned for medium-length
Japanese text: `--min-seconds 4`, `--max-seconds 70`, `--chars-per-second 5.5`, and
`--seconds-padding 1.5`. The `chars_per_second=5.5` default is calibrated from measured
Japanese explanatory speech (~5.37 chars/s); the previous `4.0` overestimated generation
seconds and pushed segments into unnecessarily large CoreML S buckets. This lets a single
AUTO request cover up to about 256 non-whitespace characters in one CoreML bucket without
splitting. Use `--seconds N` to force one fixed horizon for all requests, or send
extension field `"seconds": N` in one request to override only that request.
Request-level `seconds` must be between `0.1` and `--max-seconds`.

The server also applies a default playback speed multiplier via `--default-speed`
(default `1.0`, the safe value). It scales the effective chars-per-second used for
AUTO duration estimation and CoreML bucket selection only — explicit request-level
`seconds` and the `--seconds` server-fixed horizon are unaffected. Clients may opt
into a faster default per request with the OpenAI `speed` field (range `0.25..4.0`);
the resolved value is echoed back via `X-Irodori-Speed`. Note that `speed` >1.0
shortens the AUTO generation horizon proportionally and may truncate the end of
generated speech if set too high — for long or critical text, send an explicit
`seconds` value, which is the safest option and bypasses speed-based estimation.
Padding (`--seconds-padding`) is preserved unchanged regardless of speed.

For AUTO requests, the server selects the smallest CoreML bucket whose
`(sequence_length, text_len)` covers the request from this preset list:
`(S=256, T=32)`, `(S=512, T=64)`, `(S=1024, T=128)`, `(S=1536, T=192)`,
`(S=2048, T=256)`, all with `R=160` for the speaker context bucket. Inputs whose
estimated `patched_steps` exceed `2048` or whose `token_len` exceeds `256` are split
near the midpoint at sentence boundaries, then phrase punctuation, then whitespace
(including the full-width space `　`), then a hard split if no boundary exists.
After the runtime is loaded, AUTO requests are re-checked with the runtime tokenizer and
estimator and split further if a chunk still exceeds the bucket maximums. Pass
`--strict-coreml` to make AUTO requests raise an explicit error instead of silently
falling back to PyTorch when no usable condition cache can be prepared. `cache_mode=off`
remains an explicit PyTorch opt-out even when `--strict-coreml` is enabled.

Clients may send OpenAI fields such as `model`, `voice`, `response_format`, and `speed`.
The `voice` field and client reference fields such as `reference_audio` are accepted for
compatibility but ignored; the server always uses its configured reference wav.
Supported `response_format` values are `wav`, `mp3`, `flac`, `pcm`, `aac`, and `opus`;
`aac` and `opus` are returned as MP3 for pragmatic compatibility.
Responses include `X-Irodori-Generation-Seconds` and `X-Irodori-Seconds-Mode` headers
to show the selected horizon and whether it came from `auto`, `fixed`, or `request`.
Request-level `num_steps` is capped by `--max-num-steps` and the response includes
`X-Irodori-Num-Steps` for tuning.

### Style / control instructions

Three optional fields steer the Irodori-TTS caption-conditioned style/control input:

- `instructions` — OpenAI-compatible style instruction.
- `instruction` — singular alias for `instructions`. Provided for clients that prefer
  the singular form.
- `caption` — original Irodori extension; kept for backward compatibility.

All three are optional strings. Whitespace-only values are treated as unspecified.
The resolved value is forwarded to Irodori's internal `SamplingRequest.caption`, so
`instructions` is mapped to the `caption` style-control input. If two or more of
these fields are provided with different non-empty values the request is rejected
with HTTP 400; equal values are accepted.

Caption conditioning takes effect on caption-enabled or VoiceDesign checkpoints. On
checkpoints that do not include caption conditioning the value is effectively
ignored, even if accepted by the API.

Health and model list:

```bash
curl http://127.0.0.1:19841/v1/health
curl http://192.168.0.111:19841/v1/health
curl http://127.0.0.1:19841/v1/models
```

Generate WAV audio:

```bash
mkdir -p outputs
curl -sS http://127.0.0.1:19841/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori-tts-500m-v2","input":"こんにちは。音声合成のテストです。","voice":"default","response_format":"wav","num_steps":40}' \
  --output outputs/api_example.wav
```

Generate WAV audio with a style/control `instruction`:

```bash
curl -sS http://127.0.0.1:19841/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori-tts-500m-v2","input":"こんにちは。音声合成のテストです。","voice":"default","response_format":"wav","num_steps":40,"instruction":"明るく元気に、少し早口で"}' \
  --output outputs/api_example_instruction.wav
```

The plural `instructions` field works the same way; the legacy `caption` field is also
still accepted for backward compatibility.

OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:19841/v1",
    api_key="not-needed",
)

with client.audio.speech.with_streaming_response.create(
    model="irodori-tts-500m-v2",
    voice="default",
    input="こんにちは。音声合成のテストです。",
    response_format="wav",
    extra_body={
        "num_steps": 40,
        "instruction": "明るく元気に、少し早口で",
    },
) as response:
    response.stream_to_file("outputs/api_example.wav")
```

`instruction` is the singular Irodori extension; pass `instructions` instead if your
client prefers the OpenAI-compatible plural form. Either field is forwarded to the
caption-conditioned style/control input.

LaunchAgent autostart:

The project includes [launchd/com.ramo.irodori-tts-openai-api.plist](launchd/com.ramo.irodori-tts-openai-api.plist).
It runs `/opt/homebrew/bin/uv run python openai_api_server.py --host 0.0.0.0 --port 19841 --model-device mps --codec-device mps --preload --strict-coreml --max-seconds 70 --chars-per-second 5.5 --default-speed 1.0 --default-num-steps 30 --max-resident-speaker-kv-buckets 5` and warms up
buckets `S=256,T=32,R=160`, `S=512,T=64,R=160`, `S=1024,T=128,R=160`,
`S=1536,T=192,R=160`, and `S=2048,T=256,R=160` from
`/Users/ramo/Services/Irodori-TTS`. As a user LaunchAgent, it starts at user login.

The plist also passes `--default-condition-cache-prepare-text` with a short
Japanese phrase. At startup the server builds a segment plan from this text,
resolves the AUTO bucket, and prepares the condition / text-encoder cache against
the default reference. No audio is synthesized; only the text / condition path
is warmed so the first real short request avoids the one-time initialization
cost (~20s on first call without warmup). Under `--strict-coreml`, startup
raises `CoreMLStatefulUnavailableError` if any warmup segment fails to obtain a
condition handle (e.g., bucket oversize, CoreML unavailable); without
`--strict-coreml`, the warmup logs a warning and continues.

```bash
mkdir -p logs ~/Library/LaunchAgents
cp launchd/com.ramo.irodori-tts-openai-api.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ramo.irodori-tts-openai-api.plist
launchctl kickstart -k gui/$(id -u)/com.ramo.irodori-tts-openai-api
launchctl print gui/$(id -u)/com.ramo.irodori-tts-openai-api
```

Logs are written under the project:

```bash
tail -f logs/openai-api.out.log logs/openai-api.err.log
```

Unload the LaunchAgent:

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.ramo.irodori-tts-openai-api.plist
```

For smoke tests, reduce work server-side and per request:

```bash
./start_openai_server.sh --seconds 2
curl -sS http://127.0.0.1:19841/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"irodori-tts-500m-v2","input":"テストです。","voice":"ignored","response_format":"wav","num_steps":2}' \
  --output outputs/api_smoke.wav
```
