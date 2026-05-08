# CoreML Stateful Denoiser P1 実装計画

## Status

- 目的: P1 の実装タスク化。`MLState` KV cache を使う stateful CoreML denoiser one-step PoC/benchmark を作る。
- 対象 branch: `develop`
- remote: `origin=Ramo-Inc/Irodori-TTS-CoreML`, `upstream=Aratako/Irodori-TTS`
- 元設計: `docs/coreml-kv-cache-api-design.md`
- 既存 benchmark: `tools/coreml_real_step_benchmark.py`
- P2/P3 の API/cache manager 作業は、P1a cond-only が `PASS` するまで着手しない。`WARN` は explicit go/no-go まで block、`FAIL` は stop。

## Goal

P1a は、実 checkpoint と実モデル重みを使い、cond-only の text+speaker context KV を CoreML `MLState` に保持した 1 denoiser step を実装、変換、検証、計測する。PyTorch cached path と比較し、CoreML stateful path の correctness、`write_state/read_state` dtype 挙動、`read_state op count`、steady-state latency、ComputePlan の NE placement、`PASS` / `WARN` / `FAIL` status を JSON で出す。

P1b は optional probe として、P1a 成功後に independent CFG active step (`cond + text-uncond + speaker-uncond`) を試す。P1a の blocker にしない。

## Non-goals

- HTTP endpoints は追加しない。
- `openai_api_server.py` や runtime server behavior は変更しない。
- `irodori_tts` production runtime files は変更しない。
- production behavior change は入れない。
- cache manager / lease pool / eviction / TTL は作らない。
- generated `.mlpackage` は commit しない。
- quality を落とすための steps 削減や CFG 弱化は扱わない。
- P1a では full independent CFG を必須にしない。

## Problem

既存の no-state real-model benchmark は CoreML CPU_AND_NE の one-step 実行可能性と no-context-kv-cache の速度を示したが、毎 step で text/speaker condition 由来の context KV projection を通常入力側で処理する前提のままである。API や cache manager を先に作ると、肝心の `MLState` が real denoiser の `read_state` 付き graph で変換できるか、dtype caveat を吸収できるか、NE placement と速度が維持されるかが未検証のままになる。

したがって P1 は API 設計より前に、stateful CoreML one-step を独立 benchmark として固定する。P1a が通れば P2/P3 の cache API は実測済みの backend contract に乗せられる。P1a が失敗する場合、P2/P3 は停止し、CoreML stateful 方針または layout を見直す。

## Baseline Context

既存 no-state PoC 結果:

- `sequence_length=100`, `text_len=256`, `speaker_context_len=138`, `ref_len=137`
- CoreML CPU_AND_NE no-context-kv-cache: `14.428 ms/step`
- PyTorch MPS no-cache: `53.487 ms/step`
- PyTorch MPS cached: `43.881 ms/step`
- `compute_precision=float16`, `status PASS`, `rel_diff=0.0251`
- ComputePlan は過去に `ios16.linear/matmul/softmax` の NE placement を報告したが、P1 は `ios16.*` exact name を hard-code しない。
- 既存 no-state benchmark は normal inputs (`x_t` / `t` / text / speaker / masks) を `np.float32 TensorType` とし、`compute_precision=float16` で変換している。P1a も比較条件を揃えるため、normal prediction inputs (`x_t` / `t` / `latent_mask_f`) は初期値として `np.float32 TensorType` / `np.float32` payload を使う。

CoreML state constraints:

- Stateful conversion target は `ct.target.iOS18`。runtime は macOS 15+。
- fp32 `StateType` は `State only support fp16 dtype. Got input var cache with dtype fp32.` で失敗済み。
- registered torch buffers と `ct.StateType` で wrap する `ct.TensorType` は fp16 に固定する。
- `state.write_state` は `np.float16` を reject し、`np.float32` payload を accept した実績がある。P1 はこの挙動を再検証し、出力 JSON に記録する。
- 同一 `MLState` の concurrent use は unsafe。P1 benchmark は single-threaded とする。
- `CPU_AND_NE` は ANE 強制ではない。CoreML は `CPU_AND_NE` でも CPU placement を選び得るため、P1 は `compute_units` の指定ではなく normalized ComputePlan placement と measured latency で判断する。

P1a state layout options:

Layout A, single packed state with layer slicing, is the initial implementation target:

```text
context_k_state   fp16 [L, 1, C_ctx_bucket, H, D]
context_v_state   fp16 [L, 1, C_ctx_bucket, H, D]
valid_mask_state  fp16 [1, C_ctx_bucket]  # 1.0 valid / 0.0 invalid

L=12, H=20, D=64, T=256, R_bucket=160, C_ctx_bucket=416
state_kv_bytes per cond branch = 25,559,040
```

`valid_mask_state` は `text_mask + speaker_mask` を `T_bucket + R_bucket` に pad したものだけを持つ。latent/self mask は含めない。

Layout B is per-layer smaller states, for example `context_k_l00` / `context_v_l00` ... `context_k_l11` / `context_v_l11`, plus shared or per-layer mask state if needed. P1a does not have to implement layout B, but the benchmark must expose enough evidence to decide whether layout B is needed.

Critical risk: with layout A, a single packed 25MB+ state read per layer may accidentally lower into multiple large `read_state` operations or poorly placed slice/gather ops. That can erase the expected KV projection savings. P1 output must include `read_state op count` from the CoreML program / ComputePlan or MIL program, plus enough state/slice placement detail to triage this risk.

## Files

Add:

- `tools/coreml_stateful_step_benchmark.py`
- `tests/test_coreml_stateful_step_helpers.py`

Existing helper reuse:

- `tools/coreml_real_step_benchmark.py` の checkpoint load、condition encode、RoPE、attention、ComputePlan 走査 helper は audit する。
- shared pure helper への refactor は最小限に限る。既存 benchmark を不安定にするくらいなら、小さい helper は P1 benchmark 側に duplicate する。

Do not touch in P1:

- `openai_api_server.py`
- `irodori_tts/` runtime files
- existing docs。必要がなければ、この新規 doc 以外は変更しない。

## Implementation Phases

### P1.0 Baseline and Helper Reuse Audit

- `git status --short --branch` で作業前の dirty state を確認する。
- `tools/coreml_real_step_benchmark.py` から再利用候補を読む: model load、`prepare_real_inputs`、explicit attention、ComputePlan traversal、JSON status assembly。
- no-state benchmark が比較対象として残ることを優先し、広い refactor は避ける。
- TDD-first で pure helper の shape/mask/packing/status logic を固定する。pytest は 500M checkpoint load、CoreML conversion、実機 `MLState` write、benchmark 実行をしない。
- 最低限の helper tests:
  - `test_compute_state_kv_bytes`
  - `test_pack_context_kv_state_preserves_text_speaker_order`
  - `test_pack_context_kv_state_rejects_speaker_overflow`
  - `test_valid_mask_state_excludes_latent_mask`
  - `test_valid_mask_state_pads_invalid_to_zero`
  - `test_normalize_compute_plan_counts_accepts_ios16_and_ios18`
  - `test_status_warns_when_steady_predict_slower_than_no_state`
  - `test_status_fails_on_rel_diff_or_missing_ne_when_required`

### P1.1 Pack PyTorch Cached KV

- MPS 上の actual model で `model.encode_conditions(...)` を実行する。
- PyTorch cached baseline 用に `model.build_context_kv_cache(text_state, speaker_state)` を呼ぶ。
- `context_kv_cache[i]` の per-layer tuple から `k_text/v_text/k_speaker/v_speaker` を取り出す。
- state は text+speaker context のみを持つ。runtime attention の concat order は current path と同じく `self K/V`、`text K/V`、`speaker K/V` にする。
- `context_k_state[layer, 0, :T, :, :] = k_text`
- `context_k_state[layer, 0, T_bucket:T_bucket+speaker_len, :, :] = k_speaker`
- V も同じ order で pack する。
- `valid_mask_state[0, :T] = text_mask`; `valid_mask_state[0, T_bucket:T_bucket+speaker_len] = speaker_mask`; padding は `0.0`。
- helper tests は bucket overflow、padding zero、mask concat、KV bytes 計算、tuple order mismatch detection を見る。特に `valid_mask_state` は latent/self mask を含めないことを test で固定する。

### P1.2 Stateful CoreML Wrapper

- `RealCoreMLStatefulDenoiserStep` のような benchmark-local wrapper を `tools/coreml_stateful_step_benchmark.py` に置く。
- wrapper は production model modules を参照しつつ、state tensors を registered fp16 buffers として宣言する。
- forward input は `x_t`, `t`, `latent_mask_f` を中心にし、text/speaker state や context KV を通常 input に含めない。P1a baseline ではこれらの normal inputs を `np.float32 TensorType` / `np.float32` payload にする。
- CoreML graph 内では layer ごとに `read_state` した `context_k_state/context_v_state/valid_mask_state` を使う。
- attention 内で `k_self/v_self` を通常通り毎 step 計算し、read state の context K/V と concat する。
- `key_mask_f` は `latent_mask_f + valid_mask_state` で作り、explicit attention の additive mask に渡す。

### P1.3 Convert and Validate State Dtype

- conversion は `convert_to="mlprogram"`, `minimum_deployment_target=ct.target.iOS18`, `compute_units=ct.ComputeUnit.CPU_AND_NE`, `compute_precision=float16` を標準にする。
- `ct.StateType(wrapped_type=ct.TensorType(..., dtype=np.float16), name=...)` を使う。
- normal input `ct.TensorType` は `--io-dtype` default の `float32` に合わせて `dtype=np.float32` とする。`--io-dtype float16` は optional experiment であり、P1a PASS 判定の baseline にしない。
- state buffer は `torch.float16` で register し、fp32 混入を test/helper assert で落とす。
- `mlmodel.make_state()` を計測し、`make_state_ms` として出す。
- 初期書き込みは `np.float32` payload を使う。`write_state_ms` を計測し、`read_state` 可能なら dtype/shape/max diff を記録する。
- `np.float16` payload は optional negative/diagnostic として試してよいが、P1 acceptance は `np.float32` write path が動くことに置く。

### P1.4 Correctness vs PyTorch Cached Path

- PyTorch baseline は no-cache ではなく、`forward_with_encoded_conditions(..., context_kv_cache=context_kv_cache)` を使う。
- CoreML stateful predict は同じ `x_t`, `t`, `latent_mask` と packed state で 1 step を走らせる。
- `max_abs_diff` と `rel_diff` を計算し、default gate は `--max-rel-diff 0.05`。
- no toy weights、no dummy checkpoint、no synthetic tiny model fallback。debug override は明示オプションだけにする。

### P1.5 Benchmark and ComputePlan

- time scope は PyTorch cached も CoreML steady-state も one denoiser step のみ。
- `convert_seconds`, `make_state_ms`, `write_state_ms`, `first_predict_ms`, `steady_predict_ms` を分離する。
- generated artifacts は temporary dir か ignored `outputs/coreml-stateful-step-benchmark/` に置き、commit しない。
- ComputePlan は raw `operator_name` count と normalized category count の両方を出す。
- normalized category は suffix で `linear`, `matmul`, `softmax` に分類する。`ios16.linear` のような exact name には依存しない。
- raw と normalized の両方で preferred device counts を出す。
- CoreML program / ComputePlan / MIL program のいずれかから `read_state op count`、state name 別 read count、slice/gather placement、large read に見える op count を出す。single packed layout A の evidence が悪ければ per-layer layout B を次に検討する。
- default `require_ne_placement=true` では normalized `linear/matmul/softmax` に NE-preferred ops があることを PASS 条件にする。CLI は `--no-require-ne-placement` のみを提供し、disable 時は status reason に残して fail しない。
- `CPU_AND_NE` は ANE 強制ではないため、status 判定は `compute_units` ではなく normalized ComputePlan placement と `steady_predict_ms` を使う。

### P1.6 Optional Independent3 Probe

- P1a PASS 後だけ実施する。
- `--mode independent3` を追加し、active CFG 相当の branch layout を試す。
- `cond-only` の task-ready acceptance には含めない。
- independent3 が遅い、または mismatch しても P1a 完了を取り消さない。ただし P2/P3 の full CFG API は P1b 結果を見て設計する。

## CLI Spec

`tools/coreml_stateful_step_benchmark.py`:

- positional `text` は既存 benchmark と同様に optional。default は既存 Japanese sample。
- `--seconds 4` default。
- `--sequence-length` optional。未指定なら seconds-derived。
- `--speaker-context-bucket 160` default。
- `--compute-precision float16` default。
- `--io-dtype float32|float16` default `float32`。`float16` は optional experiment で、P1a baseline/status 判定は `float32` で行う。
- `--iterations`, `--warmup`
- `--ref-wav`, `--checkpoint`, `--codec-repo`
- `require_ne_placement=true` default。CLI flag は existing benchmark style に合わせて `--no-require-ne-placement` のみを提供し、`--require-ne-placement` との二重 flag にはしない。
- `--max-rel-diff 0.05` default。
- `--mode cond-only` initially。P1b で `independent3` を optional 追加。

JSON summary fields:

```text
sequence_length
text_len
speaker_context_len
speaker_context_bucket
c_ctx_bucket
state_kv_bytes
state_layout
io_dtype
convert_seconds
make_state_ms
write_state_ms
first_predict_ms
steady_predict_ms
pytorch_mps_cached_avg_ms
speedup_vs_pytorch_cached
max_abs_diff
rel_diff
raw_compute_plan_counts
normalized_compute_plan_counts
read_state_op_count
read_state_counts_by_state
large_read_state_ops
slice_placement_summary
status
status_reasons
```

`status` と `status_reasons` は required。`status_level` は使わず、status value は `PASS` / `WARN` / `FAIL` に統一する。

追加で `write_state_payload_dtype`, `read_state_dtype`, `mode`, `compute_precision`, `require_ne_placement`, `first_use_cost_reasons` を出すと triage が楽になる。

Status semantics:

- `PASS`: correctness、state write/read、normalized placement が pass し、`steady_predict_ms < 14.428`。P2/P3 may proceed。
- `WARN`: correctness と placement は pass するが、`steady_predict_ms >= 14.428`、`first_predict_ms` / `write_state_ms` が大きい、または `read_state op count` / slicing placement が concerning。P2/P3 remain blocked pending explicit go/no-go。
- `FAIL`: conversion、`np.float32 write_state`、correctness、required NE placement のいずれかが fail。Stop。

## Acceptance Criteria

- `uv run --with pytest pytest tests/test_coreml_stateful_step_helpers.py -q` が pass。
- helper pytest は pure helper/unit tests のみで、500M checkpoint load、CoreML conversion、`MLState` write、benchmark 実行をしない。
- stateful conversion が fp16 `StateType` で成功する。
- `np.float32` `write_state` path が動き、出力 JSON に dtype 挙動が記録される。
- normal prediction input `x_t` / `t` / `latent_mask_f` は P1a baseline で `np.float32 TensorType` を使う。`--io-dtype float16` は optional experiment であり baseline ではない。
- CoreML stateful output が PyTorch cached path に対して `rel_diff <= 0.05`。
- normalized ComputePlan placement が pass。
- JSON は required fields として `status`、`status_reasons`、`read_state_op_count`、`normalized_compute_plan_counts`、`steady_predict_ms` を持つ。
- `steady_predict_ms < 14.428` なら speed gate pass。超えた場合は correctness/placement が pass なら `status WARN` とし、`read_state op count` と slicing placement を含む理由を `status_reasons` に明記する。
- `status PASS`: correctness + placement + speed gate pass。P2/P3 may proceed。
- `status WARN`: correctness + placement pass だが speed / first-use cost / state layout evidence に懸念がある。P2/P3 remain blocked pending explicit go/no-go。
- `status FAIL`: conversion / correctness / `np.float32 write_state` / required placement fail。Stop。
- 全 metrics を JSON として stdout に出す。

## Failure Decision Tree

- fp16 `StateType` conversion fails: buffer dtype、`ct.StateType` dtype、`minimum_deployment_target=ct.target.iOS18`、state shape の static 性を確認する。ここで失敗する間は P2/P3 stop。
- `np.float32 write_state` fails: coremltools/macOS version、state name、payload shape、contiguity、dtype を確認する。write path が確定するまで P2/P3 stop。
- output mismatch: pack order、mask padding、`self/text/speaker` concat order、RoPE dtype、PyTorch baseline が cached path かを確認する。correctness が通るまで P2/P3 stop。
- NE placement collapses: raw operator diff、`read_state`/slice ops の挿入位置、deployment target、compute precision を確認する。correctness は残しても P2/P3 performance API は stop。
- steady predict slower than no-state `14.428ms`: まず `read_state op count`、single packed state の layer slicing placement、large read に見える op、per-layer layout B の必要性を確認する。次に first predict と steady predict の混同、state read overhead、padding size、ComputePlan placement を確認する。correctness/placement pass なら P1a は `WARN` にできるが、P2/P3 は explicit go/no-go 判断まで stop。

## Commands

```bash
uv run --with pytest pytest tests/test_coreml_stateful_step_helpers.py -q
uv run --with 'coremltools>=8.0' python tools/coreml_stateful_step_benchmark.py --seconds 4 --sequence-length 100 --speaker-context-bucket 160 --io-dtype float32 --iterations 5 --warmup 2
git status --short --branch
```

## Handoff Prompt

```text
Work only in /Users/ramo/Services/Irodori-TTS. Implement P1a from docs/coreml-stateful-denoiser-p1-implementation-plan.md. Add tools/coreml_stateful_step_benchmark.py and tests/test_coreml_stateful_step_helpers.py only unless a minimal pure-helper refactor is clearly safer. Do not edit production runtime or openai_api_server.py. Start TDD-first with pure helper tests only; pytest must not load the 500M checkpoint or run CoreML conversion. Keep normal prediction inputs x_t/t/latent_mask_f as np.float32 TensorType / np.float32 payload for the P1a baseline, while StateType stays fp16 and state.write_state initially uses np.float32 payload. Use actual checkpoint/model weights for the benchmark, pack text+speaker context KV into fp16 MLState, compare CoreML stateful one-step output against PyTorch forward_with_encoded_conditions with context_kv_cache, report read_state op count plus ComputePlan placement and JSON metrics, use PASS/WARN/FAIL with required status/status_reasons semantics, and keep generated mlpackages out of git. P2/P3 API work is blocked until P1a is PASS; WARN requires explicit go/no-go.
```
