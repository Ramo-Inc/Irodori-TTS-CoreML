# CoreML Stateful Denoiser P1 実装計画

## Status

- 目的: P1 の実装タスク化。`MLState` KV cache を使う stateful CoreML denoiser one-step PoC/benchmark を作る。
- 対象 branch: `develop`
- remote: `origin=Ramo-Inc/Irodori-TTS-CoreML`, `upstream=Aratako/Irodori-TTS`
- 元設計: `docs/coreml-kv-cache-api-design.md`
- 既存 benchmark: `tools/coreml_real_step_benchmark.py`
- P1 stateful CoreML one-step evidence は per-layer layout で `PASS`。P2/P3 の API/cache manager 作業は unblocked だが、per-request `MLState` と per-layer condition KV states を使う。
- Packed layout は `WARN` の comparison/debug evidence として残す。`FAIL` は引き続き stop。

## Goal

P1a は、実 checkpoint と実モデル重みを使い、cond-only の text+speaker context KV を CoreML `MLState` に保持した 1 denoiser step を実装、変換、検証、計測する。PyTorch cached path と比較し、CoreML stateful path の correctness、`write_state/read_state` dtype 挙動、`read_state op count`、steady-state latency、ComputePlan の NE placement、`PASS` / `WARN` / `FAIL` status を JSON で出す。

P1b は state layout probe として `--state-layout packed|per-layer` を比較する。independent CFG active step (`cond + text-uncond + speaker-uncond`) は P1 one-step stateful denoiser の blocker にしない。

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

既存の no-state real-model benchmark は CoreML CPU_AND_NE の one-step 実行可能性と no-context-kv-cache の速度を示したが、毎 step で text/speaker condition 由来の context KV projection を通常入力側で処理する前提のままである。P1 では、real denoiser の `read_state` 付き graph が per-layer layout で correctness、NE placement、speed を満たすことを確認した。

したがって P1 は API 設計より前に、stateful CoreML one-step を独立 benchmark として固定する。実測では per-layer layout が `PASS` したため、P2/P3 の cache API は実測済みの backend contract に乗せられる。packed layout は `WARN` なので production-facing baseline にはしない。

## Baseline Context

既存 no-state PoC 結果:

- `sequence_length=100`, `text_len=256`, `speaker_context_len=138`, `ref_len=137`
- CoreML CPU_AND_NE no-context-kv-cache: `14.428 ms/step`
- PyTorch MPS no-cache: `53.487 ms/step`
- PyTorch MPS cached: `43.881 ms/step`
- `compute_precision=float16`, `status PASS`, `rel_diff=0.0251`
- ComputePlan は過去に `ios16.linear/matmul/softmax` の NE placement を報告したが、P1 は `ios16.*` exact name を hard-code しない。
- 既存 no-state benchmark は normal inputs (`x_t` / `t` / text / speaker / masks) を `np.float32 TensorType` とし、`compute_precision=float16` で変換している。P1 cond-only baseline も比較条件を揃えるため、normal prediction inputs (`x_t` / `t` / `latent_mask_f`) は初期値として `np.float32 TensorType` / `np.float32` payload を使う。

CoreML state constraints:

- Stateful conversion target は `ct.target.iOS18`。runtime は macOS 15+。
- fp32 `StateType` は `State only support fp16 dtype. Got input var cache with dtype fp32.` で失敗済み。
- registered torch buffers と `ct.StateType` で wrap する `ct.TensorType` は fp16 に固定する。
- `state.write_state` は `np.float16` を reject し、`np.float32` payload を accept した実績がある。P1 はこの挙動を再検証し、出力 JSON に記録する。
- 同一 `MLState` の concurrent use は unsafe。P1 benchmark は single-threaded とする。
- `CPU_AND_NE` は ANE 強制ではない。CoreML は `CPU_AND_NE` でも CPU placement を選び得るため、P1 は `compute_units` の指定ではなく normalized ComputePlan placement と measured latency で判断する。

P1 state layout results:

The benchmark now exposes `--state-layout packed|per-layer`.

Packed layout:

```text
state_layout=packed_text_speaker_context_v1
read_state_op_count=3
max_read_state_bytes_fp16=12,779,520
slice_by_index total=72
steady_predict_ms around 13.04-13.07ms
status WARN
```

Packed remains fast in steady predict, but the layout produces two giant `read_state` ops. It remains useful only as comparison/debug evidence.

Per-layer layout:

```text
state_layout=per_layer_text_speaker_context_v1
state names=context_k_l00/context_v_l00 ... context_k_l11/context_v_l11 plus valid_mask_state
read_state_op_count=25
max_read_state_bytes_fp16=1,064,960
slice_by_index total=48
steady_predict_ms=13.1818ms
pytorch_mps_cached_avg_ms=41.5274ms
speedup_vs_pytorch_cached=3.150x
rel_diff=0.02507449
NE placement=linear 245/245, matmul 24/24, softmax 12/12 NE-preferred
write_state_ms=6.8466ms
make_state_ms=3.3799ms
status PASS
```

Conclusion: use `per_layer_text_speaker_context_v1` as the production-facing P2/P3 baseline. This is one-step stateful denoiser evidence, not proof that full multi-step generation is implemented.

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
- Per-layer production layout: `context_k_lNN[0, :T, :, :] = k_text`
- Per-layer production layout: `context_k_lNN[0, T_bucket:T_bucket+speaker_len, :, :] = k_speaker`
- Packed comparison layout: `context_k_state[layer, 0, ...]` uses the same order but remains debug-only.
- V も同じ order で pack する。
- `valid_mask_state[0, :T] = text_mask`; `valid_mask_state[0, T_bucket:T_bucket+speaker_len] = speaker_mask`; padding は `0.0`。
- helper tests は bucket overflow、padding zero、mask concat、KV bytes 計算、tuple order mismatch detection を見る。特に `valid_mask_state` は latent/self mask を含めないことを test で固定する。

### P1.2 Stateful CoreML Wrapper

- `RealCoreMLStatefulDenoiserStep` のような benchmark-local wrapper を `tools/coreml_stateful_step_benchmark.py` に置く。
- wrapper は production model modules を参照しつつ、state tensors を registered fp16 buffers として宣言する。
- forward input は `x_t`, `t`, `latent_mask_f` を中心にし、text/speaker state や context KV を通常 input に含めない。P1 cond-only baseline ではこれらの normal inputs を `np.float32 TensorType` / `np.float32` payload にする。
- CoreML graph 内では layer ごとに `read_state` した `context_k_lNN/context_v_lNN/valid_mask_state` を使う。Packed comparison run では `context_k_state/context_v_state` を読むが、production baseline にはしない。
- attention 内で `k_self/v_self` を通常通り毎 step 計算し、read state の context K/V と concat する。
- `key_mask_f` は `latent_mask_f + valid_mask_state` で作り、explicit attention の additive mask に渡す。

### P1.3 Convert and Validate State Dtype

- conversion は `convert_to="mlprogram"`, `minimum_deployment_target=ct.target.iOS18`, `compute_units=ct.ComputeUnit.CPU_AND_NE`, `compute_precision=float16` を標準にする。
- `ct.StateType(wrapped_type=ct.TensorType(..., dtype=np.float16), name=...)` を使う。
- normal input `ct.TensorType` は `--io-dtype` default の `float32` に合わせて `dtype=np.float32` とする。`--io-dtype float16` は optional experiment であり、P1 PASS 判定の baseline にしない。
- state buffer は `torch.float16` で register し、fp32 混入を test/helper assert で落とす。
- `mlmodel.make_state()` を計測し、`make_state_ms` として出す。
- 初期書き込みは `np.float32` payload を使う。`write_state_ms` を計測し、`read_state` 可能なら dtype/shape/max diff を記録する。
- `np.float16` payload は optional negative/diagnostic として試してよいが、P1 acceptance は `np.float32` write path が動くことに置く。

### P1.4 Correctness vs PyTorch Cached Path

- PyTorch baseline は no-cache ではなく、`forward_with_encoded_conditions(..., context_kv_cache=context_kv_cache)` を使う。
- CoreML stateful predict は同じ `x_t`, `t`, `latent_mask` と selected state layout で 1 step を走らせる。
- `max_abs_diff` と `rel_diff` を計算し、default gate は `--max-rel-diff 0.05`。
- no toy weights、no dummy checkpoint、no synthetic tiny model fallback。debug override は明示オプションだけにする。

### P1.5 Benchmark and ComputePlan

- time scope は PyTorch cached も CoreML steady-state も one denoiser step のみ。
- `convert_seconds`, `make_state_ms`, `write_state_ms`, `first_predict_ms`, `steady_predict_ms` を分離する。
- generated artifacts は temporary dir か ignored `outputs/coreml-stateful-step-benchmark/` に置き、commit しない。
- ComputePlan は raw `operator_name` count と normalized category count の両方を出す。
- normalized category は suffix で `linear`, `matmul`, `softmax` に分類する。`ios16.linear` のような exact name には依存しない。
- raw と normalized の両方で preferred device counts を出す。
- CoreML program / ComputePlan / MIL program のいずれかから `read_state op count`、state name 別 read count、`max_read_state_bytes_fp16`、slice/gather placement、large read に見える op count を出す。P1b では packed が `WARN`、per-layer が `PASS` した。
- default `require_ne_placement=true` では normalized `linear/matmul/softmax` に NE-preferred ops があることを PASS 条件にする。CLI は `--no-require-ne-placement` のみを提供し、disable 時は status reason に残して fail しない。
- `CPU_AND_NE` は ANE 強制ではないため、status 判定は `compute_units` ではなく normalized ComputePlan placement と `steady_predict_ms` を使う。

### P1.6 Optional Independent3 Probe

- Per-layer layout `PASS` 後だけ実施する。
- `--mode independent3` を追加し、active CFG 相当の branch layout を試す。
- `cond-only` の task-ready acceptance には含めない。
- independent3 が遅い、または mismatch しても P1 one-step stateful denoiser 完了を取り消さない。ただし P2/P3 の full CFG API は independent3 結果を見て設計する。

## CLI Spec

`tools/coreml_stateful_step_benchmark.py`:

- positional `text` は既存 benchmark と同様に optional。default は既存 Japanese sample。
- `--seconds 4` default。
- `--sequence-length` optional。未指定なら seconds-derived。
- `--speaker-context-bucket 160` default。
- `--compute-precision float16` default。
- `--io-dtype float32|float16` default `float32`。`float16` は optional experiment で、P1 baseline/status 判定は `float32` で行う。
- `--iterations`, `--warmup`
- `--ref-wav`, `--checkpoint`, `--codec-repo`
- `require_ne_placement=true` default。CLI flag は existing benchmark style に合わせて `--no-require-ne-placement` のみを提供し、`--require-ne-placement` との二重 flag にはしない。
- `--max-rel-diff 0.05` default。
- `--mode cond-only` initially。後続で `independent3` を optional 追加。
- `--state-layout packed|per-layer` default should be `per-layer` for production-facing runs. `packed` is comparison/debug.

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
max_read_state_bytes_fp16
large_read_state_ops
slice_placement_summary
status
status_reasons
```

`status` と `status_reasons` は required。`status_level` は使わず、status value は `PASS` / `WARN` / `FAIL` に統一する。

追加で `write_state_payload_dtype`, `read_state_dtype`, `mode`, `compute_precision`, `require_ne_placement`, `first_use_cost_reasons` を出すと triage が楽になる。

Status semantics:

- `PASS`: correctness、state write/read、normalized placement、speed、state layout evidence が production baseline として pass。P2/P3 may proceed for that layout。
- `WARN`: correctness と placement は pass するが、`steady_predict_ms >= 14.428`、`first_predict_ms` / `write_state_ms` が大きい、または `read_state op count` / `max_read_state_bytes_fp16` / slicing placement が concerning。Packed layout は fast steady predict でもこの理由で `WARN`。
- `FAIL`: conversion、`np.float32 write_state`、correctness、required NE placement のいずれかが fail。Stop。

## Acceptance Criteria

- `uv run --with pytest pytest tests/test_coreml_stateful_step_helpers.py -q` が pass。
- helper pytest は pure helper/unit tests のみで、500M checkpoint load、CoreML conversion、`MLState` write、benchmark 実行をしない。
- stateful conversion が fp16 `StateType` で成功する。
- `np.float32` `write_state` path が動き、出力 JSON に dtype 挙動が記録される。
- normal prediction input `x_t` / `t` / `latent_mask_f` は P1 baseline で `np.float32 TensorType` を使う。`--io-dtype float16` は optional experiment であり baseline ではない。
- CoreML stateful output が PyTorch cached path に対して `rel_diff <= 0.05`。
- normalized ComputePlan placement が pass。
- JSON は required fields として `status`、`status_reasons`、`read_state_op_count`、`max_read_state_bytes_fp16`、`normalized_compute_plan_counts`、`steady_predict_ms` を持つ。
- `steady_predict_ms < 14.428` は speed gate pass だが、それだけでは `PASS` にしない。Packed layout は fast steady predict でも巨大 `read_state` evidence により `WARN`。
- `status PASS`: correctness + placement + speed + state layout evidence pass。Per-layer verified run は `PASS` で、P2/P3 may proceed with per-layer layout。
- `status WARN`: correctness + placement pass だが speed / first-use cost / state layout evidence に懸念がある。その layout は production-facing baseline にしない。
- `status FAIL`: conversion / correctness / `np.float32 write_state` / required placement fail。Stop。
- 全 metrics を JSON として stdout に出す。

## Failure Decision Tree

- fp16 `StateType` conversion fails: buffer dtype、`ct.StateType` dtype、`minimum_deployment_target=ct.target.iOS18`、state shape の static 性を確認する。ここで失敗する間は P2/P3 stop。
- `np.float32 write_state` fails: coremltools/macOS version、state name、payload shape、contiguity、dtype を確認する。write path が確定するまで P2/P3 stop。
- output mismatch: pack order、mask padding、`self/text/speaker` concat order、RoPE dtype、PyTorch baseline が cached path かを確認する。correctness が通るまで P2/P3 stop。
- NE placement collapses: raw operator diff、`read_state`/slice ops の挿入位置、deployment target、compute precision を確認する。correctness は残しても P2/P3 performance API は stop。
- steady predict slower than no-state `14.428ms`: まず `read_state op count`、`max_read_state_bytes_fp16`、state slice placement、padding size、ComputePlan placement を確認する。correctness/placement pass なら `WARN`。
- steady predict が fast でも giant `read_state` がある: packed result と同様に `WARN`。per-layer layout を production-facing baseline にする。

## Commands

```bash
uv run --with pytest pytest tests/test_coreml_stateful_step_helpers.py -q
uv run --with 'coremltools>=8.0' python tools/coreml_stateful_step_benchmark.py --seconds 4 --sequence-length 100 --speaker-context-bucket 160 --io-dtype float32 --state-layout per-layer --iterations 5 --warmup 2
git status --short --branch
```

## Handoff Prompt

```text
Work only in /Users/ramo/Services/Irodori-TTS. Continue from the P1 one-step stateful CoreML evidence. Use `per_layer_text_speaker_context_v1` as the P2/P3 production-facing baseline with per-request `MLState` and per-layer condition KV state names `context_k_l00/context_v_l00` ... `context_k_l11/context_v_l11` plus `valid_mask_state`. Keep packed layout only as comparison/debug evidence because it is `WARN` due giant `read_state` ops. Do not claim full multi-step generation is already implemented.
```
