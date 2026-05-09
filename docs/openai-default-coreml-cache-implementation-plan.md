# OpenAI 互換 `/v1/audio/speech` 既定 CoreML cache 実装計画

## 1. Title / Scope / Non-Scope

**Title:** OpenAI-compatible `/v1/audio/speech` を server default `rem.wav`、自動 bucket、自動 condition cache、CoreML stateful fast path に既定化する。

**Scope:**

- `irodori` 拡張なしの OpenAI 互換リクエストを、既定で `cache_mode=auto` として扱う。
- OpenAI `voice` と client-side reference 指定は受け取っても無視し、サーバ固定 `rem.wav` のみを使う。
- `seconds` と token 長から `S={100,160,200}`, `T={64,128,256}`, `R=160` の bucket を自動選択する。
- AUTO は CoreML/bucket 失敗時も HTTP 200 を維持し、PyTorch に fallback する。
- `require` / `prepare` / `refresh` / 明示 `off` の既存 contract を維持する。
- resident reference latent、speaker state、bucketed speaker KV、必要に応じて condition packed-KV を導入し、少なくとも 2 回目リクエストで `rem.wav` reference encode と speaker KV projection を skip する。

**Non-Scope:**

- B=2/B=3 CoreML backend の再導入。runtime は B=1 `cond1` を複数回呼ぶ split cond1 戦略を維持する。
- server-side voice registry、複数 voice preload、client reference の resident 化。
- MLState pool / lease の本格導入。packed-KV layer 4 は任意、MLState residency は別計画。
- 生成済み wav response cache、PyTorch 経路自体の高速化、DACVAE/text/speaker encoder の CoreML 化。
- 細粒度 S bucket (`64` / `80` / `128`) の production 既定追加。

## 2. Current Baseline and Checkpoints

- 設計ソースは [docs/openai-default-coreml-cache-plan.md](/Users/ramo/Services/Irodori-TTS/docs/openai-default-coreml-cache-plan.md)。
- 本書作成時点の作業基準は、現在の worktree/HEAD を「P4/P5 stateful CoreML denoiser が production baseline として PASS した状態」と概念的に扱う。ここでは `git tag` / `git commit` / HEAD SHA 取得は実行しない。
- 現 baseline の重要点:
  - [openai_api_server.py](/Users/ramo/Services/Irodori-TTS/openai_api_server.py) の `_speech_cache_mode()` は `irodori is None` で `CACHE_MODE_OFF` を返す。
  - `_resolve_speech_cache()` は単一 `_SpeechCacheResolution` を返し、`AUTO + cache_id None` は cache miss として PyTorch に落ちる。
  - `_condition_cache_bucket()` の既定は `S=100,T=256,R=160` で、5 秒級生成の `patched_steps=125` に足りない。
  - `InMemoryCoreMLCacheManager` は `ReferenceCacheHandle` / `ConditionCacheHandle` のメタデータのみを保持する。
  - `InferenceRuntime.synthesize_with_condition_cache()` は毎回 `_load_reference_latent()` と `_sample_coreml_stateful_rf_cfg()` 内の `build_context_kv_cache()` / `pack_context_kv_state()` を実行する。
  - `_reference_cache_request()` は未指定時 `speaker_context_len=1` を使うため、resident default cache のメタデータには使えない。
  - `_condition_fingerprint()` は raw `seconds` を含むため、AUTO 再利用用には bucket-normalized fingerprint v2 が必要。
  - cache manager の dict / metrics / evict path は現状 lock なし。resident pool 導入前に同期境界が必要。
  - response header は既存 `X-Irodori-Condition-Cache-Id` / `X-Irodori-Reference-Cache-Id` / `X-Irodori-Denoiser-Backend` を出す。
  - 起動時 warmup は `precompile_coreml_stateful_buckets()` のみで、condition cache は温まらない。
- P0 で `git status --short --branch` と既存テスト baseline を記録する。リスクの高い P4/P5/P6 前は人間が checkpoint tag を切れる状態にしてから進む。

**Review evaluation:** 前回の critical review 指摘は現コードに照合済み。上記 baseline と一致したものだけを本計画へ採用し、runtime planning seam、runtime-resident ownership、fingerprint v2、speaker-only KV、lock、header alias、実 tensor memory accounting を必須化する。

## 3. Implementation Principles

- B=2/B=3 CoreML backend は戻さない。`joint2` / `alternating_*` は API/metadata として存在してよいが、実行は split cond1 のままにする。
- OpenAI `voice`、`reference_audio`、`reference_url`、`irodori.reference` は本計画では無視し、`rem.wav` 固定にする。観測用に `X-Irodori-Voice-Resolved: server-default` を返す。
- `AUTO` 既定は UX contract。CoreML 不可、bucket oversize、compile/state error は HTTP 200 + PyTorch fallback + header/metrics にする。
- `require` / `prepare` / `refresh` は strict。cache miss、bucket 失敗、CoreML 不可は 4xx/5xx を維持する。
- 明示 `cache_mode=off` は完全 opt-out。cache lookup、auto prepare、CoreML fast path をしない。
- real KV residency の最低条件は、同一 server `rem.wav` の 2 回目以降で reference encode と speaker KV projection が呼ばれないこと。
- resident pool への mutation は、validation と fingerprint 確定後に行う。失敗した partial payload は pool に残さない。
- actual torch tensor residency は `InferenceRuntime` の runtime-resident store が所有する。`InMemoryCoreMLCacheManager` は metadata/index/TTL/LRU/metrics のみを担当し、tensor payload を持たない。
- `memory_bytes` は実 tensor / np array から計算する。resident accounting に `DEFAULT_NUM_LAYERS` / `DEFAULT_NUM_HEADS` 由来の見積もりを使わない。
- resident pool 導入前に manager-level lock と per-key prepare locks を入れる。runtime resident payload は immutable として扱う。
- 起動 script や launchd の変更は destructive 操作を伴わない。再起動、unload/load は実装者が明示確認後に手動で行う。

## 4. Phase Plan

### P0: doc/checkpoint baseline

**Dependencies:** なし。

**Files:** `docs/openai-default-coreml-cache-plan.md`, `openai_api_server.py`, `irodori_tts/coreml_cache.py`, `irodori_tts/inference_runtime.py`, `tests/*`。

**Work items:**

- 設計 doc と現 baseline の差分を読み、既存 function seam を一覧化する。
- 現在の status と baseline test を記録する。タグや commit はこの phase では作らない。
- 実装 branch の phase checkpoint 方針を決める。推奨 tag 名は `checkpoint/openai-default-coreml-p0`, `checkpoint/openai-default-coreml-p4-pre`, `checkpoint/openai-default-coreml-p6-pre`。

**Acceptance criteria:** 実装者が対象 file、既存 test、rollback point を把握している。source code の変更なし。

**Tests / verification commands:**

- `git status --short --branch`
- `uv run pytest tests/test_openai_api_speech_cache_extension.py tests/test_openai_api_speech_coreml_fast_path.py tests/test_coreml_cache_manager.py`
- `uv run pytest tests/test_coreml_stateful_runtime.py -k "pack_context_kv_state or split_cond1 or speaker_kv_scale"`

**Rollback/checkpoint:** ここは記録のみ。以後の risky phase 前に checkpoint tag を作る。

### P1: default cache policy + voice ignore headers

**Dependencies:** P0。

**Files:** `openai_api_server.py`, `tests/test_openai_api_speech_cache_extension.py`, `tests/test_openai_api_speech_coreml_fast_path.py`。

**Work items:**

- `_speech_cache_mode(irodori)` を `irodori is None` でも `CACHE_MODE_AUTO` にする。`irodori` はあるが `cache_mode` 欄がない場合も `normalize_speech_cache_mode(None)` 相当で AUTO に統一する。
- 明示 `cache_mode=off` だけが PyTorch legacy path になるよう `_speech_cache_id()` 呼び出し条件を維持する。
- `audio_speech()` で `voice` と client reference 系 field を ignore し続け、response header `X-Irodori-Voice-Resolved: server-default` を追加する。
- `AUTO + cache_id None` はこの phase ではまだ PyTorch fallback でもよいが、header で `X-Irodori-Cache-Auto: miss-fallback` を返せる形にする。

**Acceptance criteria:**

- `irodori` なしで `_speech_cache_mode()` は AUTO。
- `voice` 未指定、`default`、`alloy`、`nova`、任意文字列の全てが 200 で `server-default`。
- `cache_mode=off` は `X-Irodori-Denoiser-Backend: pytorch` のまま。
- `require` cache miss など strict error は緩まない。

**Likely tests:**

- `test_no_irodori_defaults_to_auto_policy`
- `test_openai_voice_values_resolve_to_server_default_header`
- `test_explicit_cache_mode_off_preserves_legacy_pytorch_path`
- 既存 `test_require_without_cache_id_returns_validation_error_before_runtime`

**Verification commands:**

- `uv run pytest tests/test_openai_api_speech_cache_extension.py tests/test_openai_api_speech_coreml_fast_path.py`
- `uv run ruff check openai_api_server.py tests/test_openai_api_speech_cache_extension.py tests/test_openai_api_speech_coreml_fast_path.py`

**Rollback/checkpoint:** P1 は小さい API policy 変更。failure 時は `_speech_cache_mode()` と added headers を戻せば切り戻せる。

### P2: auto bucket selection `S={100,160,200}` + runtime planning seam

**Dependencies:** P1。

**Files:** `openai_api_server.py`, `irodori_tts/inference_runtime.py`, new `tests/test_openai_api_auto_bucket_selection.py`。

**Work items:**

- `AutoSpeechPlanningContext`（同等名可）を追加し、cache 解決前に runtime から planning 情報を作る。含める値は segment text、normalized text、`seconds`、codec `sample_rate` / `hop_length`、`model_cfg.latent_patch_size`、tokenizer fingerprint、`token_len`、`token_ids_hash`。
- `AUTO` 既定経路では cache 解決前に runtime を load/pass して planning context を作る。`T=256` 固定の保守的 P2A は一時策としてのみ許可し、最終 default には残さない。
- `InferenceRuntime.estimate_patched_steps(seconds: float) -> int` を追加し、runtime の `codec` / `model_cfg` と同じ計算で `patched_steps` を返す。unit test 用に pure helper `_patched_steps_for_seconds(seconds, sample_rate, hop_length, latent_patch_size)` を分離してよい。
- `InferenceRuntime.tokenize_for_bucket(normalized_text: str) -> tuple[int, str]` を追加し、truncation なし `tokenizer.encode()` で token 長と `token_ids_hash` を返す。`batch_encode(max_length=...)` は切り詰めるため長さ計算に使わない。
- `_auto_select_bucket(planning: AutoSpeechPlanningContext) -> _AutoBucketResolution` を追加する。成功時は `CoreMLConditionBucket`、失敗時は oversize reason と attempted bucket metadata を返す。
- `S` は `100`, `160`, `200` の最小十分値。5 秒の基準ケースは `S=160`。
- `T` は token 長に対して `64`, `128`, `256` の最小十分値。`R=160` 固定。
- explicit `irodori.bucket` がある場合は `_condition_cache_bucket()` の既存 strict parse を使い、auto selection を bypass する。
- P2 は helper/unit tests と strict response shaping までに留める。full AUTO oversize/CoreML-unavailable fallback E2E は auto prepare と per-segment resolution 完了後の P6/P8 で扱う。

**Acceptance criteria:**

- `seconds=4.0` は通常 token 長で `S=100`。
- `seconds=5.0` は `S=160`。
- `seconds=8.0` は `S=200`。
- `tokenize_for_bucket()` は `batch_encode(max_length=...)` を使わず、truncation なし token 長と安定 `token_ids_hash` を返す。
- `patched_steps>200` と `text_token_len>256` は helper が oversize reason を返す。
- fake planning context で AUTO fallback header 用 metadata と strict 422 shaping を検証する。
- explicit bucket は auto helper を呼ばない。

**Likely tests:**

- `test_auto_select_bucket_chooses_s100_s160_s200`
- `test_auto_bucket_uses_smallest_text_bucket`
- `test_tokenize_for_bucket_uses_untruncated_encode`
- `test_estimate_patched_steps_matches_runtime_codec_and_patch_size`
- `test_auto_bucket_oversize_s_returns_reason`
- `test_auto_bucket_oversize_t_returns_reason`
- `test_auto_bucket_strict_response_shaping_with_fake_planning_context`
- `test_explicit_irodori_bucket_bypasses_auto_selection`

**Verification commands:**

- `uv run pytest tests/test_openai_api_auto_bucket_selection.py`
- `uv run pytest tests/test_openai_api_speech_cache_extension.py tests/test_openai_api_speech_coreml_fast_path.py`

**Rollback/checkpoint:** 切り戻しは `_auto_select_bucket()` integration を外し、P1 の AUTO miss fallback に戻す。

### P3: per-segment cache resolution refactor

**Dependencies:** P2。

**Files:** `openai_api_server.py`, `tests/test_openai_api_speech_segment_cache_resolution.py`, `tests/test_openai_api_speech_coreml_fast_path.py`。

**Work items:**

- `_SpeechCacheResolution` を segment 単位情報に拡張する。推奨 dataclass:
  - `condition_handle: ConditionCacheHandle | None`
  - `reference_cache_id: str | None`
  - `bucket: CoreMLConditionBucket | None`
  - `auto_status: Literal["off","hit","prepared","reused","miss-fallback"]`
  - `fallback_reason: str | None`
  - `condition_created: bool`
- `_resolve_speech_cache()` の戻り値を `list[_SpeechCacheResolution]` にする。
- `prepare` / `refresh` は当面 single segment strict のまま。AUTO と OFF は multi-segment を許容する。
- `audio_speech()` の loop は `resolutions[segment_index]` を参照し、segment ごとに CoreML/PyTorch を切り替える。
- headers:
  - `X-Irodori-Backend`: `coreml-stateful` / `pytorch` / `mixed`
  - `X-Irodori-Backend-Per-Segment`
  - `X-Irodori-Cache-Condition-Id`
  - `X-Irodori-Condition-Cache-Id`（transition alias）
  - `X-Irodori-Cache-Reference-Id`
  - `X-Irodori-Reference-Cache-Id`（transition alias）
  - `X-Irodori-Cache-Condition-Count`
  - `X-Irodori-Cache-Auto`
  - `X-Irodori-Bucket`
  - fallback 時 `X-Irodori-Fallback-Reason`, `X-Irodori-Bucket-Attempted`

**Acceptance criteria:**

- 1 segment は既存 header contract と互換。
- 2 segment で segment 1 CoreML、segment 2 fallback の mixed header が出る。
- 6 segment 以上で condition id header が要約され、count が正しい。
- `require` に single `cache_id` を付けた multi-segment は既存通り conflict。

**Likely tests:**

- `test_per_segment_resolutions_drive_backend_selection`
- `test_mixed_backend_headers_for_multi_segment_auto`
- `test_condition_id_header_summarizes_many_segments`
- `test_single_cache_id_still_rejected_for_multi_segment_require`

**Verification commands:**

- `uv run pytest tests/test_openai_api_speech_segment_cache_resolution.py tests/test_openai_api_speech_coreml_fast_path.py`
- `uv run pytest tests/test_openai_api_speech_cache_extension.py`

**Rollback/checkpoint:** P3 は call graph 変更が大きい。着手前に checkpoint tag 推奨。切り戻しは single resolution signature に戻す。

### P3A: cache manager and runtime resident concurrency hardening

**Dependencies:** P3。P4/P5/P6 より前に完了する。

**Files:** `irodori_tts/coreml_cache.py`, `irodori_tts/inference_runtime.py`, `tests/test_resident_cache_concurrency.py`, `tests/test_coreml_cache_manager.py`。

**Work items:**

- `InMemoryCoreMLCacheManager` に manager-level lock と per-key prepare locks を追加する。
- lock 対象は reference/condition metadata、resident metadata、metrics、delete/evict/prune expired 経路。
- prepare は validation 後に per-key lock を取り、二重 prepare では既存 handle を reuse する。
- `InferenceRuntime` の runtime-resident store も per-key locks を持つ。payload は immutable として公開し、prepare 中の partial payload は登録しない。

**Acceptance criteria:**

- concurrent same-key prepare は 1 件だけ materialize される。
- delete/evict/prune と metrics snapshot が並行しても `KeyError` や不整合を起こさない。
- runtime resident payload は shared entry を in-place mutate しない。

### P4: resident `rem.wav` reference latent/speaker state layers 1+2

**Dependencies:** P3A。

**Files:** `irodori_tts/coreml_cache.py`, `irodori_tts/inference_runtime.py`, `openai_api_server.py`, new `tests/test_resident_reference_pool.py`, `tests/test_audio_speech_default_coreml.py`。

**Work items:**

- `InferenceRuntime` に runtime-resident `ResidentReferenceTensors` store を追加する。保持対象は `ref_latent`, `ref_mask`, `speaker_state`, `speaker_mask`, device/dtype metadata、実 tensor 由来 `memory_bytes`、created_at/last_used。
- `InMemoryCoreMLCacheManager` は tensor を持たない。reference handle の resident key/layers、materialized layer names、`memory_bytes`、last_used だけを metadata として追跡する。
- `InferenceRuntime.prepare_default_reference_tensors(ref_wav, max_ref_seconds) -> ResidentReferenceTensors` を追加し、既存 `_load_reference_latent()` と speaker encoder の処理を再利用する。
- default rem.wav resident prepare 後、runtime が実測した `ref_len`、`speaker_context_len`、`speaker_dim`、`memory_bytes` で `ReferenceCacheRequest` / `ReferenceCacheHandle` を作成または更新する。`_reference_cache_request()` の既定 `speaker_context_len=1` は resident default cache では使わない。
- `lifespan()` で `_validate_reference_wav()` 後に runtime resident layers 1+2 と cache metadata を eager 生成する。`rem.wav` 不正、reference tensor 生成失敗は起動中止。
- `synthesize_with_condition_cache()` は resident parts を明示引数で受け取るか、runtime-resident store から reference id で lookup する。metadata-only manager から tensor を取り出さない。
- reference delete/evict 時は API 層または runtime hook が resident key を runtime store から prune する。cache manager 自体は tensor payload を削除しない。
- 計測用 counter を入れる。例: `reference_encode_calls`, `resident_reference_hits`。

**Acceptance criteria:**

- lifespan 後に default reference cache id と resident layers 1+2 が存在する。
- `ReferenceCacheRequest` / `ReferenceCacheHandle` は runtime 実測の `ref_len`、`speaker_context_len`、`speaker_dim`、実 tensor 由来 `memory_bytes` を持つ。
- 同一 `rem.wav` の 2 回目以降で reference latent/speaker state 生成が skip される。
- fingerprint 不一致時は resident hit しない。
- metrics endpoint は metadata と `memory_bytes` のみを返し、tensor payload を返さない。
- reference cache 削除時に runtime resident entry も prune される。

**Likely tests:**

- `test_lifespan_prepares_default_reference_resident_layers`
- `test_default_reference_handle_uses_runtime_measured_metadata`
- `test_runtime_resident_reference_store_put_get_delete_cascade`
- `test_pack_prepare_reuses_resident_reference_tensors_on_second_request`
- `test_cache_metrics_omits_resident_tensor_payloads`
- `test_lifespan_fails_when_default_reference_prepare_fails`

**Verification commands:**

- `uv run pytest tests/test_resident_reference_pool.py tests/test_audio_speech_default_coreml.py`
- `uv run pytest tests/test_coreml_cache_manager.py tests/test_openai_api_coreml_cache_endpoints.py`

**Rollback/checkpoint:** risky phase。開始前に checkpoint tag 推奨。問題時は resident lookup を feature flag で off にして既存 `_load_reference_latent()` に戻す。

### P5: bucketed speaker KV resident pool layer 3

**Dependencies:** P4。

**Files:** `irodori_tts/coreml_cache.py`, `irodori_tts/inference_runtime.py`, `openai_api_server.py`, new `tests/test_resident_speaker_kv_pool.py`, `tests/test_coreml_stateful_runtime.py` additions。

**Work items:**

- `InferenceRuntime` の runtime-resident store に `ResidentSpeakerKVPayload` を追加する。key は `(model_fingerprint, reference_fingerprint, S_bucket, R_bucket, speaker_kv_scale_signature, branch_layouts)`。
- `InMemoryCoreMLCacheManager` は speaker KV payload を持たず、resident speaker KV metadata、LRU order、materialized layer names、実 payload 由来 `memory_bytes` だけを追跡する。
- `InferenceRuntime` / model に speaker-only KV projection helper を追加し、text projection を払わず speaker KV を作る。既存 `build_context_kv_cache()` が text/speaker 一括投影だけなので、explicit text/speaker projection helpers か clean runtime/model helper を入れる。
- zero-text から speaker KV を推定する workaround は一時診断に限る。production acceptance には含めない。
- fresh text KV と resident speaker KV を merge してから `pack_context_kv_state()` に渡す helper を追加する。
- `scale_speaker_kv_cache()` 適用済み variant は scale signature 別 entry、または clone/copy payload とする。shared resident speaker KV を in-place scale しない。
- warmup bucket ごとに resident speaker KV を precompute する。最低 `S=100` と `S=160`、可能なら `S=200`。
- `synthesize_with_condition_cache()` の `_pack_and_prepare()` で resident speaker KV hit 時に speaker KV projection を呼ばない。

**Acceptance criteria:**

- 同一 `rem.wav` + `S=160,R=160` の 2 回目で speaker KV projection が skip される。これを real KV residency の最低ラインとする。
- speaker-only helper は text projection を呼ばない。
- fresh text KV + resident speaker KV merge の shape/mask が `pack_context_kv_state()` と一致する。
- `S=100` と `S=160` は別 entry。
- `speaker_kv_scale` 有無と scale 値で別 entry。
- scaled variant は shared payload を in-place mutate しない。
- max entries 超過で LRU eviction。

**Likely tests:**

- `test_resident_speaker_kv_key_includes_bucket_and_scale_signature`
- `test_speaker_only_projection_does_not_project_text`
- `test_merge_fresh_text_kv_with_resident_speaker_kv_before_pack`
- `test_resident_speaker_kv_lru_evicts_oldest_bucket`
- `test_speaker_kv_scale_does_not_mutate_shared_payload`
- `test_pack_prepare_reuses_speaker_kv_projection_for_same_bucket`
- `test_bucketed_speaker_kv_deleted_with_reference_cache`

**Verification commands:**

- `uv run pytest tests/test_resident_speaker_kv_pool.py tests/test_coreml_stateful_runtime.py -k speaker_kv`
- `uv run pytest tests/test_audio_speech_default_coreml.py`

**Rollback/checkpoint:** P5 前に checkpoint tag 推奨。feature flag `enable_resident_speaker_kv` で off にできるようにする。

### P6: condition cache auto-prepare/reuse and optional packed-KV layer 4

**Dependencies:** P5。

**Files:** `openai_api_server.py`, `irodori_tts/coreml_cache.py`, `irodori_tts/inference_runtime.py`, `tests/test_condition_cache_fingerprint.py`, `tests/test_audio_speech_default_coreml.py`, `tests/test_openai_api_default_cache_fallbacks.py`。

**Work items:**

- explicit `/v1/tts/condition-caches` は現行 canonical fingerprint を維持する。
- internal AUTO は fingerprint v2 helper を追加する。要素は `token_ids_hash`、bucket `S/T/R`、`cfg_signature`、`branch_layouts`、`model_fingerprint`、`tokenizer_fingerprint`、`default_reference_fingerprint`、`speaker_kv_scale_signature`。raw `seconds` は含めない。
- requested `seconds` は request/runtime data として保持し、同じ bucket に入る `seconds=4.0` と `4.05` が同 condition KV identity を hit できるようにする。
- `InMemoryCoreMLCacheManager.find_condition_cache_by_fingerprint(...)` または `condition_cache_id_for_request()` を使った metadata lookup helper を追加する。
- `AUTO + cache_id None` は segment ごとに:
  - existing condition cache hit -> `auto_status=hit`
  - miss -> `prepare_condition_cache(create_or_reuse)` -> `prepared` or `reused`
  - CoreML/bucket failure -> PyTorch fallback
- optional layer 4 として runtime-resident `ResidentConditionPackedKV` pool を追加する。key は AUTO fingerprint v2。`InMemoryCoreMLCacheManager` は payload ではなく resident metadata / LRU / memory_bytes のみを追跡する。
- layer 4 hit 時は text encode + text KV projection + pack を skip し、MLState への `prepare_state()` のみ行う。
- `synthesize_with_condition_cache()` は layer 3/4 resident parts を明示引数または runtime lookup で受け取る。
- strict modes は例外を swallow しない。

**Acceptance criteria:**

- `irodori` なしの同一 text/seconds 連続 2 回で、1 回目 `X-Irodori-Cache-Auto: prepared`、2 回目 `hit`。
- `seconds=4.0` と `4.05` は同 bucket なら同 condition cache を hit。
- explicit `/v1/tts/condition-caches` の canonical fingerprint 互換性は変わらない。
- AUTO fingerprint v2 は bucket-normalized で、raw seconds を含まない。
- cfg scale/window 変更、caption 変更、tokenizer fingerprint 変更は別 cache。
- layer 4 enabled 構成では 2 回目に text encode + KV projection が skip される。
- AUTO CoreML unavailable は 200 fallback、`require` / `prepare` / `refresh` は strict error。

**Likely tests:**

- `test_auto_without_cache_id_prepares_then_hits_condition_cache`
- `test_auto_condition_cache_quantizes_seconds_by_bucket`
- `test_explicit_condition_cache_fingerprint_v1_remains_compatible`
- `test_auto_condition_fingerprint_v2_uses_token_hash_bucket_and_no_raw_seconds`
- `test_auto_condition_cache_cfg_signature_separates_entries`
- `test_layer4_packed_kv_hit_skips_text_encode_when_enabled`
- `test_auto_coreml_unavailable_returns_200_with_fallback_header`
- `test_prepare_refresh_require_remain_strict_on_coreml_failure`

**Verification commands:**

- `uv run pytest tests/test_condition_cache_fingerprint.py tests/test_audio_speech_default_coreml.py tests/test_openai_api_default_cache_fallbacks.py`
- `uv run pytest tests/test_openai_api_speech_coreml_fast_path.py tests/test_openai_api_coreml_cache_endpoints.py`

**Rollback/checkpoint:** P6 は behavior 変更の中心。開始前に checkpoint tag 推奨。`auto_prepare_default_cache=False` と `enable_condition_packed_kv_cache=False` で fallback 可能にする。

### P7: launchd/start script warmup defaults

**Dependencies:** P6。

**Files:** `openai_api_server.py`, `start_openai_server.sh`, `launchd/com.ramo.irodori-tts-openai-api.plist`, `tests/test_openai_server_settings_cli.py` または existing CLI tests。

**Work items:**

- `ServerSettings` と CLI に追加:
  - `auto_prepare_default_cache: bool = True`
  - `default_reference_cache_prepare: bool = True`
  - `default_condition_cache_prepare_text: str | None = None`
  - `max_resident_speaker_kv_buckets: int`
  - `max_resident_condition_cache_entries: int`
  - `condition_cache_default_ttl_seconds: float`
  - `strict_coreml: bool = False`
  - `enable_resident_reference_cache: bool = True`
  - `enable_resident_speaker_kv: bool = True`
  - `enable_condition_packed_kv_cache: bool = False` initially unless P6 makes it stable
- env aliases:
  - `IRODORI_AUTO_PREPARE_DEFAULT_CACHE`
  - `IRODORI_STRICT_COREML`
  - `IRODORI_MAX_RESIDENT_SPEAKER_KV_BUCKETS`
  - `IRODORI_MAX_RESIDENT_CONDITION_CACHE_ENTRIES`
- `start_openai_server.sh` は default warmup args を追加する。ただし user-supplied `$@` で override 可能にする。
- launchd plist に `--preload`, `--warmup-bucket S=100,T=256,R=160`, `--warmup-bucket S=160,T=256,R=160`, `--warmup-bucket S=200,T=256,R=160` を追加する。
- `lifespan()` warmup order を固定する: precompile cond1 -> reference resident -> speaker KV resident -> optional condition warmup。

**Acceptance criteria:**

- CLI parse で追加 field が `ServerSettings` に入る。
- start script と launchd は warmup bucket 3 件を含む。
- startup log に reference prepared、bucket precompiled、speaker KV resident count が出る。
- warmup failure は設計通り。reference failure は fatal、speaker KV/condition warmup は warning and continue。

**Likely tests:**

- `test_cli_defaults_enable_auto_prepare_and_reference_prepare`
- `test_parse_boolean_env_for_auto_prepare_default_cache`
- `test_lifespan_warmup_order_precompile_reference_speaker_kv_condition`
- `test_speaker_kv_warmup_failure_logs_warning_and_continues`

**Verification commands:**

- `uv run pytest tests/test_openai_server_settings_cli.py tests/test_audio_speech_default_coreml.py`
- `bash -n start_openai_server.sh`
- `plutil -lint launchd/com.ramo.irodori-tts-openai-api.plist`

**Rollback/checkpoint:** script/plist 変更は launchd 操作なしで commit 可能。運用反映前に manual diff review。

### P8: metrics/observability polish

**Dependencies:** P6, P7。

**Files:** `openai_api_server.py`, `irodori_tts/coreml_cache.py`, `irodori_tts/inference_runtime.py`, `tests/test_openai_api_observability_headers.py`, `tests/test_coreml_cache_manager.py`。

**Work items:**

- response headers を最終 contract に揃える:
  - `X-Irodori-Backend`
  - `X-Irodori-Denoiser-Backend` は互換 alias として当面維持
  - `X-Irodori-Backend-Per-Segment`
  - `X-Irodori-Cache-Auto`
  - `X-Irodori-Bucket`
  - `X-Irodori-Cache-Reference-Id`
  - `X-Irodori-Reference-Cache-Id` は互換 alias として当面維持
  - `X-Irodori-Cache-Condition-Id`
  - `X-Irodori-Condition-Cache-Id` は互換 alias として当面維持
  - `X-Irodori-Cache-Condition-Count`
  - `X-Irodori-Voice-Resolved`
  - `X-Irodori-Fallback-Reason`
  - `X-Irodori-Bucket-Attempted`
- metrics snapshot に追加:
  - `irodori_default_cache_auto_total{outcome}`
  - `irodori_bucket_resolution_total{outcome}`
  - `irodori_resident_layer_present{layer}`
  - `irodori_coreml_silent_fallback_total{reason}`
- cache metrics endpoint が runtime load を起こさないことを維持する。

**Acceptance criteria:**

- hit/prepared/reused/fallback/off/mixed の header 値が deterministic。
- fallback reason は `bucket_oversize_s`, `bucket_oversize_t`, `coreml_unavailable`, `mlmodel_compile_error`, `state_alloc_error`, `strict_coreml` などの列挙値。
- metrics endpoint は resident pool 状態を返すが、大きい tensor payload は返さない。
- `X-Irodori-Condition-Cache-Id` / `X-Irodori-Reference-Cache-Id` / `X-Irodori-Denoiser-Backend` は transition alias として新ヘッダと同値になる。

**Likely tests:**

- `test_headers_for_auto_prepared_hit_fallback_and_off`
- `test_backend_alias_header_preserved_for_compatibility`
- `test_metrics_snapshot_includes_default_cache_auto_outcomes`
- `test_metrics_endpoint_omits_resident_tensor_payloads`

**Verification commands:**

- `uv run pytest tests/test_openai_api_observability_headers.py tests/test_coreml_cache_manager.py`
- `uv run pytest tests/test_openai_api_coreml_cache_endpoints.py::test_cache_metrics_endpoint_does_not_load_runtime`

**Rollback/checkpoint:** header additions are additive。breaking rename はしない。

### P9: real device E2E validation

**Dependencies:** P8。

**Files:** source changes complete; validation artifacts under logs only. No source edits required unless failures are found。

**Work items:**

- Apple Silicon 実機で server を起動し、4 秒、5 秒、8 秒、oversize を curl で検証する。
- cold と warm の latency、headers、fallback reason 不在、memory resident count を記録する。
- `B=2/B=3` backend に戻っていないことを code search と runtime log で確認する。
- explicit `cache_mode=off` と `require` strict の E2E を確認する。

**Acceptance criteria:**

- `irodori` なし 5 秒 request が `S=160`、`Backend: coreml-stateful`、fallback reason なしで 200。
- 同一 2 回目 request で resident reference hit と speaker KV hit が観測できる。
- CoreML unavailable をシミュレートした AUTO は 200 PyTorch fallback。
- `require` miss は 4xx、`prepare` bucket oversize は 422、`off` は PyTorch。

**Smoke commands:**

```bash
./start_openai_server.sh --host 127.0.0.1 --port 19841 --preload \
  --warmup-bucket S=100,T=256,R=160 \
  --warmup-bucket S=160,T=256,R=160 \
  --warmup-bucket S=200,T=256,R=160
```

```bash
curl -sS -D /tmp/irodori-h5.txt -o /tmp/irodori-5s.wav \
  http://127.0.0.1:19841/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"alloy","input":"今日は五秒くらいのテスト音声です。","response_format":"wav","seconds":5.0}'
```

```bash
cat /tmp/irodori-h5.txt | rg 'X-Irodori-(Backend|Cache|Bucket|Fallback|Voice)'
```

```bash
curl -sS -D /tmp/irodori-off.txt -o /tmp/irodori-off.wav \
  http://127.0.0.1:19841/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"tts-1","voice":"nova","input":"明示的にオフにします。","response_format":"wav","seconds":4.0,"irodori":{"cache_mode":"off"}}'
```

**Rollback/checkpoint:** launchd/server restart は destructive ではないが運用影響がある。実行前に user approval を得る。failure 時は `--auto-prepare-default-cache=false` または `cache_mode=off` で回避する。

## 5. Detailed Test Matrix

**Unit tests:**

- `tests/test_openai_api_speech_cache_extension.py`
  - `_speech_cache_mode()` default AUTO。
  - voice ignore header。
  - explicit `off` preserves PyTorch。
- `tests/test_openai_api_auto_bucket_selection.py`
  - S/T/R selection table。
  - oversize S/T。
  - `tokenize_for_bucket()` の truncation なし encode と `token_ids_hash`。
  - fake planning context による strict response shaping。
  - explicit bucket bypass。
- `tests/test_condition_cache_fingerprint.py`
  - bucket quantization。
  - cfg signature separation。
  - tokenizer/text hash separation。
- `tests/test_resident_reference_pool.py`
  - layer 1+2 put/get/delete/fingerprint miss。
- `tests/test_resident_speaker_kv_pool.py`
  - layer 3 key, LRU, scale signature, cascade delete。
- `tests/test_coreml_stateful_runtime.py`
  - split cond1 remains active。
  - packed state shape validation。
  - speaker KV reuse counters。

**FastAPI/TestClient integration:**

- `tests/test_audio_speech_default_coreml.py`
  - `irodori` なし 4s -> CoreML `S=100`。
  - `irodori` なし 5s -> CoreML `S=160`。
  - same payload twice -> `prepared` then `hit`。
  - voice variants -> `server-default`。
- `tests/test_openai_api_default_cache_fallbacks.py`
  - bucket oversize AUTO -> 200 PyTorch fallback。
  - CoreML unavailable AUTO -> 200 fallback。
  - strict modes -> 4xx/5xx。
- `tests/test_openai_api_speech_segment_cache_resolution.py`
  - mixed per-segment backend。
  - condition id summarization。

**Concurrency tests:**

- `tests/test_resident_cache_concurrency.py`
  - manager-level lock が reference/condition metadata、metrics、delete/evict/prune を保護する。
  - concurrent AUTO same fingerprint prepares exactly one resident entry。
  - concurrent reads do not mutate tensors。
  - LRU eviction does not remove leased/in-use payload。
  - metrics counters remain consistent under parallel requests。

**Real CoreML smoke commands:**

- `./start_openai_server.sh --preload --warmup-bucket S=100,T=256,R=160 --warmup-bucket S=160,T=256,R=160 --warmup-bucket S=200,T=256,R=160`
- curl 4s/5s/8s requests with `voice=alloy` and no `irodori`。
- repeat same 5s request and compare `X-Irodori-Cache-Auto` plus resident hit metrics。
- curl explicit `cache_mode=off`。
- curl `cache_mode=require` with missing cache id。

**Failure/fallback tests:**

- missing CoreML backend method -> AUTO 200 fallback, strict 503。
- bucket oversize S/T -> AUTO 200 fallback, strict 422。
- `rem.wav` missing/corrupt at startup -> fatal lifespan failure。
- speaker KV warmup failure -> warning and server starts。
- resident pool memory limit -> LRU eviction and no partial mutation。

## 6. Data Structures / APIs to Add or Change

**`ServerSettings` additions:**

- `auto_prepare_default_cache: bool = True`
- `strict_coreml: bool = False`
- `default_reference_cache_prepare: bool = True`
- `default_condition_cache_prepare_text: str | None = None`
- `condition_cache_default_ttl_seconds: float = 86400.0`
- `max_resident_speaker_kv_buckets: int = 3`
- `max_resident_condition_cache_entries: int = 128`
- `enable_resident_reference_cache: bool = True`
- `enable_resident_speaker_kv: bool = True`
- `enable_condition_packed_kv_cache: bool = False`

**Runtime-resident payloads:**

- `ResidentReferenceTensors`
- `ResidentSpeakerKVKey`
- `ResidentSpeakerKVPayload`
- `ResidentConditionPackedKVKey`
- `ResidentConditionPackedKVPayload`
- `InferenceRuntime` owned put/get/delete helpers with per-key locks and immutable payloads。
- `memory_bytes` from actual tensor / np array payloads。

**Cache manager metadata/index:**

- `manager-level lock` and per-key prepare locks。
- `find_condition_cache_by_fingerprint(...)` or equivalent metadata lookup。
- resident key/layer names, last_used, TTL/LRU, metrics, actual `memory_bytes`。
- no torch tensor / np array payload ownership。

**Runtime helper methods:**

- `InferenceRuntime.estimate_patched_steps(seconds: float) -> int`
- `InferenceRuntime.tokenize_for_bucket(normalized_text: str) -> tuple[int, str]`
- `InferenceRuntime.prepare_default_reference_tensors(...)`
- `InferenceRuntime.prepare_resident_speaker_kv(...)`
- `InferenceRuntime.project_speaker_kv_only(...)` or clean equivalent
- `InferenceRuntime.merge_text_kv_with_resident_speaker_kv(...)`
- `InferenceRuntime.pack_condition_state_from_resident_parts(...)`
- Optional: `InferenceRuntime.condition_cache_prepare_counters_snapshot()`

**Per-segment resolution dataclass/list:**

- `_resolve_speech_cache(...) -> list[_SpeechCacheResolution]`
- `_SpeechCacheResolution.condition_handle`
- `_SpeechCacheResolution.reference_cache_id`
- `_SpeechCacheResolution.bucket`
- `_SpeechCacheResolution.auto_status`
- `_SpeechCacheResolution.fallback_reason`
- `_SpeechCacheResolution.condition_created`

**Response headers:**

- `X-Irodori-Backend`
- `X-Irodori-Denoiser-Backend` as compatibility alias
- `X-Irodori-Backend-Per-Segment`
- `X-Irodori-Cache-Auto`
- `X-Irodori-Bucket`
- `X-Irodori-Cache-Reference-Id`
- `X-Irodori-Reference-Cache-Id` as compatibility alias
- `X-Irodori-Cache-Condition-Id`
- `X-Irodori-Condition-Cache-Id` as compatibility alias
- `X-Irodori-Cache-Condition-Count`
- `X-Irodori-Voice-Resolved`
- `X-Irodori-Fallback-Reason`
- `X-Irodori-Bucket-Attempted`

## 7. Risk Controls

- Create checkpoint tags before P3, P4, P5, and P6. Do not mix phase changes in one large commit.
- Keep feature flags for resident reference, resident speaker KV, packed-KV layer 4, auto prepare, and strict CoreML.
- Memory budget guards:
  - cap speaker KV bucket entries;
  - cap packed-KV entries;
  - track memory_bytes from actual tensor / np array payloads;
  - never return tensor payloads from metrics endpoints。
- Avoid mutation before validation:
  - validate bucket, fingerprint, cfg, reference id, tensor shape, dtype, device;
  - only then write to resident pools;
  - failed prepare must not leave handle/payload mismatch。
- No reboot/server destructive operations in implementation. launchd unload/load or server restart is manual validation only.
- Concurrency:
  - manager-level lock for metadata/metrics/delete/evict/prune;
  - per-key lock for prepare;
  - immutable resident payloads;
  - no in-place scaling of shared speaker KV; scaled variants must copy or have separate key。
- Compatibility:
  - keep `X-Irodori-Denoiser-Backend` while adding `X-Irodori-Backend`;
  - keep `X-Irodori-Condition-Cache-Id` and `X-Irodori-Reference-Cache-Id` while adding `X-Irodori-Cache-Condition-Id` and `X-Irodori-Cache-Reference-Id`;
  - strict modes remain strict;
  - explicit `cache_mode=off` remains PyTorch。

## 8. Definition of Done

- `POST /v1/audio/speech` with no `irodori` and any OpenAI `voice` uses server `rem.wav`, returns 200, and normally uses CoreML stateful fast path.
- 5 second default request auto-selects `S=160,T<=256,R=160` and does not fail due to the old `S=100` default.
- AUTO CoreML/bucket failures return 200 PyTorch fallback with deterministic fallback headers and metrics.
- `require` / `prepare` / `refresh` strict semantics and explicit `off` behavior are covered by tests.
- Same text/seconds repeated request hits condition cache. At minimum it skips `rem.wav` reference encode and speaker KV projection.
- Multi-segment requests can mix CoreML and PyTorch per segment with correct headers.
- Header transition emits both old and new cache id names, with `X-Irodori-Denoiser-Backend` retained as backend alias.
- start script and launchd defaults warm up `{100,160,200}` cond1 buckets and resident default reference/speaker KV.
- Unit, TestClient integration, concurrency, and real CoreML smoke tests pass or have documented environment blockers.

## 9. Suggested First Implementation Prompt for a Coding Agent

Implement P1 and P2 only. Work in `/Users/ramo/Services/Irodori-TTS`. Do not introduce resident tensor pools yet.

Tasks:

1. In `openai_api_server.py`, make `_speech_cache_mode()` default to `CACHE_MODE_AUTO` when `irodori` is absent or `cache_mode` is omitted, while preserving explicit `cache_mode=off`.
2. Keep OpenAI `voice` and client reference fields ignored, and add `X-Irodori-Voice-Resolved: server-default` to `/v1/audio/speech` responses.
3. Add `AutoSpeechPlanningContext`, `InferenceRuntime.estimate_patched_steps()`, and `InferenceRuntime.tokenize_for_bucket()`. Token length must come from truncation-free `tokenizer.encode()`, not `batch_encode(max_length=...)`.
4. Add auto bucket selection helpers for `S={100,160,200}`, `T={64,128,256}`, `R=160`. Use runtime planning context. Explicit `irodori.bucket` must bypass auto selection.
5. For P2, return structured oversize metadata and cover strict response shaping with fake planning context. Full AUTO oversize/CoreML-unavailable fallback E2E belongs to the later auto prepare/per-segment phases.
6. Add focused tests in `tests/test_openai_api_speech_cache_extension.py`, `tests/test_openai_api_speech_coreml_fast_path.py`, and new `tests/test_openai_api_auto_bucket_selection.py`.

Verification:

```bash
uv run pytest tests/test_openai_api_speech_cache_extension.py tests/test_openai_api_speech_coreml_fast_path.py tests/test_openai_api_auto_bucket_selection.py
uv run ruff check openai_api_server.py tests/test_openai_api_speech_cache_extension.py tests/test_openai_api_speech_coreml_fast_path.py tests/test_openai_api_auto_bucket_selection.py
```
