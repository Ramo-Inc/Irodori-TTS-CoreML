# OpenAI 互換エンドポイントの CoreML 既定経路化 設計プラン

## 1. Title and Objective

**Title:** OpenAI-compatible `/v1/audio/speech` を、`irodori` 拡張に依存せず既定で CoreML stateful fast path に乗せる。

**Objective:**

P4/P5 で per-layer stateful CoreML denoiser（`per_layer_text_speaker_context_v1`）が production baseline として PASS したが、`irodori` フィールドを持たない素の OpenAI クライアント（`POST /v1/audio/speech`）は依然として `cache_mode=off` として PyTorch 経路に落ちる。
本プランは、追加クライアント実装なしでの既定動作を「サーバ既定 rem.wav 参照 + 自動 bucket 選択 + condition cache の自動 prepare/再利用 + CoreML stateful 実行」とし、明示的に旧経路を要求した場合のみ PyTorch を使う形に揃えることを目的とする。

## 2. Current Behavior (code-level references)

以下は `develop` ブランチ時点の実体で、本プランは要件として固定する。

### 2.1 `cache_mode` の既定値が `off`

`openai_api_server.py::_speech_cache_mode()` は `irodori` オブジェクト不在時に `CACHE_MODE_OFF` を返す。`audio_speech()` はこの値を `_resolve_speech_cache()` に渡し、`condition_cache_handle` は `None` のままになるため、`audio_speech()` 内の `denoiser_backend = "coreml-stateful" if condition_cache_handle is not None else "pytorch"` 分岐で常に `pytorch` 側へ落ちる。

### 2.2 `cache_mode=auto` で `cache_id` 不在時は PyTorch にフォールバックする

`openai_api_server.py::_resolve_speech_cache()` は `cache_mode in {AUTO, REQUIRE}` で `cache_id` がない／一致しない場合、AUTO は `_SpeechCacheResolution(None, None, False, False)` を返す（= cache miss としてサイレントに PyTorch 経路へ）。
これは「明示 cache_id を渡さない限り CoreML を踏まない」という現在仕様を意味する。`prepare`/`refresh` のみ自発的に condition cache を作る。

### 2.3 既定 bucket は S=100/T=256/R=160 で 5 秒生成で破綻する

`openai_api_server.py::_condition_cache_bucket()` の既定値は `sequence_length=100, text_len=256, speaker_context_len=160` で、`InferenceRuntime.synthesize_with_condition_cache()` 内の `irodori_tts/inference_runtime.py` で `patched_steps = math.ceil(latent_steps / latent_patch_size)` が計算され、`patched_steps > bucket_steps` の場合は `CoreMLStatefulUnavailableError("requested patched_steps {N} exceeds CoreML condition cache bucket {S}")` を投げる。
`seconds=5.0` の典型 latent では `patched_steps=125` となり、ユーザが明示的に `bucket.sequence_length=160` 等を渡さない限り stateful 経路が prepare 段階または synthesize 段階で失敗する。

### 2.4 rem.wav は「既定パス」止まりで residency なし

`openai_api_server.py::_make_settings()` で `--reference-wav` の既定 `"rem.wav"` を `settings.reference_wav` に正規化保存する。`lifespan()` は `_validate_reference_wav()` でファイル存在のみ確認する。
`InMemoryCoreMLCacheManager` （`irodori_tts/coreml_cache.py`）は `ReferenceCacheHandle` / `ConditionCacheHandle` を **メタデータのみ** 保持し、`ref_latent` / `speaker_state` / `speaker_kv` の実テンソルは保持しない。実体は毎リクエスト `InferenceRuntime` の `_pack_and_prepare()` 経路（`encode_conditions()` → `build_context_kv_cache()` → `pack_context_kv_state()`）でその都度組み直され、`CoreMLStatefulDenoiserBackend.prepare_state()` / `make_state()` で `MLState` に書き込まれる。
すなわち、既定参照 rem.wav に対して reference cache / speaker KV / condition packed-KV のいずれも startup 時に常駐していない。

### 2.5 起動時 warmup は bucket precompile のみ

`lifespan()` は `settings.warmup_buckets` がある場合に限り `runtime.precompile_coreml_stateful_buckets(...)`（`inference_runtime.py`）を呼び、`cond1` の `mlpackage` 変換のみ行う。condition cache 自体は warmup されない。

### 2.6 split cond1 戦略と branch layout のメタデータ拡張

`coreml_cache.py` の branch layout 定数は現状以下を含む：`BRANCH_LAYOUT_COND1`、`BRANCH_LAYOUT_INDEPENDENT_TEXT_SPEAKER3`、`BRANCH_LAYOUT_JOINT2`、`BRANCH_LAYOUT_ALTERNATING_TEXT2`、`BRANCH_LAYOUT_ALTERNATING_SPEAKER2`。
ただし、`joint2` / `alternating_text2` / `alternating_speaker2` は **API 契約 / メタデータのスキーマ定義** に留まり、実機 runtime では 1 mlmodel に B>=2 を載せた CoreML 実行計画が `-14`（execution plan failure）を再現した経緯から、CFG active step は **B=1 cond1 を 3 回連続実行** する split cond1 戦略のみが実行パスとして稼働している（`independent_text_speaker3` も同様に「3 個の B=1 cond1 を直列実行」というメタ表現で、内部実装は split cond1）。本プランも split cond1 を runtime 前提として維持し、`joint2` / `alternating_*` の実行パスは導入しない。

## 3. Desired Default Contract for OpenAI-Compatible Clients

OpenAI 互換クライアント（`POST /v1/audio/speech`）は、`irodori` フィールド一切なしで以下を満たす。

### 3.1 Voice / Reference handling（明示固定）

- 本プラン範囲では、サーバは **常にサーバ起動時に固定された既定 reference（rem.wav）** のみを使う。
- 受信側で以下はすべて **無視** する：
  - OpenAI の `voice` パラメータ（`"alloy"` / `"echo"` / `"nova"` / `"default"` 等、未指定含むすべての値）。
  - 任意の client-side reference 指定（`reference_url` / `reference_audio` / `irodori.reference` 等の入力）。
- 結果として、`voice` 未指定 / `voice="default"` / `voice="alloy"` / `voice="<任意>"` は **すべて rem.wav にマップされる**。
- 将来 server-side voice registry（複数 voice の preload + voice→reference のサーバ内マッピング）を実装するまで、この振る舞いを変更しない（§8）。
- API 仕様上のエラーにはしない（`voice` を送ってきても 4xx は出さない）が、ヘッダ `X-Irodori-Voice-Resolved: server-default` で観測可能にする。

### 3.2 Cache mode / fast path

1. `cache_mode` が **未指定** の場合の既定値を `auto` に変更する（従来 `off`）。
2. `auto` ＋ `cache_id` 不在のとき、サーバが「現在の text/seconds/cfg/bucket に合致する condition cache を自動準備または再利用」する。準備失敗（unsupported bucket、CoreML 利用不可、mlmodel compile error 等）は **silent fallback to PyTorch** とし HTTP 200 を保つ。
3. Bucket は `seconds` と tokenized `text_len` から自動選択する（§4.2）。
4. `cache_mode=off` を明示した場合のみ、現状の PyTorch 経路をそのまま使う（後方互換）。
5. `cache_mode=require` は厳格運用継続（cache miss/mismatch / bucket 失敗で 4xx）。
6. `cache_mode=prepare`/`refresh` の意味は維持（bucket 失敗・CoreML 失敗で 4xx/5xx）。
7. binary response の contract（`audio/mpeg`, `audio/wav` など）は変更しない。`X-Irodori-*` ヘッダで観測情報のみ追加する。

## 4. Proposed Architecture

### 4.1 Default cache policy normalization

- `_speech_cache_mode()` の `irodori is None` ブランチを `CACHE_MODE_AUTO` に変更する。
- 同時に `irodori` が存在し `cache_mode` だけ未指定の場合も `AUTO` 既定にする。
- `AUTO` 既定経路では cache 解決前に runtime planning 情報を取得する。`AutoSpeechPlanningContext`（名称は同等なら可）を導入し、segment text、正規化済み text、`seconds`、codec `sample_rate` / `hop_length`、`model_cfg.latent_patch_size`、tokenizer fingerprint、`token_len`、`token_ids_hash` を持たせる。
- `AutoSpeechPlanningContext` は `InferenceRuntime` から作る。runtime がまだロードされていない場合は AUTO prepare の前に runtime をロードする。`T=256` 固定などの保守的 fallback は P2A の一時策に限り、最終既定経路には残さない。
- `_resolve_speech_cache()` の `cache_mode == AUTO and cache_id is None` のブランチを「サイレント miss」から「自動 prepare/再利用」に変更する。具体的には fingerprint で resident pool / cache manager をルックアップし、ヒットすれば handle を返す、ミスなら `_prepare_or_refresh_speech_cache()` を `cache_mode=PREPARE` 同等で内部呼び出しする。
- 失敗時の動作は `cache_mode` 別に分岐（§4.6 表）：
  - `AUTO`：例外 swallow → segment ごとに `condition_cache_handle = None` → PyTorch 経路。observability にカウントだけ残す。
  - `PREPARE` / `REFRESH`：HTTP 4xx/5xx をそのまま返す（既存挙動維持）。
  - `REQUIRE`：従来どおり厳格。
  - `OFF`：何もせず PyTorch。
- `settings.auto_prepare_default_cache: bool`（CLI `--auto-prepare-default-cache` / env `IRODORI_AUTO_PREPARE_DEFAULT_CACHE`）。
  - **既定値は `True`**（通常 server / `start_openai_server.sh` / launchd plist いずれも True で起動）。
  - 障害時の opt-out / 安全装置として `False` を選べる、というだけの位置付けであり、既定値は固定。
  - opt-out された場合の挙動は「`AUTO` の暗黙 prepare を行わず、cache_id ヒットのみ CoreML、それ以外 PyTorch silent fallback」。

### 4.2 Automatic bucket selection

- 新規ヘルパー `_auto_select_bucket(planning: AutoSpeechPlanningContext) -> CoreMLConditionBucket` を `openai_api_server.py` に追加する。
- runtime helper を追加する：
  - `InferenceRuntime.estimate_patched_steps(seconds: float) -> int`：codec `sample_rate` / `hop_length` と `model_cfg.latent_patch_size` から、runtime と同じ `patched_steps` を返す。
  - `InferenceRuntime.tokenize_for_bucket(normalized_text: str) -> tuple[int, str]`：truncation なしの tokenizer encode で token 長と `token_ids_hash` を返す。`batch_encode(max_length=...)` は長さが切り詰められるため使わない。
- 入力ドメイン：
  - `seconds`：segment plan が決めた値、もしくは payload 由来の `seconds` 推定。
  - `text_token_len`：`AutoSpeechPlanningContext.token_len`。必ず runtime tokenizer の truncation なし encode から得る。
- 選択ロジック：
  1. `patched_steps = runtime.estimate_patched_steps(seconds)` を使う。API 側で `DEFAULT_*` 推定値を重複実装しない。
  2. **Initial production S buckets**：候補 `S_bucket ∈ {100, 160, 200}` に固定する。`S_bucket >= patched_steps` の最小値を採用。
     - `seconds=5.0` の典型 `patched_steps=125` は `S=160` を選ぶ（これが本プランの基準ケース）。
     - 64 / 80 / 128 等の細粒度バケットは required / default 集合に含めない（§8）。
  3. `T_bucket ∈ {64, 128, 256}` から `T_bucket >= text_token_len` の最小値。
  4. `R_bucket` は rem.wav 既定 reference を前提に固定 `160`（実測 `speaker_context_len=138`）。本プランでは複数 reference を扱わないため動的化しない（§8）。
- bucket 解決失敗（`patched_steps > 200` / `text_token_len > 256` 等）の HTTP 振る舞いは §4.6 の表に従う（`AUTO` は silent fallback to PyTorch、`REQUIRE` / `PREPARE` / `REFRESH` は 4xx）。
- 明示 `irodori.bucket` が来たらこのロジックは **bypass** する。

### 4.3 Required residency layers

「`encode_conditions()` を二度走らせない」を真に達成するため、以下を residency レイヤとして定義する。番号順に強い必須度。

1. **Default reference latent / mask**：rem.wav 由来の `ref_latent` および `ref_mask`（DACVAE 出力 + padding mask）。fingerprint key = `reference_fingerprint(rem.wav)`。
2. **Default speaker state / mask**：rem.wav 由来の `speaker_state` および `speaker_mask`。同 key。
3. **Bucketed speaker KV**：`(model_fingerprint, reference_fingerprint, R_bucket, S_bucket, speaker_kv_scale_signature, branch_layouts)` を key とする speaker-only projection 出力。bucket ごとに 1 entry。
4. **(Optional) Condition packed-KV payload cache**：AUTO fingerprint v2（§4.4）を key とする `pack_context_kv_state()` の出力（text 側 KV + resident speaker KV を含む完成形 condition cache）。MLState に流し込む直前のテンソル。これを持つと 2 回目以降は text encode + KV projection もスキップ可能。
5. **(Optional) MLState pool / lease**：`MLState` インスタンスをスレッド/リクエスト境界でプールしてリースで貸し出す。**正常系の正確性と並行 lease が証明された後でのみ** 導入する（§8）。本プランの初期スコープでは MLState は従来どおり毎リクエスト `make_state()` で確保する。

residency 設計上の不変条件：

- actual torch tensor residency（`ref_latent` / `ref_mask` / `speaker_state` / `speaker_mask` / speaker KV / packed KV）は `InferenceRuntime` 内の runtime-resident store が所有する。key は cache id / fingerprint。`InMemoryCoreMLCacheManager` は metadata / index / TTL / LRU / metrics のみを持ち、torch tensor の lifetime を所有しない。
- `InMemoryCoreMLCacheManager` は materialized layer 名、resident key、`memory_bytes`、last_used を追跡してよいが、metrics endpoint は tensor payload を絶対に返さない。
- `memory_bytes` は実テンソル / np array の `numel * element_size` または `nbytes` から計算する。`DEFAULT_NUM_LAYERS` / `DEFAULT_NUM_HEADS` 等による見積もりを resident accounting に使わない。
- (1)(2) は lifespan で eager 構築されサーバ生存中常駐する。
- default rem.wav の resident prepare では、runtime が実測した `ref_len`、`speaker_context_len`、`speaker_dim`、`memory_bytes` を使って `ReferenceCacheRequest` / `ReferenceCacheHandle` を作成または更新する。旧 metadata default（例：`speaker_context_len=1`）は resident default cache では不可。
- (3) は warmup フェーズで bucket ごとに precompute する（少なくとも S=160 と S=100 を必須、S=200 はオプション）。
- (3) は speaker-only KV projection helper、または model/runtime の明示的な speaker projection helper で作る。zero-text を渡して speaker KV を取り出す方法は一時診断に限り、production acceptance にはしない。fresh text KV と resident speaker KV は `pack_context_kv_state()` 前に merge する。
- shared resident speaker KV に `speaker_kv_scale` を適用する場合は clone/copy した payload か scale signature 別 entry を使い、共有 payload を in-place 変更しない。
- (4) は AUTO 経路で req 1 回目に prepare、2 回目以降ヒット。TTL あり（既定 24h、`condition_cache_default_ttl_seconds`）。
- (5) は将来導入。本プランでは設計記述のみ。
- `synthesize_with_condition_cache()` は resident parts を明示引数で受け取るか、runtime-resident store から condition/reference id で lookup する。metadata-only manager から tensor を取り出す設計にはしない。
- 「rem.wav 以外」の reference を扱うパス（拡張 `irodori.reference`）は本プランでは resident pool に入れない。リクエスト時に temporary に encode する従来挙動を維持（§8）。

### 4.3.1 Concurrency hardening

- P4/P5/P6 の resident prepare 前に、`InMemoryCoreMLCacheManager` に manager-level lock と per-key prepare locks を入れる（P3 hardening subphase として実施可）。
- lock 対象は reference / condition metadata、resident metadata、metrics、delete / evict / TTL prune 経路。
- `InferenceRuntime` の runtime-resident store も per-key lock を持ち、payload は immutable として公開する。prepare 中の partial payload は store / metadata のどちらにも登録しない。

### 4.4 Condition cache auto-prepare / reuse

- `_resolve_speech_cache()` の `AUTO + cache_id None` 分岐は **segment ごと** に動作する（§4.5）。各 segment について：
  1. fingerprint 計算：
     - explicit `/v1/tts/condition-caches` は既存 canonical fingerprint を維持する（互換性優先）。
     - internal AUTO は fingerprint v2 を使う。要素は `token_ids_hash`（正規化 text を runtime tokenizer で truncation なし encode した ID 列）、bucket `S/T/R`、`cfg_signature`、`branch_layouts`、`model_fingerprint`、`tokenizer_fingerprint`、`default_reference_fingerprint`、`speaker_kv_scale_signature`。
     - raw `seconds` は fingerprint v2 に入れない。実リクエストの `seconds` は request/runtime data として保持し、同じ bucket に入る `seconds=4.0` と `4.05` は同 condition KV identity をヒットできる。
     - `cfg_signature`：cfg mode + scales + min/max_t をシリアライズしたタプル。
  2. condition packed-KV pool（§4.3 layer 4）にヒットすれば、その payload から `ConditionCacheHandle` を構築して返す。
  3. ミスなら：
     - speaker KV pool（layer 3）にヒットしたら text 側 KV のみ計算し pack。
     - 全ミスなら `_pack_and_prepare()` 全工程を走らせ、終了後に layer 4 と（必要なら）layer 3 に書き戻す。
- TTL：`condition_cache_default_ttl_seconds`（CLI 設定可能）。デフォルト reference 由来の condition cache は長めの既定（例 24h）。
- speaker KV / packed-KV の容量上限は `max_resident_speaker_kv_buckets` / `max_resident_condition_cache_entries` で抑える。LRU。

### 4.5 Multi-segment behavior

- 既存 `audio_speech()` は `segment_plan.segments` を順に `runtime.synthesize(...)` する。AUTO 自動 prepare は **segment 単位** で実施する。
- `_resolve_speech_cache()` の戻り値はもはや単一 `condition_cache_handle` ではなく **segment 並びと同形の `list[_SpeechCacheResolution]`** とする：
  ```
  resolutions: list[_SpeechCacheResolution] = [
      _SpeechCacheResolution(reference_handle=ref_h, condition_handle=cond_h_seg_i, ...),
      ...
  ]
  ```
  - `reference_handle` は全 segment 共通（rem.wav 固定のため）。
  - `condition_handle` は segment 個別。あるセグメントで CoreML が組めなかった場合、その segment の handle のみ `None`（= PyTorch にフォールバック）にできる。
- `audio_speech()` は segment ごとに `resolutions[i].condition_handle is not None` で backend を切り替える（mixed CoreML+PyTorch を許容）。
- ヘッダは複数 segment 対応：
  - `X-Irodori-Cache-Condition-Id`：カンマ区切りで全 segment の cache id を出す（segment 1 件なら従来形と同一）。長すぎるときは「先頭 ID + `,+N more`」で要約して `X-Irodori-Cache-Condition-Count` に件数を入れる。
  - transition 中は既存ヘッダ `X-Irodori-Condition-Cache-Id` も同じ値で併記する。
  - `X-Irodori-Cache-Reference-Id` と既存 `X-Irodori-Reference-Cache-Id` も併記する。
  - `X-Irodori-Backend`：全 segment が CoreML なら `coreml-stateful`、全 PyTorch なら `pytorch`、混在なら `mixed`。`X-Irodori-Backend-Per-Segment` でカンマ区切り詳細を出す。
- 既存 `X-Irodori-Denoiser-Backend` は `X-Irodori-Backend` の互換 alias として当面維持する。
- group ID（`cg_...`）は本プランでは導入しない（§8）。`irodori.cache_ids` / `cache_group_id` の入力受理は既存仕様のまま。
- 同一テキストの reflow（`seed` のみ違う等）は segment hash + bucket 一致で同 condition cache を再利用する。

### 4.6 Fallback semantics

明示的に表で固定する。`silent` = HTTP 200 を維持して PyTorch 経路で生成。`fail` = HTTP 4xx/5xx。

| cache_mode | cache_id 有 | cache_id 無 | bucket 解決失敗 | CoreML 利用不可 / mlmodel error |
| --- | --- | --- | --- | --- |
| `off`        | 無視（PyTorch） | PyTorch | n/a | n/a |
| `auto`（既定） | ヒット → CoreML / 不一致 → 自動 prepare → CoreML / 不能なら silent fallback | 自動 prepare → CoreML / 不能なら silent fallback | **silent fallback to PyTorch（200）** | **silent fallback to PyTorch（200）** |
| `prepare`    | reuse / 不一致なら refresh | 自動 prepare | 422 | 503 |
| `refresh`    | 強制 refresh | 自動 prepare | 422 | 503 |
| `require`    | reuse のみ／不一致 fail | 404/422 | 422 | 503 |

要点：

- **`auto` は OpenAI 既定経路** であり、bucket オーバーサイズや CoreML 不可は **必ず 200 + PyTorch + ヘッダ** で返す。422/503 を返すのは `require` / `prepare` / `refresh` または `--strict-coreml` 明示時のみ。
- silent fallback 時は次を必ず観測に残す：`X-Irodori-Fallback-Reason`（`bucket_oversize_s` / `bucket_oversize_t` / `coreml_unavailable` / `mlmodel_compile_error` / `state_alloc_error` 等）、`X-Irodori-Bucket-Attempted`、metrics カウンタ。

### 4.7 Startup warmup policy

- `start_openai_server.sh` / `launchd/com.ramo.irodori-tts-openai-api.plist` の起動引数を以下に拡張：
  - `--preload`
  - `--warmup-bucket S=100,T=256,R=160`
  - `--warmup-bucket S=160,T=256,R=160`（5–6 秒帯。本プランの基準ケース）
  - `--warmup-bucket S=200,T=256,R=160`（最大帯）
  - `--auto-prepare-default-cache`（既定 True、明示記載でも可）
  - `--default-reference-cache-prepare`（lifespan 内で rem.wav の reference cache を eager 化）
  - `--default-condition-cache-prepare "<sample text>"`（任意。dummy text で代表 condition cache を温める。silent fallback 検証用）
- 起動シーケンス：
  1. `precompile_coreml_stateful_buckets()`（`cond1` の `S∈{100,160,200}`, `T=256`, `R=160`）。
  2. `prepare_reference_cache(server_default)` → resident layer 1 + 2 充填。
  3. layer 3（bucketed speaker KV）を bucket ごとに precompute（少なくとも S=160）。
  4. `--default-condition-cache-prepare` 指定時に layer 4 を一件温める。
- 起動失敗時：
  - reference prepare 失敗（layer 1/2）→ 致命扱いで起動中止（rem.wav 不正）。
  - speaker KV warmup 失敗（layer 3）→ ログ警告のみ、起動継続。
  - condition warmup 失敗（layer 4）→ ログ警告のみ、起動継続（runtime AUTO 経路でリトライ）。

### 4.8 Metrics / Observability

`X-Irodori-*` ヘッダおよびカウンタ：

ヘッダ：

- `X-Irodori-Backend`: `coreml-stateful` | `pytorch` | `mixed`
- `X-Irodori-Backend-Per-Segment`: カンマ区切り（mixed の場合のみ）
- `X-Irodori-Cache-Auto`: `hit` | `prepared` | `reused` | `miss-fallback`
- `X-Irodori-Bucket`: `S=NNN,T=NNN,R=NNN`
- `X-Irodori-Cache-Reference-Id`: `ref_...`
- `X-Irodori-Reference-Cache-Id`: `ref_...`（互換 alias）
- `X-Irodori-Cache-Condition-Id`: `cond_...`（カンマ区切り or 要約）
- `X-Irodori-Condition-Cache-Id`: `cond_...`（互換 alias）
- `X-Irodori-Cache-Condition-Count`: `<N>`
- `X-Irodori-Voice-Resolved`: `server-default`
- `X-Irodori-Fallback-Reason`（fallback 時のみ）
- `X-Irodori-Bucket-Attempted`（fallback 時のみ）

カウンタ：

- `irodori_default_cache_auto_total{outcome}` （hit / prepared / fallback）
- `irodori_bucket_resolution_total{outcome}` （ok / oversize_s / oversize_t / fail）
- `irodori_resident_layer_present{layer=1..4}`
- `irodori_coreml_silent_fallback_total{reason}`
- metrics snapshot は resident metadata と `memory_bytes` のみ返し、torch tensor / np array payload は返さない。

## 5. Implementation Phases (acceptance criteria)

各フェーズは前フェーズが PASS してから着手する。コード詳細は本書では書かない。

### Phase A: Default cache_mode = auto + voice ignore contract

- 変更対象: `openai_api_server.py::_speech_cache_mode()`、`_resolve_speech_cache()`、`audio_speech()` の voice 受信ハンドラ、ヘッダ追加。
- 受け入れ:
  - `irodori` 未指定 / `cache_mode` 未指定の payload で `_speech_cache_mode()` が `AUTO` を返す単体テスト。
  - `voice` 未指定 / `"default"` / `"alloy"` / `"nova"` / `"<garbage>"` の各ケースで rem.wav が使われ、`X-Irodori-Voice-Resolved: server-default` が付くテスト。
  - 既存 `cache_mode=off` 経路の PyTorch 動作が変わらない回帰テスト。

### Phase B: Auto bucket selection（S∈{100,160,200}）

- 変更対象: `InferenceRuntime.estimate_patched_steps()`、`InferenceRuntime.tokenize_for_bucket()`、`AutoSpeechPlanningContext`、`openai_api_server.py` の `_auto_select_bucket()`。
- 受け入れ:
  - `tokenize_for_bucket()` は truncation なし encode を使い、`batch_encode(max_length=...)` を使わない単体テスト。
  - `estimate_patched_steps()` が runtime codec / `latent_patch_size` と一致する単体テスト。
  - `seconds=4.0` の典型 token 長で `S_bucket=100` を選ぶ単体テスト。
  - `seconds=5.0` で `S_bucket=160` を選ぶ単体テスト（基準ケース、`patched_steps=125`）。
  - `seconds=8.0` で `S_bucket=200` を選ぶ単体テスト。
  - `patched_steps>200` / `text_token_len>256` は helper が oversize reason を返す単体テスト。
  - strict response shaping は fake planning context で検証する。full AUTO fallback E2E は auto prepare / per-segment resolution 完成後の Phase G に置く。
  - `cache_mode=require` で `patched_steps>200` のとき 422 を返すテスト。
  - `irodori.bucket` 明示時に `_auto_select_bucket()` がスキップされるテスト。

### Phase C: Resident layers 1 + 2 (default reference latent / mask, speaker state / mask)

- 変更対象: `InferenceRuntime` runtime-resident store に `ResidentReferenceTensors`、`coreml_cache.py` は resident metadata 追跡のみ、`inference_runtime.py::_pack_and_prepare()` の reference 取り出しを runtime store 経由化、`openai_api_server.py::lifespan()` で eager prepare、`_validate_reference_wav()` の致命扱い化。
- 受け入れ:
  - lifespan 終了後 `state.default_reference_cache_id` がセットされ、layer 1+2 が pool に存在する integration test。
  - `ReferenceCacheRequest` / `ReferenceCacheHandle` が runtime 実測の `ref_len`、`speaker_context_len`、`speaker_dim`、実 tensor 由来 `memory_bytes` で作られるテスト。
  - 同一 rem.wav 由来の `_pack_and_prepare()` が二度目以降 `encode_conditions()` の reference 部分（DACVAE + speaker encoder）をスキップする（カウンタで検証）。
  - rem.wav 不正で lifespan が起動を中止する。

### Phase D: Resident layer 3 (bucketed speaker KV)

- 変更対象: `InferenceRuntime` runtime-resident store に bucketed speaker KV payload、`coreml_cache.py` は metadata / LRU / memory_bytes 追跡、`inference_runtime.py` に speaker-only KV projection helper、warmup フックで bucket ごと precompute。
- 受け入れ:
  - speaker-only projection helper が text projection を払わず speaker KV を作る。zero-text workaround は production 受け入れ不可。
  - fresh text KV と resident speaker KV を merge してから `pack_context_kv_state()` へ渡すテスト。
  - scale 済み variant は clone/copy または別 entry で、shared resident payload を in-place 変更しないテスト。
  - 同一 rem.wav + bucket=160 の `_pack_and_prepare()` が二度目以降に **speaker KV projection もスキップ** する（カウンタで検証）。これが「real KV residency」の最低ラインの受け入れ。
  - bucket=100 と bucket=160 で別 entry になることを確認するテスト。
  - `max_resident_speaker_kv_buckets` を超えたとき LRU 退避するテスト。

### Phase E: Condition cache auto-prepare / reuse + (optional) layer 4

- 変更対象: `InMemoryCoreMLCacheManager.find_condition_cache_by_fingerprint()` 相当の metadata lookup、runtime-resident layer 4 packed-KV pool（任意）、`_resolve_speech_cache()` の AUTO 分岐、AUTO fingerprint v2 計算ヘルパー。
- 受け入れ:
  - `irodori` なしで連続 2 回同一 text/seconds リクエスト → 1 回目 `Cache-Auto: prepared`、2 回目 `Cache-Auto: hit`。
  - `seconds=4.0` と `seconds=4.05` が同 `S_bucket=100` で同 condition cache をヒットさせるテスト（quantization 検証）。
  - explicit `/v1/tts/condition-caches` の canonical fingerprint 互換性を維持するテスト。
  - AUTO fingerprint v2 が `token_ids_hash`、bucket `S/T/R`、cfg、branch layouts、model/tokenizer/reference fingerprints、speaker scale signature を含み raw seconds を含まないテスト。
  - cfg 設定変更で別 cache が作られるテスト。
  - layer 4 を有効化した構成で、2 回目に text encode + KV projection もスキップされることをカウンタで検証（layer 4 は optional の受け入れ）。

### Phase F: Per-segment resolution + multi-segment headers

- 変更対象: `_resolve_speech_cache()` の戻り値を `list[_SpeechCacheResolution]` 化、`audio_speech()` の segment ループ、ヘッダ生成。
- 受け入れ:
  - 2 segment 入力で各 segment の condition_id がヘッダに出る（カンマ区切り）。
  - segment 1 が CoreML、segment 2 が bucket oversize で PyTorch、`X-Irodori-Backend: mixed`、`X-Irodori-Backend-Per-Segment: coreml-stateful,pytorch` が付くテスト。
  - 6 segment 以上で `X-Irodori-Cache-Condition-Id` が要約形になる（先頭 ID + `,+N more`）テスト。

### Phase G: Silent fallback semantics（end-to-end）

- 変更対象: `audio_speech()` の例外捕捉、metrics、`--strict-coreml` フラグ。
- 受け入れ:
  - `auto` で bucket oversize → 200 + `Backend: pytorch` + `Fallback-Reason: bucket_oversize_s`。
  - `auto` で CoreML 利用不可（mlpackage 欠如シミュレーション）→ 200 + `Fallback-Reason: coreml_unavailable`。
  - 同条件で `cache_mode=require` は 422、`prepare` は 422、`off` は 200（PyTorch）、`--strict-coreml` 時の `auto` は 503。

### Phase H: Startup warmup wiring

- 変更対象: `lifespan()`、`start_openai_server.sh`、`launchd/com.ramo.irodori-tts-openai-api.plist`（`--auto-prepare-default-cache` を既定 True で含める）。
- 受け入れ:
  - 起動ログで「reference prepared」「N buckets precompiled」「speaker KV resident: N」が出る。
  - rem.wav 不正時に lifespan が起動を中止する。

### Phase I: Observability

- 変更対象: response ヘッダ追加、metrics カウンタ追加。
- 受け入れ:
  - 各経路（hit / prepared / fallback / pytorch-explicit / mixed）でヘッダ値が一致する。

## 6. Testing Plan

### 6.1 Unit

- `tests/test_speech_cache_mode_default.py`：`_speech_cache_mode()` の既定値、`auto_prepare_default_cache=False` 時のフォールバック挙動、voice 値群（unset / `"default"` / `"alloy"` / `"nova"` / `"<garbage>"`）が rem.wav にマップされること。
- `tests/test_auto_select_bucket.py`：bucket 選択テーブル（S∈{100,160,200}）、`tokenize_for_bucket()` の truncation なし encode、上限超過時の oversize reason、fake planning context での strict response shaping、明示 bucket bypass。
- `tests/test_condition_cache_fingerprint.py`：explicit canonical fingerprint 互換、AUTO fingerprint v2 の bucket 正規化、raw seconds 非依存、`token_ids_hash` 利用。
- `tests/test_resident_reference_pool.py`：layer 1+2 の put/get/evict、cascade、フィンガープリント不一致時の miss。
- `tests/test_resident_speaker_kv_pool.py`：layer 3 の bucket ごと entry、LRU、再ヒット時に speaker KV projection を呼ばないことの検証（mock counter）。
- `tests/test_resident_cache_concurrency.py`：manager-level lock、per-key prepare lock、runtime-resident payload immutability、metrics/delete/evict/prune の並行安全性。

### 6.2 Integration

- `tests/test_audio_speech_default_coreml.py`（新設）：
  - rem.wav 既定 + irodori 無し + seconds=4 → CoreML stateful、200、`S=100`。
  - rem.wav 既定 + irodori 無し + seconds=5 → CoreML stateful、200、`S=160`。
  - 同 payload 連続 2 回 → 2 回目は `Cache-Auto: hit` かつ reference encode + speaker KV projection スキップ（counter）。
  - bucket oversize（`seconds=15` 等）→ silent fallback、200、`Backend: pytorch`、`Fallback-Reason: bucket_oversize_s`。
  - CoreML 利用不可シミュレーション → 200、`Fallback-Reason: coreml_unavailable`。
  - `voice="alloy"`（OpenAI 既定）でも rem.wav が使われ 200 を保つ。
  - `cache_mode=require` 単独で送ると 404/422 を維持。
  - `cache_mode=off` で従来 PyTorch 経路。
  - 2 segment 入力で混在 backend のヘッダが正しく付与される。

### 6.3 End-to-end (実機)

- macOS Apple Silicon 上で `start_openai_server.sh` 起動 → 4 秒 / 5 秒 / 8 秒の生成を `curl` で計測。
  - 計測項目: first request latency、warm request latency、`X-Irodori-Backend`、`X-Irodori-Cache-Auto`、メモリ常駐量。
  - 5 秒生成の warm request が PyTorch 経路を踏まないことを `Fallback-Reason` 不在で確認。
- B=1 cond1 を 3 連発する split cond1 経路の retain（B=2/B=3 統合 mlmodel への分岐が出ていないことの確認）。

### 6.4 既存回帰

- `tests/test_tts_cache_api.py`、`tests/test_coreml_real_step_benchmark_import.py`、`tests/test_coreml_stateful_step_helpers.py` の green 維持。
- `uv run ruff check`、`python3 -m compileall` の green 維持。

## 7. Risks and Tradeoffs

- **silent fallback がデバッグを難しくする**：ヘッダとメトリクスで観測可能性を担保するが、ユーザは「速いはず」と期待しているのに気付かず PyTorch を踏み続け得る。`--strict-coreml` フラグで silent fallback を fail に切り替えられる脱出口を残す。
- **resident layer のメモリ増**：rem.wav 1 本でも layer 1+2 で数 MB、layer 3 を S∈{100,160,200} 全部温めると bucket 当たり数 MB が固定常駐する。1 本想定なら許容、将来 multi-voice 化時に pool 上限が必要。
- **fingerprint 衝突**：text の正規化漏れがあると別テキストで同 cache が hit する。`tokenizer_fingerprint` と `text_normalization_version` を必ず含め、tokenizer 出力 ID 列のハッシュを使う。
- **bucket misalignment**：`seconds=5.0` が `S=128` で十分だった場合 R/T が無駄に大きい model を温める。S∈{100,160,200} の粗い集合は initial production 向けの割り切りで、計測 evidence が出てから細粒度集合へ拡張する（§8）。
- **CoreML 実行計画 -14 リスクの再来**：本プランは split cond1 維持なので影響は無いが、将来 B>=2 の同一 mlmodel に戻したくなった際は別 RFC 必須（§8）。
- **AUTO の挙動変更が後方互換ブレイク**：既存クライアントで `irodori` 無し ＋ 高速化前提のないユーザは音は同じでも latency 分布が変わる。`--auto-prepare-default-cache=False` で opt-out 可能、changelog に明記。
- **per-segment resolution への戻り値変更**：内部 API 変更なので `_resolve_speech_cache()` の呼び出し側全てを同時に直す必要がある（プランは Phase F で集約）。

## 8. Explicit Non-Goals

- B=2/B=3 のバッチ統合 CoreML model への切り替え。実機で CoreML 実行計画 `-14` を踏んだ実績があるため、cond1 を 3 回連続実行する split cond1 戦略を runtime 前提として維持する。`joint2` / `alternating_text2` / `alternating_speaker2` は branch layout の **メタデータ / API 契約**としては存在するが、実行パスは導入しない。
- 細粒度 S buckets（64 / 80 / 128 等）。initial production は `{100, 160, 200}` のみ。将来 measurement evidence 付きで追加検討する future optimization。
- DACVAE / text encoder / speaker encoder の CoreML 化（既存 P6 と同じ位置付け）。
- Server-side voice registry（複数 voice の preload + voice→reference のサーバ内マッピング）。本プランでは `voice` パラメータと client-side reference は **すべて無視**。
- multi-segment group cache（`cg_...`）の HTTP 化。
- `MLState` の永続化／プロセス再起動跨ぎ。
- MLState pool / lease（resident layer 5）。正常系の正確性と並行 lease の安全性が証明された段階で別 RFC で扱う。
- PyTorch 経路自体の高速化や CFG 動的縮小。
- response cache（生成済み wav の再配信）。
- rem.wav 以外の reference を resident pool に入れる仕組み。

## 9. Open Questions

1. **resident pool の上限と GC**：rem.wav のみ前提だが、将来 multi-voice を扱う場合の LRU 件数上限と eviction policy を本プランで決めるか、multi-voice RFC まで後送りにするか。
2. **`--strict-coreml` の必要性**：default が silent でも、SLO 計測中は強制 fail のほうが原因切り分けが速い。両モード提供の運用負荷（テスト網羅）が見合うかは要判断。
3. **layer 4 (condition packed-KV payload cache) を初期スコープに含めるか**：layer 3 までで「reference encode + speaker KV projection をスキップ」は達成できるが、text 側 KV も含めて完全 hit にするには layer 4 が要る。memory コストとヒット率次第で Phase E の任意部分に置くか必須化するか。
4. **fingerprint における `cfg_min_t/cfg_max_t` の扱い**：既存 KV cache 設計どおり「branch layout の互換性」として比較するが、独立 fingerprint としての保存粒度は再確認したい（既存 explicit `condition_fingerprint` 互換を壊さない範囲に限る）。
5. **要約形ヘッダの閾値**：`X-Irodori-Cache-Condition-Id` を要約に切り替える segment 数の閾値（仮に 6 と置いたが、実運用で 3 / 4 / 8 のいずれが妥当か観測してから決める）。
