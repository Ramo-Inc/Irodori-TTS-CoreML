# CoreML KV Cache API 設計案

## 結論

このキャッシュは raw sample wav ではない。最大性能を狙うべきキャッシュは、音声波形そのものや最終生成音声ではなく、TTS モデルが毎 denoise step で参照する text / speaker conditioning を、各 diffusion block の attention 用 Key / Value へ射影済みにしたテンソルである。

参照音声は最終的に `speaker_state` へエンコードされ、テキストは `text_state` へエンコードされる。さらに高速化対象は、それらを各 layer の `k_text/v_text/k_speaker/v_speaker` に変換する context KV projection である。CoreML 側では、この per-layer conditioning KV を `MLState` に一度だけ詰め、denoise loop の各 step では `x_t`、`t`、`latent_mask`、step flags だけを `predict` に渡す。25MB 級の KV テンソルを毎 step の通常 input として渡す設計は避ける。

この設計の第一目的は Mac mini Apple Silicon 上で `TextToLatentRFDiT` の denoiser step を CoreML CPU_AND_NE に固定し、既存 PyTorch の context KV cache と同等の意味を CoreML stateful model と HTTP API で公開することである。

## 前提とスコープ

- 対象 branch: `develop`
- production code は本設計書では変更しない。
- 対象 checkpoint: `Aratako/Irodori-TTS-500M-v2`
- 対象 config:
  - `model_dim=1280`
  - `num_layers=12`
  - `num_heads=20`
  - `head_dim=64`
  - `text_dim=512`
  - `speaker_dim=768`
  - `text_len max=256`
  - `use_speaker_condition=True`
  - `use_caption_condition=False`
- 既存 PoC 実測:
  - `sequence_length=100`
  - `text_len=256`
  - `speaker_context_len=138`
  - `ref_len=137`
  - CoreML CPU_AND_NE no-context-kv-cache one step: `14.428 ms`
  - PyTorch MPS no-cache one step: `53.487 ms`
  - PyTorch MPS cached one step: `43.881 ms`
  - CoreML ComputePlan: `ios16.linear 293/293 NE`, `ios16.matmul 24/24 NE`, `ios16.softmax 12/12 NE`, `total_ne_preferred_ops=1626`
  - `compute_precision=float16`, `status PASS`, `rel_diff=0.0251`

参考資料:

- Core ML Tools Stateful Models: https://apple.github.io/coremltools/docs-guides/source/stateful-models.html
- Apple Core ML `MLState`: https://developer.apple.com/documentation/coreml/mlstate

## 既存コードの根拠

### `irodori_tts/model.py`: context KV があれば projection を省略する

```python
   311        if context_kv is None:
   312            projected = self.project_context_kv(
   313                text_context=text_context,
   314                speaker_context=speaker_context,
   315                caption_context=caption_context,
   316            )
   317        else:
   318            projected = context_kv
```

この分岐が CoreML 設計の中心である。毎 step 同じ `text_context` / `speaker_context` から KV を作るのではなく、事前に projected KV を作って attention に渡す。

### `irodori_tts/model.py`: per-layer cache を block に渡す

```python
   738        text_state: torch.Tensor,
   739        text_mask: torch.Tensor,
   740        speaker_state: torch.Tensor | None,
   741        speaker_mask: torch.Tensor | None,
   742        caption_state: torch.Tensor | None = None,
   743        caption_mask: torch.Tensor | None = None,
   744        latent_mask: torch.Tensor | None = None,
   745        context_kv_cache: list[tuple[torch.Tensor, ...]] | None = None,
...
   753        for i, block in enumerate(self.blocks):
   754            x = block(
...
   763                freqs_cis=freqs,
   764                self_mask=latent_mask,
   765                context_kv=context_kv_cache[i] if context_kv_cache is not None else None,
   766            )
```

CoreML state はこの `context_kv_cache[i]` の役割を持つ。つまり state の粒度は少なくとも layer 次元を含む必要がある。

### `irodori_tts/model.py`: cache は per-layer projected conditioning KV

```python
   817    def build_context_kv_cache(
   818        self,
   819        text_state: torch.Tensor,
   820        speaker_state: torch.Tensor | None,
   821        caption_state: torch.Tensor | None = None,
   822    ) -> list[tuple[torch.Tensor, ...]]:
   823        """
   824        Build per-layer projected conditioning KV tensors for faster repeated sampling steps.
   825        """
   826        return [
   827            block.attention.project_context_kv(
   828                text_context=text_state,
   829                speaker_context=speaker_state,
   830                caption_context=caption_state,
   831            )
   832            for block in self.blocks
   833        ]
```

ここで生成されるものは raw wav でも `ref_latent` でもなく、各 diffusion block の conditioning KV である。

### `irodori_tts/rf.py`: CFG 用に複数の context KV cache を作る

```python
   363    # Force-speaker scaling operates on projected speaker K/V, so it requires context KV caches.
   364    effective_use_context_kv_cache = bool(use_context_kv_cache or (speaker_kv_scale is not None))
   365
   366    context_kv_cond = None
   367    context_kv_cfg = None
   368    context_kv_joint_uncond = None
   369    context_kv_alternating: dict[str, list[tuple[torch.Tensor, ...]]] = {}
   370    if effective_use_context_kv_cache:
   371        context_kv_cond = model.build_context_kv_cache(
   372            text_state=text_state_cond,
   373            speaker_state=speaker_state_cond,
   374            caption_state=caption_state_cond,
   375        )
   376        if use_independent_cfg and cfg_batch_mult > 1:
   377            context_kv_cfg = model.build_context_kv_cache(
   378                text_state=independent_text_state,
   379                speaker_state=independent_speaker_state,
   380                caption_state=independent_caption_state,
   381            )
   382        elif use_joint_cfg:
   383            if enabled_cfg_names:
   384                context_kv_joint_uncond = model.build_context_kv_cache(
   385                    text_state=joint_uncond_bundle[0],
   386                    speaker_state=joint_uncond_bundle[2],
   387                    caption_state=joint_uncond_bundle[4],
   388                )
   389        elif use_alternating_cfg:
   390            for name in enabled_cfg_names:
   391                bundle = alternating_bundles[name]
   392                context_kv_alternating[name] = model.build_context_kv_cache(
   393                    text_state=bundle[0],
   394                    speaker_state=bundle[2],
   395                    caption_state=bundle[4],
   396                )
```

CoreML の condition cache は cond だけでは不十分で、CFG active step で使う uncond branch の KV も含める必要がある。

### `irodori_tts/rf.py`: independent CFG は batch を増やす

```python
   422        use_cfg = bool(enabled_cfg_names) and (cfg_min_t <= t.item() <= cfg_max_t)
   423        if use_cfg:
   424            if use_independent_cfg:
   425                x_t_cfg = torch.cat([x_t] * cfg_batch_mult, dim=0).to(dtype)
   426                tt_cfg = tt.repeat(cfg_batch_mult)
   427                v_out = model.forward_with_encoded_conditions(
   428                    x_t=x_t_cfg,
   429                    t=tt_cfg,
   430                    text_state=independent_text_state,
   431                    text_mask=independent_text_mask,
   432                    speaker_state=independent_speaker_state,
   433                    speaker_mask=independent_speaker_mask,
   434                    caption_state=independent_caption_state,
   435                    caption_mask=independent_caption_mask,
```

DEFAULT_CHECKPOINT は speaker-conditioned / no-caption なので、text CFG と speaker CFG が有効な independent mode では通常 `cond + text-uncond + speaker-uncond` の `B_eff=3` になる。

### `openai_api_server.py`: 既存 API は segment ごとに synthesize する

```python
   625            for segment_index, segment in enumerate(segment_plan.segments):
   626                segment_seed = None if seed is None else int(seed) + segment_index
   627                result = await asyncio.to_thread(
   628                    runtime.synthesize,
   629                    SamplingRequest(
   630                        text=segment.text,
   631                        caption=caption,
   632                        ref_wav=str(settings.reference_wav),
   633                        ref_latent=None,
   634                        no_ref=False,
   635                        num_steps=int(num_steps),
   636                        seconds=float(segment.seconds),
   637                        max_ref_seconds=settings.max_ref_seconds,
   638                        seed=segment_seed,
   639                    ),
```

新 API はこの segment loop を壊さず、segment ごとに condition cache を使えるようにする。OpenAI-compatible `/v1/audio/speech` は binary audio response のまま保つ。

## キャッシュ階層

| 層 | 内容 | 再利用単位 | 主目的 |
| --- | --- | --- | --- |
| reference/audio cache | optional な `ref_latent` と `ref_mask` | reference audio + codec/model config | 参照音声の decode / normalize / DACVAE encode を省く |
| speaker_state cache | `speaker_state`, `speaker_mask` | reference audio + model fingerprint | diffusion denoiser の speaker condition 入力を再利用する |
| speaker_kv cache | per-layer `k_speaker/v_speaker` | reference + model + bucket + CFG branch + speaker_kv_scale 設定 | reference 由来の speaker KV projection を省く |
| text_state/text_kv cache | `text_state/text_mask` と per-layer `k_text/v_text` | text chunk + tokenizer + model + bucket + branch | text encoding と text KV projection を省く |
| CoreML MLState condition cache | packed fp16 `context_k/context_v/valid_mask` state | request/chunk/bucket/CFG layout | denoise step の CoreML `predict` で conditioning tensors を input に渡さない |

### ID と fingerprint

cache id は内容 hash だけに依存させず、互換性判定に必要な metadata を必ず持つ。

- `model_fingerprint`: checkpoint repo/path、safetensors hash、config hash、CoreML mlpackage build id
- `tokenizer_fingerprint`: text tokenizer repo/revision、normalization version
- `codec_fingerprint`: DACVAE repo/revision、encode settings
- `reference_fingerprint`: audio bytes hash、normalize settings、trim/max_ref_seconds
- `condition_fingerprint`: text、caption、seconds、sequence bucket、CFG mode/scales/window
- `bucket_id`: `S{sequence_length}_T{text_len}_R{speaker_context_len}_C{caption_len}_B{branch_layout}`
- `state_layout_version`: packed state の schema version

## CoreML 実装設計

### stateful model 方針

CoreML Tools の stateful model は `ct.StateType` を使い、実行時に `MLModel.make_state()` で `MLState` を作って `predict(..., state=state)` に渡す。stateful `mlprogram` conversion では coremltools が公開している target として `ct.target.iOS18` を使う。current coremltools 9.0 では `ct.target.macOS15` は exposed ではないため、変換 target として使わない。runtime requirement は引き続き macOS 15+ とする。

実装上の dtype 制約:

- Stateful model conversion では state tensor は実質 fp16 が必須である。PoC では fp32 `StateType` が次の error で失敗した: `State only support fp16 dtype. Got input var cache with dtype fp32.`
- PyTorch wrapper 側で registered torch buffer にする cache tensor は `torch.float16` にする。
- `ct.StateType` で wrap する `ct.TensorType` も `dtype=np.float16` にする。
- Python runtime API には caveat がある。PoC では `state.write_state` が `np.float16` values を reject する一方、`np.float32` values は accepted され、read back も float32 になった。初期実装では「state schema は fp16、write payload は np.float32」として書き込み、`read_state` と `predict` の挙動を P1 で明示検証する。

方針:

- `context_k` / `context_v` / `valid_mask` は `ct.StateType` として定義する。
- Python から `MLState.write_state(...)` で condition KV を一度だけ書き込む。書き込み dtype は上記 caveat に従い、P1 の検証完了までは `np.float32` payload を標準にする。
- denoise loop の各 step は `predict` input を小さく保つ。
- 同一 `MLState` の同時使用は避ける。condition cache は state lease lock または state pool を持つ。

### 通常 input と state

CoreML denoiser step の通常 input:

```text
x_t          fp16 [1, S_bucket, patched_latent_dim]
t            fp16 [1] or [B_eff]、model 内部で branch 展開する
latent_mask  fp16/bool [1, S_bucket]
step_flags   small int/float flags: cfg_active, speaker_kv_scaled_active, alt_branch_index
```

CoreML `MLState`:

```text
context_k_state     fp16 [L, B_eff, C_ctx_bucket, H, D]
context_v_state     fp16 [L, B_eff, C_ctx_bucket, H, D]
valid_mask_state    fp16 [B_eff, C_ctx_bucket]  # 1.0 valid, 0.0 invalid
cfg_scale_state     fp16 [B_eff]                # cond branch is 0 or unused
layout_state        small int/float             # optional; preferably fixed by compiled model
```

`C_ctx_bucket = T_bucket + R_bucket (+ C_bucket)`。DEFAULT_CHECKPOINT では caption が無効なので `T_bucket + R_bucket` でよい。

mask semantics は current explicit attention implementation と PoC に合わせる。`valid_mask_state` は token が有効なら `1.0`、padding/無効なら `0.0` を持つ。CoreML model 内で additive mask を次の形に変換する。

```text
additive_mask = (1.0 - valid_mask_state) * -10000.0
```

重要: `T_bucket=256`, `R_bucket=160`, `H=20`, `D=64`, `L=12` の fp16 condition KV は 1 branch あたり `25,559,040 bytes` になる。

```text
2(K,V) * (256 + 160) * 20 * 64 * 12 * 2 bytes = 25,559,040 bytes
```

実際の `speaker_context_len=138` でも、StateType は fixed shape なので `R_bucket=160` 分を占有する。`cond1 + independent_text_speaker3` は合計 4 branches なので、KV だけで `102,236,160 bytes` になる。これは masks/scales/state object overhead を含まない。

これを毎 step の normal input として渡すと、40 steps で約 1.02GB の input 転送相当になり、CoreML/ANE の計算改善を潰す。`MLState` に保持し、step input は latent と scalar だけにする。

### bucketized fixed shapes

CoreML / ANE では dynamic shape を広く取るより、固定 shape の bucket を事前 compile する。

推奨 bucket:

- `S_bucket`: 64, 80, 100, 128, 160, 200
- `T_bucket`: 64, 128, 256
- `R_bucket`: 64, 96, 128, 160, 192
- `branch_layout`: `cond1`, `independent_text_speaker3`, `joint2`, `alternating_text2`, `alternating_speaker2`

初期 production target は PoC に合わせて `S=100/T=256/R=160/cond1` と `S=100/T=256/R=160/independent_text_speaker3` から始める。実測で hit rate が低ければ bucket を増やす。

### precompiled mlpackages

mlpackage は以下で分ける。

- model fingerprint
- `S_bucket/T_bucket/R_bucket/C_bucket`
- CFG mode / branch layout
- speaker caption condition 有無
- state layout version

起動時に `CoreMLDenoiserBackend` が registry を読み、初回 request 前によく使う bucket を compile/load する。`compute_precision=float16` と `ct.ComputeUnit.CPU_AND_NE` を標準にし、`MLComputePlan` で `linear`, `matmul`, `softmax` が NE に乗っていることを検証する。

警告: 現在の real-model PoC が証明したのは no-context-kv-cache CoreML model の NE placement であり、stateful KV model の placement ではない。`read_state` や `slice_by_index` を入れた full denoiser で、再度 `MLComputePlan` を確認する必要がある。tiny `read_state` PoC は API feasibility の確認にはなるが、real-model placement の結論には使えない。

### denoise step の流れ

1. `ConditionCacheManager` が condition cache を lookup する。
2. `CoreMLDenoiserBackend` が bucket と branch layout に合う compiled model を取得する。
3. CFG inactive step では `cond1` state/model を使う。
4. CFG active step では `independent_text_speaker3` などの CFG state/model を使う。
5. model は `MLState` から per-layer context K/V を読み、self K/V と concat して attention を実行する。
6. model は branch outputs を内部で CFG 合成して `v` を返す。Python は Euler update のみ行う。

CFG 合成を CoreML model 内に入れる理由は、active step で `B_eff` 個の output を Python に戻して chunk/combine する転送と Python overhead を避けるためである。ただし初期検証では branch output を返して Python で既存式と一致確認し、その後 CoreML 内合成へ移す。

## CFG の扱い

### independent CFG default

DEFAULT_CHECKPOINT では caption が無効、speaker condition が有効である。既存 default は `cfg_guidance_mode="independent"`、`cfg_scale_text=3.0`、`cfg_scale_speaker=5.0` なので active window では次の 3 branches が必要になる。

```text
branch 0: cond            text_cond     speaker_cond
branch 1: text_uncond     text_zero     speaker_cond
branch 2: speaker_uncond  text_cond     speaker_zero
```

したがって condition cache は cond だけでなく、text-uncond と speaker-uncond の KV も持つ。`cfg_min_t/cfg_max_t` の外では cond-only cache を使えるため、active window 以外で `B_eff=3` を走らせない。

`num_steps=40`, `cfg_min_t=0.5`, `cfg_max_t=1.0` の既存 schedule は `t_schedule = linspace(1.0, 0.0, 41) * 0.999` なので、概ね前半 20 steps が CFG active、後半 20 steps が cond-only になる。

### joint mode

`joint` は cond branch と joint-uncond branch の 2 branches が基本になる。複数 CFG target の scale が一致しない場合は既存コードと同様に reject する。state layout は `joint2` とし、`cfg_scale_state` は single scale を持つ。

### alternating mode

`alternating` は step index により uncond 対象が切り替わる。設計は 2 案ある。

- 案 A: `cond + text_uncond`、`cond + speaker_uncond` の state を別々に持ち、`alt_branch_index` で model/state を切り替える。
- 案 B: `cond + text_uncond + speaker_uncond (+ caption_uncond)` の state を持ち、step flag で使う uncond branch だけを選ぶ。

初期実装は案 A が単純で、不要 branch を active step で計算しない。model 数は増えるが、maximum performance には合う。

### speaker_kv_scale

既存 `speaker_kv_scale` は projected speaker K/V に直接 in-place scale をかけ、`speaker_kv_min_t` を跨いだら inverse scale で戻す。CoreML では state を step 中に書き換えない方が安全なので、以下を推奨する。

- `speaker_kv_scale is None`: normal state のみ。
- `speaker_kv_scale != None`: scaled state と normal state を両方 prepare し、`speaker_kv_min_t` を跨ぐ step で state を切り替える。
- `speaker_kv_max_layers` は state creation 時に対象 layer の speaker slice のみ scale する。

## 公開 HTTP Interface

### 共通規約

- JSON API は `application/json`。
- 音声 response は既存 `/v1/audio/speech` と同じく binary body を返す。
- 互換拡張は top-level の `irodori` object に入れる。
- `cache_mode`:
  - `off`: cache を使わない。
  - `auto`: cache があれば使い、なければ従来経路で合成する。
  - `prepare`: cache がなければ作る。
  - `require`: cache がなければ 404/410/409 を返す。
  - `refresh`: 既存 cache を捨てて作り直す。
- cache id prefix:
  - `ref_...`: reference cache
  - `cond_...`: condition cache
  - `cg_...`: multi-segment condition cache group

### `POST /v1/tts/reference-caches`

参照音声由来の cache を作成または再利用する。synchronous ready を基本にし、長時間処理が必要な場合のみ `202 Accepted` を許可する。

Request:

```json
{
  "source": {
    "type": "server_default"
  },
  "cache_mode": "create_or_reuse",
  "ttl_seconds": 3600,
  "normalize_db": -16.0,
  "max_ref_seconds": 30.0,
  "metadata": {
    "label": "default-speaker"
  }
}
```

別 source 例:

```json
{
  "source": {
    "type": "base64",
    "media_type": "audio/wav",
    "data": "UklGR..."
  },
  "cache_mode": "create_or_reuse",
  "ttl_seconds": 3600
}
```

Response `201 Created`:

```json
{
  "id": "ref_01hvx4r8q3y6v9",
  "status": "ready",
  "reused": false,
  "model": "Aratako/Irodori-TTS-500M-v2",
  "model_fingerprint": "sha256:...",
  "reference_fingerprint": "sha256:...",
  "layers": {
    "ref_latent": true,
    "speaker_state": true,
    "speaker_kv": "lazy"
  },
  "shapes": {
    "ref_len": 137,
    "speaker_context_len": 138,
    "speaker_dim": 768
  },
  "memory_bytes": 0,
  "expires_at": "2026-05-08T10:30:00+08:00"
}
```

Status codes:

- `200 OK`: 既存 cache を再利用。
- `201 Created`: 新規作成。
- `202 Accepted`: async prepare 中。
- `400 Bad Request`: source 指定不正。
- `413 Payload Too Large`: base64/file が上限超過。
- `415 Unsupported Media Type`: decode 非対応。
- `422 Unprocessable Entity`: reference が短すぎる、または encode 不能。
- `507 Insufficient Storage`: cache memory/disk budget 超過。

### `GET /v1/tts/reference-caches/{id}`

Response `200 OK`:

```json
{
  "id": "ref_01hvx4r8q3y6v9",
  "status": "ready",
  "model_fingerprint": "sha256:...",
  "reference_fingerprint": "sha256:...",
  "created_at": "2026-05-08T09:30:00+08:00",
  "last_used_at": "2026-05-08T09:35:21+08:00",
  "expires_at": "2026-05-08T10:30:00+08:00",
  "hit_count": 4,
  "shapes": {
    "ref_len": 137,
    "speaker_context_len": 138
  },
  "resident_layers": ["ref_latent", "speaker_state"],
  "resident_buckets": ["S100_T256_R160_cond1"]
}
```

Status codes: `200`, `404`, `410`。

### `DELETE /v1/tts/reference-caches/{id}`

Response: `204 No Content`。

Status codes: `204`, `404`。reference cache を消す場合、それを参照する condition cache は cascade evict するか `409 Conflict` で拒否する。初期実装は cascade evict を推奨する。

### `POST /v1/tts/condition-caches`

text/chunk と reference cache から、CoreML `MLState` を含む condition cache を作る。

Request:

```json
{
  "reference_cache_id": "ref_01hvx4r8q3y6v9",
  "input": "こんにちは。今日は良い天気ですね。",
  "caption": null,
  "seconds": 4.0,
  "num_steps": 40,
  "cache_mode": "create_or_reuse",
  "ttl_seconds": 900,
  "bucket": {
    "sequence_length": 100,
    "text_len": 256,
    "speaker_context_len": 160
  },
  "cfg": {
    "mode": "independent",
    "scale_text": 3.0,
    "scale_speaker": 5.0,
    "scale_caption": 0.0,
    "min_t": 0.5,
    "max_t": 1.0
  },
  "coreml": {
    "compute_units": "CPU_AND_NE",
    "precision": "float16"
  }
}
```

Response `201 Created`:

```json
{
  "id": "cond_01hvx55m0p8f2a",
  "status": "ready",
  "reused": false,
  "reference_cache_id": "ref_01hvx4r8q3y6v9",
  "model_fingerprint": "sha256:...",
  "condition_fingerprint": "sha256:...",
  "bucket_id": "S100_T256_R160_independent_text_speaker3",
  "branch_layouts": {
    "cond": "cond1",
    "cfg_active": "independent_text_speaker3"
  },
  "shapes": {
    "sequence_length": 100,
    "text_len": 256,
    "speaker_context_len": 138,
    "speaker_context_len_bucket": 160,
    "branches_active": 3
  },
  "memory_bytes": 102236160,
  "expires_at": "2026-05-08T09:45:00+08:00"
}
```

`memory_bytes` は `cond1` 1 branch と `independent_text_speaker3` 3 branches の KV state 合計である。actual `speaker_context_len=138` でも fixed `R_bucket=160` の StateType に pad されるため、計算は bucket size で行う。

Status codes:

- `200 OK`: 既存 condition cache を再利用。
- `201 Created`: 新規作成。
- `202 Accepted`: async prepare 中。
- `400 Bad Request`: input/cfg/bucket 指定不正。
- `404 Not Found`: reference cache なし。
- `409 Conflict`: model/bucket/CFG が既存 resource と非互換。
- `410 Gone`: reference cache 期限切れ。
- `422 Unprocessable Entity`: text が token 上限を超え bucket に収まらない。
- `503 Service Unavailable`: 対応 mlpackage 未 load。
- `507 Insufficient Storage`: state memory budget 超過。

### `GET /v1/tts/condition-caches/{id}`

Response `200 OK`:

```json
{
  "id": "cond_01hvx55m0p8f2a",
  "status": "ready",
  "reference_cache_id": "ref_01hvx4r8q3y6v9",
  "bucket_id": "S100_T256_R160_independent_text_speaker3",
  "cfg": {
    "mode": "independent",
    "active_steps_estimate": 20,
    "branches_active": 3
  },
  "resident_states": ["cond1", "independent_text_speaker3"],
  "created_at": "2026-05-08T09:30:15+08:00",
  "last_used_at": null,
  "hit_count": 0
}
```

Status codes: `200`, `404`, `410`。

### `DELETE /v1/tts/condition-caches/{id}`

Response: `204 No Content`。

Status codes: `204`, `404`。

### `POST /v1/audio/speech` extension

既存 OpenAI-compatible endpoint はそのまま残す。`irodori.cache_id` と `irodori.cache_mode` を追加で受ける。response は従来通り requested audio format の binary。

Request:

```json
{
  "model": "irodori-tts-500m-v2",
  "voice": "default",
  "input": "こんにちは。今日は良い天気ですね。",
  "response_format": "mp3",
  "num_steps": 40,
  "seed": 1234,
  "irodori": {
    "cache_id": "cond_01hvx55m0p8f2a",
    "cache_mode": "require"
  }
}
```

Behavior:

- `cache_mode=require`: `cache_id` がない、期限切れ、または input/seconds/num_steps/CFG/bucket と一致しない場合は audio を生成せず error。
- `cache_mode=auto`: `cache_id` が有効なら使う。なければ従来の `runtime.synthesize` 経路。
- `cache_mode=prepare`: cache がなければ内部で condition cache を作るが、binary response の前に準備時間が入る。
- multi-segment input の場合は `irodori.cache_ids` または `cg_...` group id を使う。単一 `cond_...` を multi-segment に流用しようとしたら `409 Conflict`。

Error examples:

```json
{
  "error": {
    "type": "cache_mismatch",
    "message": "condition cache input fingerprint does not match request input/seconds/cfg",
    "cache_id": "cond_01hvx55m0p8f2a"
  }
}
```

Status codes:

- `200 OK`: audio binary。
- `400 Bad Request`: extension fields 不正。
- `404 Not Found`: `cache_mode=require` で cache なし。
- `409 Conflict`: cache と request が非互換。
- `410 Gone`: cache 期限切れ。
- `422 Unprocessable Entity`: bucket に収まらない。
- `503 Service Unavailable`: CoreML backend unavailable。

### `POST /v1/audio/speech/prepare-and-synthesize`

便利 endpoint としては有用だが、必須ではない。HTTP round trip を 1 回にまとめたいクライアント向けに、reference cache と condition cache を内部作成してから合成する。

採用条件:

- mobile/web client が `reference-caches` と `condition-caches` の 2 段階 API を扱いにくい。
- 同じ request 内では cache を使い、終了後 TTL で短期保持したい。

Request:

```json
{
  "input": "こんにちは。",
  "response_format": "wav",
  "reference": {
    "source": {
      "type": "server_default"
    }
  },
  "irodori": {
    "cache_mode": "prepare",
    "ttl_seconds": 300
  }
}
```

Response は audio binary。cache metadata は header に入れる。

```text
X-Irodori-Reference-Cache-Id: ref_...
X-Irodori-Condition-Cache-Id: cond_...
X-Irodori-Cache-Hit: reference;condition
```

## 内部 Python Interface

### dataclasses

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


CacheStatus = Literal["preparing", "ready", "failed", "evicted", "expired"]
CFGMode = Literal["independent", "joint", "alternating"]


@dataclass(frozen=True)
class ReferenceCacheHandle:
    id: str
    status: CacheStatus
    model_fingerprint: str
    codec_fingerprint: str
    reference_fingerprint: str
    ref_latent_key: str | None
    speaker_state_key: str | None
    speaker_mask_key: str | None
    speaker_context_len: int
    created_at: datetime
    expires_at: datetime | None
    memory_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionCacheHandle:
    id: str
    status: CacheStatus
    reference_cache_id: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    condition_fingerprint: str
    bucket_id: str
    sequence_length: int
    text_len: int
    speaker_context_len: int
    cfg_mode: CFGMode
    branch_layouts: tuple[str, ...]
    mlstate_keys: tuple[str, ...]
    created_at: datetime
    expires_at: datetime | None
    memory_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)
```

### `CoreMLDenoiserBackend`

```python
class CoreMLDenoiserBackend:
    def __init__(
        self,
        model_registry: CoreMLModelRegistry,
        compute_units: str = "CPU_AND_NE",
        precision: str = "float16",
    ) -> None: ...

    def load_bucket_model(self, bucket_id: str, branch_layout: str) -> CoreMLBucketModel: ...

    def make_condition_state(
        self,
        *,
        bucket_id: str,
        branch_layout: str,
        packed_context_k: np.ndarray,
        packed_context_v: np.ndarray,
        packed_valid_mask: np.ndarray,
        cfg_scales: np.ndarray | None,
    ) -> MLStateHandle: ...

    def predict_step(
        self,
        *,
        state: MLStateHandle,
        bucket_id: str,
        branch_layout: str,
        x_t: np.ndarray,
        t: float,
        latent_mask: np.ndarray,
        step_flags: StepFlags,
    ) -> np.ndarray: ...

    def synthesize_with_condition_cache(
        self,
        *,
        condition_cache: ConditionCacheHandle,
        num_steps: int,
        seed: int,
        cfg_window: tuple[float, float],
    ) -> np.ndarray: ...
```

### cache managers

```python
class ReferenceCacheManager:
    def prepare_reference_cache(self, request: ReferenceCacheRequest) -> ReferenceCacheHandle: ...

    def get(self, cache_id: str) -> ReferenceCacheHandle: ...

    def evict_cache(self, cache_id: str, *, cascade: bool = True) -> None: ...


class ConditionCacheManager:
    def prepare_condition_cache(
        self,
        request: ConditionCacheRequest,
        *,
        reference_cache_id: str,
    ) -> ConditionCacheHandle: ...

    def synthesize_with_condition_cache(
        self,
        cache_id: str,
        request: SamplingRequest,
    ) -> SamplingResult: ...

    def evict_cache(self, cache_id: str) -> None: ...
```

### manager responsibilities

`ReferenceCacheManager`:

- audio source を canonicalize して fingerprint を作る。
- reference wav を decode / normalize / crop する。
- `ref_latent/ref_mask` を作る。
- `speaker_state/speaker_mask` を作る。
- speaker_kv は bucket 依存なので lazy にする。ただし hot bucket は background warmup してよい。

`ConditionCacheManager`:

- text/caption を normalize/tokenize する。
- `text_state/text_mask` を作る。
- reference cache から `speaker_state/speaker_mask` を取得する。
- CFG mode に応じて cond/uncond bundles を作る。
- per-layer KV を作り、bucket padding して fp16 schema の packed state に変換する。P1 検証完了までは `state.write_state` には `np.float32` payload を渡す。
- `CoreMLDenoiserBackend.make_condition_state()` を呼んで `MLState` を作る。
- TTL/LRU/metrics を管理する。

`CoreMLDenoiserBackend`:

- CoreML model load/compile/cache を管理する。
- `MLState` の生成、書き込み、lease、predict serialization を管理する。
- `MLComputePlan` 検証結果を metrics に出す。
- PyTorch fallback は明示 opt-in にする。

## Performance Estimate

### single conditioned step

既存 PoC:

- CoreML no-context-kv-cache: `14.428ms`
- no-cache total MAC: 約 `35.63G`
- context KV projection: 約 `7.28G MAC`
- context KV share: 約 `20.4%`
- cached total MAC: 約 `28.35G`

単純な MAC 比では `14.428ms * 28.35 / 35.63 = 11.48ms`。CoreML runtime overhead と state read を考慮し、cached single-conditioned step は `11.5-12.5ms` と見積もる。これは no-cache CoreML からさらに `15-25%` の改善である。ただし、この数値は stateful KV placement が no-cache PoC と同程度に NE に乗るという仮定を含む。`read_state` / state slicing 追加後の実測で更新する。

PyTorch MPS cached `43.881ms` と比べると、CoreML cached `11.5-12.5ms` は single conditioned step で約 `3.5-3.8x`。

### independent CFG, 40 steps

DEFAULT independent CFG では active step が `B_eff=3` になる。`num_steps=40`, `cfg_min_t=0.5`, `cfg_max_t=1.0` では概ね 20 steps active、20 steps inactive。

active step は 3 branch を batch 化できるが、NE が完全に 1 branch と同じ時間で処理するわけではない。ここは必ず実測する。

見積もり:

- inactive cond-only: `11.5-12.5ms/step`
- active optimistic: `23-28ms/step` (`B_eff=3` だが 1.8-2.2x 程度で収まる仮定)
- active conservative: `31-38ms/step` (ほぼ 2.7-3.0x の仮定)
- total denoiser optimistic: `20 * 11.5-12.5 + 20 * 23-28 = 約 0.69-0.81s`
- total denoiser conservative: `20 * 11.5-12.5 + 20 * 31-38 = 約 0.85-1.01s`

この範囲は denoiser のみである。end-to-end latency には tokenize、reference encode、condition state 作成、DACVAE decode、MP3/WAV serialization、HTTP response 書き出しが入る。reference cache は繰り返し request の reference preparation と CoreML state creation を主に削減する。初回 request の end-to-end は cache preparation の分だけ重い。

## 実装ロードマップ

### P0: doc + benchmark baseline

- 本設計書を追加する。
- 現行 PyTorch cached/no-cache と CoreML no-cache PoC を同条件で再測定する。
- `S=100/T=256/R=138 or 160/steps=40` の active CFG 比率をログに出す。

### P1: stateful CoreML denoiser

- PyTorch wrapper を作り、conditioning KV を state buffer として register する。
- registered torch buffers と `ct.StateType` wrapped `TensorType` は fp16 に固定する。
- `ct.StateType` / `MLState` を使って `mlprogram` を変換する。
- `minimum_deployment_target=ct.target.iOS18`、`compute_precision=float16`、`CPU_AND_NE`。`ct.target.macOS15` は使わない。
- `state.write_state` は初期実装で `np.float32` payload を使い、fp16 state schema に対する read/predict behavior を検証する。
- state write once, predict many の loop benchmark を作る。
- `read_state` / `slice_by_index` を含む stateful full denoiser で `MLComputePlan` を再取得し、`linear/matmul/softmax` の NE-preferred counts を報告する。
- P1 acceptance gate は tiny `read_state` PoC ではなく、stateful full denoiser の read/predict correctness と NE placement report とする。

### P2: reference cache manager + HTTP endpoints

- `ReferenceCacheManager` を追加する。
- `POST/GET/DELETE /v1/tts/reference-caches` を追加する。
- server default reference と base64 upload を扱う。
- TTL/LRU と memory accounting を入れる。

### P3: condition cache manager + `/v1/audio/speech` extension

- `ConditionCacheManager` を追加する。
- `POST/GET/DELETE /v1/tts/condition-caches` を追加する。
- `/v1/audio/speech` の `irodori.cache_id/cache_mode` を追加する。
- cache mismatch を `409` として返す。

### P4: CFG branch/state layout support

- independent `cond1` + `independent_text_speaker3` を実装する。
- joint `joint2` を実装する。
- alternating は state split 案 A から実装する。
- `speaker_kv_scale` は normal/scaled state 切替で対応する。

### P5: precompile buckets + LRU eviction + metrics

- hot buckets を起動時 warmup する。
- bucket miss 時の fallback 方針を決める。
- cache metrics:
  - reference hit/miss
  - condition hit/miss
  - state creation ms
  - denoiser predict ms per step/mode
  - NE placement summary
  - evictions and memory bytes

### P6: optional CoreML DACVAE/encoder migration

- end-to-end の bottleneck が denoiser 以外に移った後に着手する。
- DACVAE decode、reference encode、text/speaker encoder の CoreML 化を検討する。
- P1-P5 の API はこの移行後も互換維持する。

## Acceptance Criteria

- stateful CoreML denoiser が PyTorch cached path と同じ branch semantics を持つ。
- Stateful conversion は fp16 registered buffer と fp16 `ct.StateType` で成功する。
- `state.write_state` の dtype 方針がテストで固定され、`np.float32` payload 書き込み後の `read_state` / `predict` 結果が検証されている。
- `rel_diff` は既存 PoC の `0.0251` と同程度、初期 gate は `<= 0.03` を目安にする。
- CoreML `predict` の normal input に per-layer KV tensor が含まれていない。
- `MLComputePlan` は no-context-kv-cache PoC ではなく、stateful full denoiser で再確認する。`read_state` / `slice_by_index` 導入後の `linear/matmul/softmax` NE-preferred counts を記録する。
- single conditioned cached step が `11.5-12.5ms` 付近、少なくとも no-cache CoreML `14.428ms` より速い。
- independent CFG active step と 40-step denoiser time を実測値として記録する。
- `/v1/audio/speech` 既存 request は拡張なしで従来通り動く。
- `irodori.cache_mode=require` は cache miss/mismatch 時に audio を生成しない。
- reference cache delete で dependent condition cache が安全に evict される。
- 同一 `MLState` の concurrent predict が起きない。

## Validation Commands / Tests

ドキュメント追加のみ:

```bash
git status --short --branch
ls -l docs/coreml-kv-cache-api-design.md
sed -n '1,260p' docs/coreml-kv-cache-api-design.md
```

既存 import / PoC tests:

```bash
python -m pytest tests/test_coreml_real_step_benchmark_import.py tests/test_coreml_ane_poc.py
```

CoreML 実機 benchmark:

```bash
uv run --with 'coremltools>=8.0' python tools/coreml_real_step_benchmark.py \
  --seconds 4 \
  --sequence-length 100 \
  --compute-precision float16 \
  --iterations 3 \
  --warmup 1
```

P1 追加後に必要な新規 tests:

```bash
python -m pytest tests/test_coreml_stateful_denoiser.py
python -m pytest tests/test_tts_cache_api.py
```

追加すべき assertions:

- stateful CoreML output と PyTorch cached output の numerical diff。
- `predict` input keys に `context_kv` が存在しない。
- state tensors の schema が fp16 で、`valid_mask_state` が `1.0 valid / 0.0 invalid` になっている。
- condition cache が `cond1` と CFG active layout の両方を持つ。
- cache mismatch で `409`。
- expired cache で `410`。
- memory budget 超過で `507`。

## Risks

- CoreML stateful model の PyTorch tracing/export が現在の module 構造と相性が悪い可能性がある。その場合は MIL builder で state read を明示する。
- fp32 `StateType` は conversion で失敗する。state schema は fp16 に固定し、buffer dtype の混入を CI で検出する必要がある。
- `MLState` の state write/read API は coremltools/macOS version 依存があるため、P1 で最小 reproducer を先に固める。
- `state.write_state` の write payload dtype は直感と異なる可能性がある。PoC では `np.float16` が reject され、`np.float32` が accepted/read back float32 だったため、predict path での実効 dtype を必ず確認する。
- tiny `read_state` PoC は state API の確認にすぎず、real-model の ANE placement を保証しない。
- `MLState` は同一 state の concurrent prediction が unsafe なので、HTTP 並列 request では state lease/pool が必要。
- B_eff=3 の active CFG は single branch の 3 倍未満になる保証がない。必ず実測して bucket/model layout を調整する。
- state memory が大きい。`T=256/R_bucket=160` の independent CFG は 1 condition chunk で `cond1 + B_eff=3` を持つと KV だけで `102,236,160 bytes` になる。
- bucket が細かすぎると compile/load artifact が増える。粗すぎると padding 計算が増える。
- fp16 CoreML の `rel_diff=0.0251` は既に小さくない。音質 regression は数値 diff だけで判断しない。
- DACVAE decode と audio serialization が end-to-end bottleneck になる可能性がある。

## Non-goals

- raw wav や最終 generated audio の response cache は本設計の主対象ではない。
- `/v1/audio/speech` の OpenAI-compatible binary response contract は壊さない。
- P1-P5 では DACVAE / tokenizer / text encoder / speaker encoder の全面 CoreML 化はしない。
- 全 sequence length を単一 dynamic CoreML model で最適化することは狙わない。
- process restart 後も `MLState` を永続復元する設計は初期 scope 外。metadata と source fingerprint から再作成する。
- 品質を犠牲にして steps や CFG を減らす高速化はこの設計の対象外。
