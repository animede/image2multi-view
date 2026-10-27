# flux3-klein-studio 高速化の調査と提案(2026-10-27)

対象: `~/flux3-klein-studio`(FLUX 3 Image Klein distilled、INT8 ConvRot、Diffusers ModularPipeline)。
環境: RTX PRO 6000 Blackwell 96GB、torch 2.14.1+cu130、diffusers 0.41.0.dev0、comfy-kitchen 0.2.37。
計測は studio と同じ `Flux3ConvRotPipeline` を別プロセスで動かし、studio がアイドルのときだけ測っています(参照 1 枚、同じプロンプト、seed 42)。

## 1. 現状の内訳(どこに時間がかかっているか)

| 段階 | 512² ドラフト(参照 512) | 1024² 最終(参照 1024) |
|---|---:|---:|
| テキストエンコード(Qwen3-VL-4B、CPU メモリ→1 層ずつ GPU) | **0.83 s** | 0.83 s |
| VAE エンコード(参照) | 0.06 s | 0.12 s |
| ノイズ除去 16 step | 1.13 s(70 ms/step) | 5.72 s(356 ms/step) |
| VAE デコード | 0.03 s | 0.14 s |
| 合計 | **2.07 s** | **6.8 s** |

- 512² のドラフトでは **テキストエンコードが 40%**。プロンプトは 320 トークン(80 の倍数にパディング)で、
  「CPU メモリに置いて 1 層ずつ GPU に送る」方式は 8GB GPU 向けの妥協で、この GPU では純粋な損です。
- 1024² ではノイズ除去が 85%。1 step あたり 356 ms で、トークン数は 4096(出力)+ 4096(参照)+ 320(文)。
- ModularPipeline 自体のオーバーヘッドは 0.01 s 程度で無視できます。

## 2. 実測した改善策

### 2.1 テキストエンコーダを GPU に常駐させる(効果大、実装小)

`accelerate.cpu_offload` のフックを外して `text_encoder.to("cuda")` にするだけです。VRAM は +8.3GiB(合計 14.9GiB)。

| | 512² 16 step | 512² 8 step | 1024² 16 step |
|---|---:|---:|---:|
| 現状(オフロード) | 2.07 s | 1.5 s(推定) | 6.8 s |
| GPU 常駐 | **1.2 s** | **0.66 s** | **5.9 s** |

テキストエンコードは 0.83 s → **0.02 s**。結果は同じ(エンコーダの重みは同一、計算順序が変わるだけ)。

実装案: `KLEIN_TEXT_ENCODER=gpu` を追加し(既存の `cpu` / `gpu-offload` と並ぶ 3 つ目)、`flux3_int8.py` の `__init__` で
`cpu_offload(...)` の代わりに `.to(device)` を呼ぶ。`KLEIN_MEMORY_BUDGET_GIB` は 8 のままだと足りないので、`gpu` のときは
既定を 24 程度に上げる(2K は `KLEIN_HIRES_MEMORY_BUDGET_GIB` も同様)。空き VRAM が足りなければ `gpu-offload` に自動で落とす。

### 2.2 プロンプト埋め込みのキャッシュ(2.1 の代替・補完)

同じキャプションの 2 回目は `prompt_embeds`(1×320×20480、bf16 で 13MB)を再利用すれば、エンコードは 0 になります。
image2multi-view ではドラフトと最終が同じキャプションなので、最終の 0.83 s がそのまま消えます。
2.1 を入れるなら効果は 0.02 s 分だけなので優先度は下がりますが、8GB 環境のまま速くしたい場合には有効です。
Diffusers には FLUX.1 向けの `CachedFluxTextEncoderStep` カスタムブロックの前例があります。

### 2.3 注意機構(attention)のバックエンド切替: 効果なし

`transformer.set_attention_backend()` で比較。既定(SDPA)はすでに flash カーネルを選んでいました。

| backend | 512² | 1024² | 既定との PSNR |
|---|---:|---:|---:|
| 既定 / `native` / `_native_flash` | 69 ms/step | 356 ms/step | 同一 |
| `_native_cudnn` | 98 ms/step | 363 ms/step | 47〜53 dB |
| `_native_efficient` | 77 ms/step | 466 ms/step | 50 dB |

`flash_4_hub`・`sage_blackwell_hub`(Hugging Face `kernels` 経由)は未インストールのため未計測。SGLang の記録では
Blackwell で FA4 と SDPA の差はわずか(むしろ SDPA が速い例もある)なので、期待値は小さいです。

### 2.4 torch.compile: 現在の INT8 層では不可

`torch.compile(transformer.forward)` は comfy-kitchen の `int8_linear` が DLPack 経由でテンソルを渡すところで
graph break になり、速くなりませんでした(「Developer debug context: tensor.__dlpack__」)。
comfy-kitchen 側がカスタム op として登録されるまでは、INT8 ConvRot と compile は組み合わせられません。

### 2.5 公式 fp8r 版(rowwise FP8)

BFL 公式の `variants/fp8r/model.safetensors`(rowwise FP8、5.6GB)を Diffusers の `BFLQuantizationConfig` で読み込み、
INT8 ConvRot(comfy-kitchen)と同条件で比較しました。読み込みは 0.6 秒、VRAM は INT8 より +5.9GiB。

| | INT8 ConvRot(現状) | fp8r(Diffusers、`torch._scaled_mm`) | fp8r vs INT8 の PSNR |
|---|---:|---:|---:|
| 512² 16 step | 69 ms/step(1.2 s) | **89 ms/step(1.53 s)** | 36.6 dB(絵はほぼ同じ) |
| 512² 8 step | 0.66 s | 0.83 s | |
| 1024² 16 step | 354 ms/step(5.9 s) | **486 ms/step(8.05 s)** | 33.4 dB |

**fp8r は遅い**。Diffusers の `BFLFP8RLinear.forward` は、活性化の行単位量子化を素の PyTorch で
`reshape → float() → amax → div/clamp → cast` と 5 つのカーネルに分けて行い(fp32 への拡張で転送量も 2 倍)、
それが 1 step に 234 層分乗ります。comfy-kitchen の INT8 ConvRot はアダマール回転と量子化を融合した 1 カーネルなので、
Blackwell でも現状の INT8 のほうが速い。fp8r が生きるのは、この量子化が融合されたカーネル(SGLang の FlashInfer PrimsTS GEMM
や、下の torch.compile)を使うときだけです。

#### fp8r + torch.compile: 現状より 1.3〜1.5 倍(実測した中で最速)

fp8r 版は comfy-kitchen の DLPack を通らない素の PyTorch 演算なので、`torch.compile(transformer.forward, dynamic=False)` が
graph break なしに通り、上の 5 演算の量子化が GEMM の前処理として融合されます。

| | INT8 ConvRot(現状) | fp8r + compile | 削減 |
|---|---:|---:|---:|
| 512² 16 step | 69 ms/step、合計 1.2 s | **52 ms/step、合計 0.95 s** | −21% |
| 512² 8 step | 0.66 s | 約 0.5 s(推定) | |
| 1024² 16 step | 354 ms/step、合計 5.9 s | **231 ms/step、合計 3.9 s** | −34% |

コンパイルは出力サイズごとに 1 回ずつ約 30 秒かかります(512² で 28.8 s、1024² で 31.1 s。`dynamic=False` のため形状が
変わると再コンパイル)。studio の実運用では、起動時に使うサイズ(ドラフト 512²・最終 1024² など)をウォームアップするか、
`dynamic=True` / `mark_dynamic` で形状を一般化する必要があります。VRAM は fp8r の transformer が 5.9GiB 増(INT8 比)で、
テキストエンコーダ常駐と合わせて約 21GiB。

### 2.6 キャンセル時のモデル解放をやめる(効果大、実装小)

`app.py` の `except Cancelled:` で `engine.unload()` を呼んでいるため、停止した次のジョブは Transformer・VAE・テキストエンコーダを
読み直します。**読み直しの実測は 3.2〜3.9 秒**(`Flux3ConvRotPipeline` の構築時間)。image2multi-view はドラッグのたびに
最終生成を停止するので、停止直後のドラフトが 1.9 s → 3.5 s になっていました(2.1 節の実測でも確認)。

提案: 停止ではモデルを保持し、解放は OOM・モデル切替・明示的な `/api/unload` だけにする。停止は Transformer の
forward の pre-hook で例外を投げて抜けるだけなので、状態は残りません(ModularPipeline の state はジョブごとに作り直す)。
8GB 環境でも、停止後に VRAM を返す必要はありません(次のジョブで同じだけ使う)。

### 2.7 最終生成の参照画像サイズとステップ数(品質とのトレードオフ)

1024² の最終生成で、参照画像の辺とステップ数だけを変えた結果(INT8、同じ seed)。

| 設定 | ノイズ除去 | 削減 | 見た目 |
|---|---:|---:|---|
| 参照 1024px、16 step(現状) | 5.63 s | — | 基準 |
| 参照 768px | **4.04 s** | −28% | 構図・向き・服は同じ、細部がわずかに変わる(PSNR 27 dB) |
| 参照 512px | 3.05 s | −46% | 同上、顔の細部が少し甘い(PSNR 23 dB) |
| 12 step | 4.32 s | −23% | ほぼ同一(PSNR 37.6 dB) |
| 8 step | 2.87 s | −49% | 細部が柔らかい(PSNR 30.7 dB) |

参照トークンは 1024px で 4096、768px で 2304、512px で 1024。参照 768px + 12 step なら 1024² の最終が約 3.1 秒
(fp8r + compile と組み合わせれば約 2 秒)と見込めます。image2multi-view 側の設定で選べるようにするのが良い。

### 2.8 ステップ数を減らす(実装済み)

同条件で 16 → 8 step はノイズ除去がちょうど半分(512²: 1.1 s → 0.57 s)。蒸留モデルなので 4 step でも構図と向きは
変わらず、細部が柔らかくなるだけでした(image2multi-view のドラフト既定は 8)。`steps` 引数は studio に追加済みです。

## 3. 未計測だが候補になる策(調査結果)

### 3.1 公式 NVFP4 版 + FlashInfer(Blackwell 専用)

Diffusers の `BFLQuantizationConfig` は nvfp4 も読めますが、速いカーネルは **FlashInfer**(`flashinfer-python` と
`flashinfer-cubin` を同じバージョンで)がある場合だけで、無いと毎ステップ bf16 に戻します。今回はパッケージ追加を
行わず未計測。参考値(SGLang、GB300、Base 50 step、1024²): T2I bf16 3.21 s → fp8r 2.18 s → nvfp4 2.14 s、
I2I 6.31 / 4.49 / 4.31 s。**fp8r と nvfp4 の差は数%** で、sm_120(RTX PRO 6000 / 5090)は sm_100 とカーネル経路が別で
成熟度が低い(SGLang も RTX 5090 では SDPA に留めている)。fp8r + compile の後で試す程度。

### 3.2 cache-dit(DBCache)

cache-dit 1.5.x は 2026-10-20 に Diffusers ModularPipeline と FLUX.3 に対応。PR #1124 の実測(RTX PRO 5000、bf16、1024²、16 step):
なし 4.90 s → DBCache F4 B0 W4 R0.30 で 3.86 s(1.27×、PSNR 34 / SSIM 0.957)、fp8 + compile + cache で 1.88 s(2.60×)。
**編集(I2I)では早いステップのキャッシュで SSIM が 0.89 まで落ちる**ため `max_warmup_steps=8` が推奨(1.28×、SSIM 0.967)。
8 step のドラフトにはほぼ余地がなく、効くのは 16 step の最終だけ(5.9 → 4.6 s 程度の見込み)。studio の段階式
パイプラインでは `denoise_pipe` の transformer に掛ける形になり、INT8 の自作層・compile との相性は要確認。

### 3.3 Hugging Face `kernels` 経由の FA4 / SageAttention

`flash_4_hub`・`sage_blackwell_hub` は `pip install kernels` で事前ビルド済みカーネルを取ってきます。2.3 節のとおり
既定がすでに flash 系で、SGLang の記録でも Blackwell で FA4 と SDPA の差はわずか(SSIM 0.98 のずれあり)。
1024² でも attention は 1 step の 3 割程度と推定され、効いても 1 割未満。優先度は低。

### 3.4 CUDA Graph

SGLang の breakable CUDA graph は nvfp4 の distilled T2I でだけ 0.563 → 0.436 s の効果で、他の構成では逆効果と報告。
fp8r + compile に `mode="reduce-overhead"` を足せば同じ効果を狙えます(512² のように起動オーバーヘッド比率が高い条件で
効く可能性)。2.5 の後の小さな追加実験として妥当。

### 3.5 参照画像の VAE 潜在のキャッシュ

同じ参照画像なら VAE エンコード(512 で 0.05 s、1024 で 0.12 s)は 1 回で済みます。効果は小さいが実装も小さい
(参照画像のハッシュ + 辺 → `image_cond_latents` 等の LRU)。

### 3.6 SGLang / vLLM-Omni / TensorRT-LLM への載せ替え

BFL 公式が案内する推論スタックで、FP8R の FlashInfer PrimsTS GEMM(QK-norm / RoPE / SwiGLU 融合)、CUDA Graph、cache-dit、
動的バッチを備えます。GB300 の distilled nvfp4 T2I 0.44 s、H200 fp8r Base I2I 11.2 s など。ただし、
- studio の機能(prompt upsampler の切替、キャラクターシート、bbox、アウトペイント LoRA、ギャラリー)は API の向こうに
  再実装が必要で、
- RTX PRO 6000(sm_120)は SGLang の計測対象(GB300 / H200 / RTX 4090 / 5090)の中では 5090 に近く、そこでは SDPA + eager、
- Docker か main ブランチからのインストールが前提(リリース前)。
現行サーバーを別ポートの SGLang と並べて比較する価値はありますが、「fp8r + compile で 1.3〜1.5 倍」を先に取ってから
判断するのが安全です。

### 3.7 GPU の共有

計測中、同じ GPU 上の他のプロセス(動画生成や音声合成など計 7 プロセス)で VRAM 90〜95GB・使用率 100% の
時間帯があり、1 step が 70 ms → 110〜160 ms に落ちました。ドラフトの体感には、サーバー側の最適化より GPU の同居を
減らすほうが効く場面もあります。

## 4. 提案する順番

| 順 | 施策 | 期待効果 | 工数 | リスク |
|---|---|---|---|---|
| 1 | 停止時に `engine.unload()` しない | 停止直後のジョブ −3.2〜3.9 s | 小 | ほぼ無し |
| 2 | テキストエンコーダ GPU 常駐(`KLEIN_TEXT_ENCODER=gpu`、予算 24GiB) | 全ジョブ −0.8 s(512² ドラフト 2.07 → 1.2 s) | 小 | 8GB 環境は自動フォールバック |
| 3 | 最終生成の参照 768px・12 step を選択可(image2multi-view 側) | 1024² 5.9 → 約 3.1 s | 小 | 細部がわずかに変わる |
| 4 | fp8r 版 + torch.compile の読み込み経路(`KLEIN_TRANSFORMER=fp8r-compile`) | 512² −21%、1024² −34% | 中 | 起動時 30 s × サイズ数のウォームアップ、VRAM +6GiB |
| 5 | プロンプト埋め込み・参照 latent のキャッシュ | 2 回目以降 −0.1 s(2 適用後) | 小 | 無し |
| 6 | cache-dit(最終 16 step のみ、warmup 8) | 1024² −20% | 中 | 編集での品質低下 |
| 7 | nvfp4 + FlashInfer、CUDA Graph、SGLang 比較 | fp8r 比 −数% 〜 −20% | 大 | 依存追加、機能の再実装 |

1〜3 だけで、512² 8 step のドラフトが **約 0.66 s**、1024² の最終が **約 3 s** になります。4 を足すと 512² が約 0.5 s、
1024²(参照 768、12 step)が約 2 s の見込みです。

## 5. 外部提案(2026-10-27 受領)との照合

| 外部提案 | 実測との照合 | 評価 |
|---|---|---|
| 1. 停止時に unload しない | 確認。`app.py` の `except Cancelled` で解放、再読み込み 3.2〜3.9 s | **同意、最優先**(2.6 節) |
| 2. Qwen3-VL 常駐 | 実測 0.83 → 0.02 s、VRAM +8.3GiB | **同意**(2.1 節) |
| 3. 参照 VAE のキャッシュ | VAE エンコードは 0.05〜0.12 s | 効果は小さいが実装も小さい(3.5 節) |
| 4. FP8R / NVFP4、SGLang 等へ移行 | Diffusers の fp8r 単体は INT8 より **遅い**(89 vs 69 ms/step)。compile 併用で 1.3〜1.5 倍 | fp8r + compile は同意。スタック移行は機能の再実装が必要で、本命とするのは早い(3.6 節) |
| 5. FlashAttention 系 | 既定がすでに flash 系で、他のバックエンドは同等以下 | **効果なし**(2.3 節) |
| 6. 最終の参照を 768px に | −28%、肉眼で差なし | **同意**(2.7 節) |
| 7. 最終に 12 step | −23%、PSNR 37.6 dB | **同意**(2.7 節) |
| 8. cache-dit をドラフトで有効 | 8 step ではキャッシュできる余地がほぼ無い。効くのは 16 step の最終 | 向きが逆。最終に warmup 8 で使うのが筋(3.2 節) |
| 9. 動的バッチ | 1 ユーザーの応答には効かない | 同意(低優先) |

## 付録

計測は使い捨てのスクリプトで行いました(studio の `Flux3ConvRotPipeline` を別プロセスで構築し、段階ごとに `torch.cuda.synchronize()` を挟んで計時)。
実装後の再現は、studio の `/api/generate` の `metrics`(`seconds`、`transformer`、`compiled`、`text_encoder`)で確認できます。
