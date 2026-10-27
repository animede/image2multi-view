# image2multi-view

1枚の画像を、マウスで X / Y / Z の 3 軸(±180°、10°刻み)に回して見る Web アプリです。
画像生成は [flux3-klein-studio](../flux3-klein-studio) の API(FLUX 3 Image Klein、参照画像 1 枚の Edit)を使います。

- 動かしている間は小さいドラフト(既定 512²、約 2 秒)を生成し、マウスを離してから 0.5 秒後に正規サイズで再生成します。ボタンを押している間は、止めていてもドラフトだけです(矢印キー・ホイールは最後の操作から 0.5 秒後)。
- 角度 → 英文キャプションの変換が本体です(`viewprompt.py`)。prompt upsampler(LLM)はバイパスし、完成形のキャプションを直接送ります。
- 同じ角度・設定の結果はキャッシュされ、2 回目は即表示します。

## 起動

```bash
# 1. 生成サーバー(別ターミナル)。VRAM が 24GiB 以上あるなら高速プロファイルで(studio の README「高速プロファイル」)
cd ../flux3-klein-studio && KLEIN_TEXT_ENCODER=gpu KLEIN_TRANSFORMER=fp8r KLEIN_COMPILE=1 CUDA_VISIBLE_DEVICES=0 ./run.sh   # http://127.0.0.1:7864

# 2. このアプリ(隣の flux3-klein-studio/.venv をそのまま使う)
./run.sh                                                          # http://127.0.0.1:7870
```

flux3-klein-studio が別の場所・別ポートなら `KLEIN_URL=http://host:port ./run.sh`。自前の venv を使うなら `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`。
`MV_HOST` / `MV_PORT` で待ち受けを変えられます。

## 操作

| 操作 | 動き |
|---|---|
| キャンバスをドラッグ | 横 = Y(被写体の向き)、縦 = X(カメラの上下) |
| Shift + ドラッグ / ホイール | Z(画像の回転) |
| ← → ↑ ↓ / Q E | Y / X / Z を 10° ずつ |
| スライダー、− + | 各軸を直接指定 |
| 履歴のサムネイル | その角度と画像に戻す |

角度の意味(`viewprompt.py` の規約、flux3-klein-studio のキャラクターシートと同じ向き):

| 軸 | + | − |
|---|---|---|
| X(pitch) | カメラが上から見下ろす(90° で真上) | 下から見上げる(−90° で真下) |
| Y(yaw) | 被写体が自分の左を向く(顔は画像の右へ、180° で真後ろ) | 自分の右を向く |
| Z(roll) | 画像が時計回りに回る | 反時計回り |

X が ±90° を超えると、天頂を越えて反対側へ回り込んだ視点(Y+180°、上下逆)として記述します(画面に正規化後の角度を表示)。

## 速くするためにしていること

| 項目 | ドラフト | 最終 |
|---|---|---|
| prompt upsampler | バイパス(`upsample=false`、LLM を呼ばない) | 既定はバイパス。チェックで有効化(数秒〜数十秒増) |
| ステップ数 | 既定 8(4〜16 を選択)。蒸留モデルなので 4 step でも構図と向きは変わらず、細部が柔らかくなるだけ | 16(モデル既定) |
| 出力サイズ | 元画像の縦横比で約 512²(256²〜640² を選択) | 元画像サイズ(≤1MP)/ 長辺 768〜2048 |
| 参照画像 | ドラフトと同じ辺に縮小(Transformer に入るトークン数が減る) | 長辺 1024 |
| キュー | 最新の角度だけを生成(途中の角度は捨てる)、最終生成中に動かしたら停止 | |
| キャッシュ | 角度・サイズ・シード・プロンプトが同じなら即表示 | |

実測(RTX PRO 6000、INT8、16 step): 512² ドラフト 約 1.9 秒、384² 約 1.5 秒、256² 約 1.4 秒(固定分 約 1.2 秒はプロンプトのテキストエンコード)、1024² の最終 約 6.6 秒。
studio の高速プロファイル(テキストエンコーダ常駐 + fp8r + torch.compile)では、512² 8 step のドラフト 0.5 秒(壁時計 0.65 秒)、1024² 16 step の最終 3.9 秒、768×1024 参照 768px 12 step 1.8 秒。停止直後のジョブもモデルを読み直さなくなりました。
ステップ数を減らした効果(同条件・同一プロセスでの比較、512²、参照 512): 16 step 3.9 秒 → 12 step 3.3 秒 → 8 step 2.8 秒 → 4 step 2.1 秒。固定分(テキストエンコードと VAE)が大きいので、半分にしても時間は 3 割減です。

`steps` は flux3-klein-studio の `POST /api/generate` に 2026-10-27 に追加した引数です。古い studio に対しては `/openapi.json` で未対応を検出し、ドラフトも 16 step で生成します(画面のドラフト欄に注意書きが出ます)。
生成は flux3-klein-studio 側で同時 1 件なので、Studio の画面を同時に使うと順番待ちになります(409 は自動で再試行)。

## プロンプト

`viewprompt.py` の `build_prompt(yaw, pitch, roll, subject, framing, extra, roll_in_prompt)` が、次の順で 1 つのキャプションを組み立てます。画面下に、実際に送る英文がそのまま出ます。

1. 保つもの: 「同じ人物をちょうど 1 人、顔・髪・服・小物・体型・色・画風をそのまま」
2. Y: 「自分の左へ N 度向いた」+「鼻がどちらの画像端を向くか」+「見えている体の側」+ 三分の一・横顔・後ろ 3/4 など角度帯ごとの体の形(後ろ向きでは「前側だけの柄は見えない」、キャラクターの左側がどちらに来るか)
3. X: 「上から N 度の俯瞰」「床が背景」/ 「下から N 度、カメラは床近く、足が一番近くて大きい、天井・空が背景」
4. Z: ダッチアングル(N 度時計回り、頭がどちらへ傾く)
5. 構図: 元画像と同じ / 全身 / クローズアップ
6. 背景と光は元画像のまま、物や文字を足さない + 追加の指示

効き方の目安(人物の全身写真で確認):

- Y(向き)は 0〜±180° まで良く効きます。10〜30° は実際より浅めに出ます。
- X は +20〜+90°(俯瞰)と −20〜−60°(あおり)が効きます。−90°(真下)は正面に戻りがちです。
- Z はプロンプトでは弱く、±30° 程度まで。画像面内の回転なので「Z軸の扱い: キャンバス回転」にすると再生成せず正確に回ります(既定はプロンプト)。
- 「被写体: 物体」を選ぶと、顔や服の代わりに前面・側面・背面の見え方で書きます。

## API(このアプリ)

| メソッド | パス | 内容 |
|---|---|---|
| `POST` | `/api/upload` | `image` をアップロード。`id`、サイズ、ドラフト・最終の出力サイズを返す |
| `GET` | `/api/prompt?yaw=&pitch=&roll=&subject=&framing=&extra=&roll_in_prompt=` | 角度からキャプションを作って返す(生成しない) |
| `POST` | `/api/render` | JSON `{session, yaw, pitch, roll, mode: draft|final, seed, subject, framing, extra, roll_in_prompt, draft_side, draft_steps, final_size, upsample, upsampler, shift}`。キャッシュがあれば `status: done` を即返す |
| `GET` | `/api/render/{id}` | 状態(`queued` / `generating` / `saving` / `done` / `error` / `cancelled`、`step` / `steps`、`image_url`、`seconds`) |
| `POST` | `/api/render/{id}/cancel` | 停止 |
| `GET` | `/api/sessions/{id}/views` | 生成済みの一覧 |
| `GET` | `/api/health` | 生成サーバーの状態 |

画像は `data/{session}/views/{key}.png`(同名の `.json` にプロンプトと角度)に保存され、自動削除しません。

## テスト

```bash
../flux3-klein-studio/.venv/bin/python -m pytest -q tests
```

## ライセンス

このアプリのコードは [MIT License](LICENSE)(Copyright 2026 YUZUKI)です。
画像生成に使う FLUX 3 Image Klein のモデルは FLUX Kommunity License v1.0 で、flux3-klein-studio 側の
[第三者ライセンスと注意事項](../flux3-klein-studio/THIRD_PARTY_NOTICES.md)に従います。このリポジトリにモデルの重みは含みません。
