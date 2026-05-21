# 精密Lab. 受付・案内ロボット

東大五月祭「精密Lab.」展示向けの受付・案内ロボットシステム。

![デモ](demo.png)

館内フロアマップをリアルタイム表示し、Firebase から取得した整理券待ち状況をオーバーレイ表示する。
ブランチによっては音声合成・音声認識・Gemini API を使った来場者との対話機能も搭載。

---

## ブランチ一覧（引き継ぎ資料）

ブランチは機能追加の積み重ねで成長している。以下の順で派生。

```
main                      … 対話モード付きフル機能版
 ├─ with-other-exhibits-info … 他団体企画・五月祭全体マップを追加
 └─ face-detection        … 全面マップ表示・顔検知自動切替・Arduino 連携
      └─ mayfes           … 五月祭2026 当日実際に使用したブランチ（最終形）
```

---

### `main` — 対話モード付きフル機能版

来場者と音声で対話できるフル機能ブランチ。次回以降の出展でフル機能に戻す場合はここが起点になる。

**主な機能**

- **待機モード**: gTTS で合成した呼び込みメッセージを一定間隔で発話。動画をループ再生
- **対話モード**: スペースキーで切替。マイク入力 → Google STT → Gemini API → gTTS 読み上げ
- **企画ハイライト**: Gemini が返した企画名に対応する座標に赤丸＋波紋アニメーションを表示
- **Firebase 連携**: 整理券待ち人数を30秒ごとに取得し、プロンプトに注入して回答に反映
- **バージイン**: 発話中にマイクが大きな音を検知すると再生を中断して聞き直す

**起動**

```bash
python main.py
```

**依存**: `GEMINI_API_KEY`、`firebase_admin.json`

---

### `with-other-exhibits-info` — 他団体企画・五月祭全体マップ追加版

main の機能に加えて、精密Lab. 以外の他団体企画の場所も Gemini が案内できるようにしたブランチ。

**追加機能**

- `assets/festival_map.png` に五月祭全体キャンパスマップを追加。他団体の建物を聞かれると右パネルに全体マップを表示し、該当建物をハイライト
- `stitch_map.py`: 複数マップを結合するユーティリティ
- Gemini コンテキストキャッシュを使用（API コスト削減・起動時に TTL=12h でキャッシュ登録）
- キーボード入力でも質問を送れるテキスト入力欄を追加

**起動**

```bash
python main.py
```

**依存**: `GEMINI_API_KEY`、`firebase_admin.json`

---

### `face-detection` — 全面マップ・顔検知自動切替・Arduino 連携

main の対話機能にハードウェア連携を追加し、レイアウトも全面マップに変更した最多機能ブランチ。

**追加機能**

- **全面マップ表示**: 画面全体をフロアマップに使うレイアウト。動画パネルなし。左右にサイドラベルとロゴをオーバーレイ
- **顔検知自動切替**: OpenCV で来場者の顔を検知し、顔あり → 対話モード・顔なし → 待機モードに自動遷移
  - `face_tune.json` で検知感度パラメータを調整
- **Arduino サーボ制御**: Python が発話状態を USB シリアルで Arduino に5秒ごとに送信し、サーボモーターを動かす（詳細は後述）
- **外部マイク指定**: `--mic` でデバイスインデックスを指定可能

**起動**

```bash
python main.py                                    # 基本
python main.py --serial /dev/cu.usbmodem1101      # Arduino あり
python main.py --serial /dev/cu.usbmodem1101 --mic 2  # 外部マイクも指定
```

**依存**: `GEMINI_API_KEY`、`firebase_admin.json`、Arduino（任意）

#### Arduino スケッチ（`arduino_servo.ino`）

サーボ4本（ピン 3, 5, 10, 11）を制御。シリアルで `0\n` / `1\n` を受信するとモードを切り替える。

| モード | 条件 | 動作 |
|--------|------|------|
| 0 | 発話中 | servo3・servo5 が1秒ごとにランダム動作 |
| 1 | 非発話 | servo10・servo11 が2秒ごとにランダム動作 |

ボーレート: 115200

---

### `mayfes` — 五月祭2026 当日使用版（最終形）

**五月祭2026当日に実際に動かしたブランチ。** face-detection をベースに、会場運用に合わせて機能を絞り込んだ。

**main/face-detection からの変更点**

- 対話モード・音声認識・カメラ入力を完全削除（しゃべらないロボット）
- 動画ループ再生を削除
- 呼び込みメッセージを gTTS で事前合成して一定間隔（10〜15秒）で発話
- 画面下部に字幕テキストを表示（発話中のメッセージ内容）
- Arduino シリアル通信は維持（発話状態を送信してサーボを動かす）

**起動**

```bash
python main.py
python main.py --serial /dev/cu.usbmodem1101  # Arduino あり
```

**依存**: `firebase_admin.json`（Gemini API 不要）

---

## 音声合成について（gTTS を採用した理由）

全ブランチとも音声合成は **gTTS（Google Text-to-Speech）** を使用している。

当初は VOICEVOX（ローカル実行の高品質 TTS）を採用していたが、以下の問題から gTTS に切り替えた。

- **合成ラグが顕著**: VOICEVOX は音声合成時に Python GIL を長時間保持するため、合成中は pygame の描画・音声再生が数秒間フリーズする
- **環境構築コスト**: モデルのダウンロードに数 GB が必要で、起動時の初期化にも数十秒かかる
- gTTS は合成が速く（1〜2秒）、インターネット接続さえあれば追加ダウンロード不要

gTTS の音声パラメータ（速度・ピッチ・サーバリージョン）は `GttsTTS` クラスで調整できる。対話モード中はキーボードショートカットでリアルタイムに変更可能（`↑↓` ピッチ、`←→` 速度）。

---

## システム構成

### main / face-detection ブランチ

```
Firebase Firestore（整理券待ち人数）← 30秒ごとにポーリング
    ↓
マイク入力（待機モード: 呼び込み発話 / 対話モード: 来場者の音声）
    ↓ [対話モード時]
Google STT（音声認識） → Gemini API（回答・企画名を JSON で生成）
    ↓
gTTS（音声合成）→ ffmpeg（WAV変換）→ pygame（スピーカー出力）
    ↓ [face-detection のみ]
Arduino シリアル送信 → サーボモーター動作
```

### mayfes ブランチ

```
Firebase Firestore（整理券待ち人数）← 30秒ごとにポーリング
    ↓
gTTS で事前合成した呼び込みメッセージを一定間隔で再生
    ↓
pygame（全面マップ・字幕・混雑状況を表示）
    ↓
Arduino シリアル送信 → サーボモーター動作
```

---

## 画面レイアウト

### main / with-other-exhibits-info

```
┌──────────────────────────────────────────────────┐
│  館内マップ（左半分）  │  動画 or 企画写真（右半分）│
│  ※混雑状況吹き出し付  │  ※待機中:動画 / 対話中:写真│
├──────────────────────────────────────────────────┤
│  音量バー（マイク感度確認用）                     │ ← 28px
└──────────────────────────────────────────────────┘
```

### face-detection / mayfes

```
┌──────────────────────────────────────────────────┐
│←都市工学│                            │→精密ラボ  │
│  科企画  │      館内マップ（全画面）   │（精密工学 │
│          │      ※混雑状況吹き出し付   │  科企画）  │
│          │                            │           │
├──────────────────────────────────────────────────┤
│  字幕テキスト（呼び込みメッセージ）               │ ← 下部
└──────────────────────────────────────────────────┘
```

---

## ファイル構成

```
entrance_robot/
├── main.py                    # メインロジック
├── README.md                  # 本ファイル（引き継ぎ資料）
├── arduino_servo.ino          # Arduino サーボ制御スケッチ
├── requirements.txt           # 依存ライブラリ
├── .env                       # APIキー（Git管理外）
├── firebase_admin.json        # Firebase 秘密鍵（Git管理外）
├── face_tune.json             # 顔検知パラメータ（face-detection ブランチ）
├── preview_map.py             # 企画座標の確認・調整ツール
├── stitch_map.py              # マップ結合ツール（with-other-exhibits-info）
├── prompts/
│   └── system_prompt.txt      # Gemini へのシステムプロンプト
├── assets/
│   ├── logo.png               # 精密Lab. ロゴ
│   ├── festival_map.png       # 五月祭全体マップ（with-other-exhibits-info）
│   └── photos/                # 企画ポスター（PDF形式）
│       └── {キー名}.pdf
├── 案内図/
│   └── facility_map.png       # 館内フロアマップ（PNG）
├── movies/                    # 待機中ループ動画（main / face-detection）
│   └── *.mp4 / *.mov 等
└── audio/                     # 音声ファイル格納先（予備）
```

---

## セットアップ

### 1. 依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 2. ffmpeg のインストール

gTTS が生成する MP3 を WAV に変換するために必要。

```bash
brew install ffmpeg   # macOS
```

### 3. 環境変数の設定（main / face-detection / with-other-exhibits-info）

`.env` ファイルをプロジェクトルートに作成:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

> `mayfes` ブランチは Gemini を使わないため不要。

### 4. Firebase サービスアカウントキーの設置（全ブランチ共通）

混雑状況のリアルタイム取得に Firebase Admin SDK を使用している。

1. [Firebase Console](https://console.firebase.google.com/) → プロジェクト設定 → サービスアカウント
2. 「新しい秘密鍵の生成」で JSON ファイルをダウンロード
3. プロジェクトルートに `firebase_admin.json` として配置（`.gitignore` 済み）

### 5. 起動

```bash
# 基本（全ブランチ共通）
python main.py

# Arduino あり（face-detection / mayfes）
python main.py --serial /dev/cu.usbmodem1101

# 外部マイク指定（face-detection）
python main.py --serial /dev/cu.usbmodem1101 --mic 2
```

---

## コンテンツの追加・更新

### 企画ポスターの追加

```
assets/photos/{キー名}.pdf
```

キー名は `main.py` の `EXHIBIT_KEY_MAP` の左辺と一致させること（例: `truck`, `space`, `media`）。

### 企画座標の調整

`preview_map.py` で座標を視覚確認しながら調整できる:

```bash
python preview_map.py
```

調整後は `main.py` の `EXHIBIT_LOCATIONS` を更新:

```python
"ジャングル・スコープ": {"x": 0.393, "y": 0.086, "bx": 0.381, "by": 0.016},
#                        ↑マップ上の赤丸座標       ↑吹き出し矢印の位置
```

### 呼び込みメッセージの変更

`main.py` の `STANDBY_MESSAGES` を編集（変更後は再起動で再合成される）:

```python
STANDBY_MESSAGES = [
    "精密ラボへようこそ！...",
]
```

### 新しい企画を追加するには

1. `EXHIBIT_KEY_MAP` にキーと企画名を追加
2. `EXHIBIT_LOCATIONS` に座標を追加（`preview_map.py` で確認）
3. `assets/photos/{キー名}.pdf` に企画ポスターを追加
4. `prompts/system_prompt.txt` に企画の説明を追記（main / face-detection 系のみ）

---

## パラメータ調整

`main.py` 冒頭の定数で動作を調整できる。

**全ブランチ共通**

| 定数                     | 初期値   | 説明                              |
| ------------------------ | -------- | --------------------------------- |
| `FIREBASE_POLL_INTERVAL` | 30       | Firebase ポーリング間隔（秒）     |
| `DISPLAY_WIDTH`          | 1200     | ウィンドウ幅（px）                |
| `DISPLAY_HEIGHT`         | 700      | ウィンドウ高さ（px）              |

**mayfes / face-detection のみ**

| 定数                     | 初期値   | 説明                              |
| ------------------------ | -------- | --------------------------------- |
| `STANDBY_INTERVAL_RANGE` | (10, 15) | 呼び込み発話の間隔（秒）の最小・最大 |

**main / face-detection（対話機能あり）のみ**

| 定数               | 初期値 | 説明                                         |
| ------------------ | ------ | -------------------------------------------- |
| `MIC_THRESHOLD`    | 300    | マイク感度（高いほど鈍感）                   |
| `LISTEN_TIMEOUT`   | 8      | 無音タイムアウト（秒）。超えると待機モードへ |
| `BARGE_IN_THRESHOLD` | 10000 | バージイン検知の音量閾値                    |

---

## スレッド構成（main / face-detection ブランチ）

```
メインスレッド          : pygame イベント処理・描画（30fps）
├─ 待機/対話スレッド    : モードロジック・音声合成・認識
├─ 動画再生スレッド     : OpenCV フレームデコード → キュー（main のみ）
├─ 音量モニタースレッド : PyAudio でリアルタイム RMS 計測
├─ Firebase ポーラー   : 30秒ごとに混雑状況を取得
└─ Arduino 送信スレッド : 5秒ごとに発話状態をシリアル送信（face-detection のみ）
```

---

## 注意事項

- Gemini API は無料枠に制限あり。使い切ると 429/503 エラーが発生するが、自動でエラーメッセージを発話してリトライ待機する
- gTTS・Google STT・Gemini API はすべてインターネット接続が必要
- `debug.log` にログが出力される（起動ごとに上書き）
- `firebase_admin.json` と `.env` は Git 管理外。次回利用時は各自で再取得・再配置が必要
- フォントは macOS のヒラギノ角ゴシックを参照している。他 OS では `_load_jp_font()` のパスを変更する必要がある
