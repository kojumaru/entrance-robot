## これからやること

こうじ

- 混雑状況からおすすめ企画に誘導
- 五月祭パンフレットを読み込ませて、精密ラボ以外のことを聞かれた時にも答えられるようにする

太田

- ウェブカメラからの画像認識で話しかけられているモードを検知し、`robot._switch_to(RobotState.INTERACTION)` を呼んで対話モードへ自動切替。STANDBYモードへの戻し方も実装。
  詳細は上記「モード切替の仕組み（外部連携向け）」セクションを参照
- 目の実装
- 口の実装
- 手の実装　←優先度低め？

---

# 精密Lab. 受付・案内ロボット

東大五月祭「精密Lab.」展示向けの受付・案内ロボットシステム。
来場者の音声による質問に答え、展示企画の場所をマップと写真・動画で案内する。

---

## システム Abstruct

```
Firebase Firestore（整理券待ち人数）← 30秒ごとにポーリング
    ↓
マイク入力
    ↓
Google STT（音声認識）
    ↓
Gemini API（回答・企画名を JSON で生成）← 混雑状況をプロンプトに注入
    ↓
gTTS（音声合成） → スピーカー出力
        ↓
    企画名が含まれる場合
        ↓
マップ上に赤丸アニメーション表示 ＋ 企画写真
```

---

## 動作モード

### 待機モード (STANDBY)

- 起動直後から自動で開始
- `movies/` ディレクトリの動画をループ再生（右パネル）
- 5〜7秒おきに、起動時に事前合成した呼び込みメッセージを発話
- 左パネルには常に館内マップを表示

### 対話モード (INTERACTION)

- スペースキーで待機モードから切り替え（仮）
- 「こんにちは！何かご質問はありますか？」と発話してマイク待機
- 来場者の発話を認識 → Gemini に送信 → 回答を音声で読み上げ
- 企画の場所を聞かれた場合:
  - マップの該当箇所に**脈動＋波紋アニメーション**で赤丸を表示
  - 右パネルに企画写真を表示
- 8秒間無音が続くと自動的に待機モードへ復帰

### モード切替・操作キー

| キー                     | 動作                                         |
| ------------------------ | -------------------------------------------- |
| スペースキー             | 待機 ↔ 対話 の切り替え                       |
| ↑ / ↓                   | 音声ピッチ +0.05 / -0.05（gTTS のみ）        |
| → / ←                   | 読み上げ速度 +0.05 / -0.05（gTTS のみ）      |
| T                        | 音声の tld を切り替え（com→co.jp→co.uk→…）  |
| ESC / ウィンドウを閉じる | 終了                                         |

ピッチ・速度・tld を変更すると全事前合成音声がバックグラウンドで自動再合成される。現在の値はコンソールに出力される。

---

## モード切替の仕組み（外部連携向け）

> **注**: 現在はスペースキーで手動切替しているが、将来的にウェブカメラの画像認識による自動切替に置き換える予定。

### 現在の実装（スペースキー）

`main.py` の `run()` メソッド内でキーイベントを監視し、スペースキーが押されると `toggle()` を呼ぶ。

```python
# main.py の run() 内
elif event.key == pygame.K_SPACE:
    self.toggle()
```

### `toggle()` の動作

```python
def toggle(self) -> None:
    next_state = (
        RobotState.INTERACTION if self.state == RobotState.STANDBY
        else RobotState.STANDBY
    )
    self._switch_to(next_state)
```

現在の状態を反転して `_switch_to()` を呼ぶだけ。

### `_switch_to()` の内部動作

```
① _stop_event.set()          … 今動いているループに「止まれ」を通知
② join(timeout=2)            … ループが終わるまで最大2秒待機
③ _stop_event.clear()        … フラグをリセット
④ state = new_state          … 状態を更新
⑤ 新しいスレッドを起動        … 対応するループ関数をバックグラウンドで開始
```

### 画像認識に向けて: 連携方法

カメラ側のスクリプトから `EntranceRobot` インスタンスの **`toggle()`** または **`_switch_to()`** を呼ぶだけで切り替えられる。

```python
# 例: 人物を検知したら対話モードへ、いなくなったら待機モードへ
if person_detected and robot.state == RobotState.STANDBY:
    robot._switch_to(RobotState.INTERACTION)

if not person_detected and robot.state == RobotState.INTERACTION:
    robot._switch_to(RobotState.STANDBY)
```

**注意点**:

- `_switch_to()` はどのスレッドから呼んでも安全（スレッドセーフ）
- 対話モード中は `listen()` のブロッキングにより、切替反応が最大 **8秒** 遅れる場合がある
- 対話モード → 待機モードへの強制遷移は、音声認識の途中でも止まる（`_stop_event` で制御）

---

## 画面レイアウト

```
┌──────────────────────────────────────────────────┐
│                    │                             │
│   館内マップ（左半分）  │  動画 or 企画写真（右半分）  │
│   ※常に固定          │  ※待機中:動画 / 対話中:写真  │
│                    │                             │
├──────────────────────────────────────────────────┤
│  音量バー  RMS: 312  mic: 800（白線）              │ ← 28px
└──────────────────────────────────────────────────┘
```

- ウィンドウはリサイズ可能（アスペクト比を維持してスケーリング）
- 画面下部の音量バーでマイク感度の調整が可能

---

## 技術構成

### 音声認識

- **ライブラリ**: SpeechRecognition + PyAudio
- **エンジン**: Google STT（無料・APIキー不要）
- **言語**: 日本語 (ja-JP)
- `MIC_THRESHOLD = 800` でマイク感度を調整（大きいほど鈍感）
- 8秒間無音でタイムアウト → 待機モードへ

### AI 応答生成

- **API**: Google Gemini API (`google-genai`)
- **モデル**: `gemini-2.5-flash`
- **出力形式**: JSON固定

```json
{
  "speech": "読み上げる回答文",
  "exhibit": "企画名 or null"
}
```

- `exhibit` が返ってきた場合のみマップにハイライト表示

### 音声合成

- **デフォルト**: gTTS（Google翻訳TTS）― 無料・要インターネット
- **オプション**: VOICEVOX Core（ローカル実行・オフライン）― `--tts voicevox` で起動

#### gTTS パラメータ（実行中にキーで調整可能）

| パラメータ | 初期値 | 説明 |
| ---------- | ------ | ---- |
| `speed`    | 1.15   | 読み上げ速度（ffmpeg atempo） |
| `pitch`    | 1.5    | ピッチ倍率（ffmpeg asetrate） |
| `tld`      | com    | Googleドメイン（声質に影響） |

- 発音修正は `WORD_FIXES` 辞書で単語単位に置換して対応
- 呼び込みメッセージは**起動時に事前合成**してWAVキャッシュ → 動画再生を止めない
- パラメータ変更時はバックグラウンドで全事前合成を再実行

### マップ表示

- `assets/企画場所.pdf` を PyMuPDF (dpi=150) で画像に変換
- 企画座標は `EXHIBIT_LOCATIONS` 辞書に相対座標 (0.0〜1.0) で手動登録
- ハイライトは pygame で毎フレーム描画（アニメーション）
  - 赤丸：sin 波で半径が脈動
  - 波紋：外側に広がって透明になる半透明の輪

### 動画再生

- OpenCV (`cv2`) でフレームをデコード → numpy 配列としてキューに積む
- メインスレッドがキューから取り出して pygame サーフェスに変換（スレッド安全）
- 世代番号管理により待機モード復帰時に古い動画スレッドを確実に停止

---

## ファイル構成

```
entrance_robot/
├── main.py                    # メインロジック
├── README.md                  # 本ファイル
├── spec.txt                   # 仕様書
├── requirements.txt           # 依存ライブラリ
├── .env                       # APIキー（Git管理外）
├── .gitignore
├── prompts/
│   └── system_prompt.txt      # Gemini へのシステムプロンプト
├── assets/
│   ├── 企画場所.pdf            # 館内マップ（1ページ）
│   └── photos/                # 企画写真
│       └── {企画名}.jpg        # ファイル名 = 企画名と完全一致
├── movies/                    # 待機中に再生する動画
│   └── *.mp4 / *.mov 等
└── voicevox_core/             # VOICEVOXを使う場合のみ必要
    ├── onnxruntime/
    ├── dict/
    └── models/
```

---

## セットアップ

### 1. システム依存ライブラリのインストール（macOS）

```bash
brew install portaudio ffmpeg
```

### 2. Python依存ライブラリのインストール

```bash
pip install -r requirements.txt
```

### 3. 環境変数の設定

`.env` ファイルを各自作成し、 Gemini API キーを記載:

```
GEMINI_API_KEY=your_api_key_here
```

### 4. Firebase サービスアカウントキーの設置

混雑状況のリアルタイム取得に Firebase Admin SDK を使用しています。

1. [Firebase Console](https://console.firebase.google.com/) → プロジェクト設定 → サービスアカウント
2. 「新しい秘密鍵の生成」でJSONファイルをダウンロード
3. ダウンロードしたファイルをプロジェクトルートに `firebase_admin.json` という名前で配置

```
entrance_robot/
└── firebase_admin.json   ← ここに置く（.gitignore済み・Gitには含まれない）
```

> このファイルはGit管理外のため、各自で取得・配置が必要です。

### 5. 起動

```bash
python main.py              # gTTS（デフォルト）
python main.py --tts voicevox  # VOICEVOX（要別途セットアップ）
```

### VOICEVOX を使う場合の追加セットアップ

```bash
pip install "https://github.com/VOICEVOX/voicevox_core/releases/download/0.16.4/voicevox_core-0.16.4-cp310-abi3-macosx_11_0_arm64.whl"
curl -sSfL https://github.com/VOICEVOX/voicevox_core/releases/latest/download/download-osx-arm64 -o download
chmod +x download
./download --exclude c-api
```

---

## コンテンツの追加・更新

### 企画写真の追加

```
assets/photos/{企画名}.jpg
```

企画名は `EXHIBIT_LOCATIONS` のキーと完全一致させること。

### 待機動画の追加

```
movies/{任意のファイル名}.mp4
```

対応形式: mp4, mov, avi, mkv。複数ある場合はファイル名の昇順でループ再生。

### 企画座標の調整

`main.py` の `EXHIBIT_LOCATIONS` を編集:

```python
"企画名": {"x": 0.52, "y": 0.14},  # マップ全体を1.0とした相対座標
```

### 呼び込みメッセージの変更

`main.py` の `STANDBY_MESSAGES` を編集（変更後は再起動で再合成）:

```python
STANDBY_MESSAGES = [
    "精密ラボへようこそ！...",
    "こんにちは！...",
]
```

### 発音修正の追加

`main.py` の `GttsTTS.WORD_FIXES` を編集:

```python
WORD_FIXES: dict[str, str] = {
    "工学": "こう学",
    "単語": "読み方",
}
```

---

## パラメータ調整

`main.py` 冒頭の定数で動作を調整できる。

| 定数               | 初期値 | 説明                                                             |
| ------------------ | ------ | ---------------------------------------------------------------- |
| `MIC_THRESHOLD`    | 800    | マイク感度（高いほど鈍感）。画面下部の音量バーで確認しながら調整 |
| `LISTEN_TIMEOUT`   | 8      | 無音タイムアウト（秒）。この時間無音なら待機モードへ             |
| `VOICEVOX_SPEAKER` | 3      | 話者ID（1: ずんだもん、3: ずんだもんあまあま）※voicevox使用時のみ |

---

## スレッド構成

```
メインスレッド          : pygame イベント処理・描画（30fps）
├─ 待機/対話スレッド    : モードロジック・音声合成・認識
├─ 動画再生スレッド     : OpenCV フレームデコード → キュー
├─ 音量モニタースレッド : PyAudio でリアルタイム RMS 計測
└─ 再合成スレッド       : TTS パラメータ変更時に事前合成を再実行
```

---

## 注意事項

- VOICEVOX の音声を使用する際は「VOICEVOX:ずんだもん」等のクレジット表記が必要
- Gemini API は無料枠に制限あり。使い切ると 429/503 エラーが発生するが、自動でエラーメッセージを発話してリトライ待機する
- インターネット接続が必要（Google STT・Gemini API・gTTS）
- 音声認識は Google STT を使用（APIキー不要・無料）
