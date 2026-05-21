"""
精密ラボ. 受付・案内ロボット
  - Space キーでモード切替（YOLO 実装後は toggle() を直接呼ぶ）
  - 待機モード : 10〜15 秒ごとにランダム呼び込み + movies/ の動画をループ再生
  - 対話モード : Google STT 音声認識 → Gemini → VOICEVOX 読み上げ + マップ表示
"""

import concurrent.futures
import json
import os
import queue
import random

from dotenv import load_dotenv
load_dotenv()

import collections
import cv2
import math
import mediapipe as mp
import multiprocessing
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path

import argparse
import subprocess
import serial as _serial

import fitz  # PyMuPDF
import openai
import PIL.Image as PILImage
import PIL.ImageDraw as PILImageDraw
import PIL.ImageFont as PILImageFont
import pygame
import speech_recognition as sr

# ── ログファイル出力（stdout/stderr を両方ファイルにも書き出す）────────────────
import sys as _sys

class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            try: s.write(data)
            except Exception: pass
    def flush(self):
        for s in self._streams:
            try: s.flush()
            except Exception: pass
    def isatty(self): return False

_log_path = Path(__file__).parent / "debug.log"
_log_file = open(_log_path, "w", encoding="utf-8", buffering=1)
_sys.stdout = _Tee(_sys.__stdout__, _log_file)
_sys.stderr = _Tee(_sys.__stderr__, _log_file)
print(f"=== ログ出力先: {_log_path} ===")

# ── 定数 ──────────────────────────────────────────────────────────────────────

MIC_THRESHOLD = 300
LISTEN_TIMEOUT = 8

BARGE_IN_THRESHOLD = 10000  # バージイン検知の閾値
BARGE_IN_DELAY = 0.5        # 再生開始後、この秒数は監視しない（エコー対策）
_FIREBASE_CREDENTIAL = Path(__file__).parent / "firebase_admin.json"

# 企画key → 企画名マッピング（Geminiの出力・Firebase両方で使用）
EXHIBIT_KEY_MAP = {
    # 3階
    "truck":    "ジャングル・スコープ",
    "space":    "現実拡張空間",
    "media":    "精密メディアアート",
    "balloon":  "バルーンロボット",
    "switch":   "せいみつスイッチ",
    "arm":      "ロボットアーム",
    "soccer":   "スーパーロボットサッカー",
    # 1階
    "ai_lab":   "AI精密ラボ",
    "connect4": "立体四目並べ",
    "pong":     "せいみつPONG!",
    "shooting": "お絵描きシューティング",
    "tank":     "ARタンク",
    "pendulum": "スーパー倒立振子",
    "dress":    "AI着せ替えカメラ",
    "janken":   "じゃんけんAI",
}

# 企画名 → ポスターPDFファイル名マッピング
EXHIBIT_POSTER_MAP = {
    "ジャングル・スコープ":     "truck.pdf",
    "現実拡張空間":             "space.pdf",
    "精密メディアアート":       "media.pdf",
    "バルーンロボット":         "balloon.pdf",
    "せいみつスイッチ":         "switch.pdf",
    "ロボットアーム":           "arm.pdf",
    "スーパーロボットサッカー": "soccer.pdf",
    "AI精密ラボ":               "ai_lab.pdf",
    "立体四目並べ":             "connect4.pdf",
    "せいみつPONG!":            "pong.pdf",
    "お絵描きシューティング":   "shooting.pdf",
    "ARタンク":                 "tank.pdf",
    "スーパー倒立振子":         "pendulum.pdf",
    "AI着せ替えカメラ":         "dress.pdf",
    "じゃんけんAI":             "janken.pdf",
}

FIREBASE_POLL_INTERVAL = 30  # 混雑状況の更新間隔（秒）
FACE_TUNE_SAVE_PATH = Path(__file__).parent / "face_tune.json"  # チューニング値の保存先
QUEUE_BUFFER_MINUTES = 10   # 番号呼び出し後に列に並ぶまでの目安時間（分）

STANDBY_MESSAGES = [
    "精密ラボへようこそ！1階右と3階で工学のさまざまな企画を展示しています！",
    "こんにちは！精密ラボの展示をご案内します！",
]

FAREWELL_KEYWORDS = [
    "ありがとう", "どうも", "助かりました", "わかりました", "了解",
    "大丈夫です", "いいです", "結構です", "以上です", "バイバイ", "さようなら",
]
FAREWELL_RESPONSES = [
    "どういたしまして。ぜひ楽しんでくださいね！",
    "お役に立てて嬉しいです。ごゆっくりどうぞ！",
    "またいつでも声をかけてくださいね！",
]

THINKING_MESSAGES = [
    "うううん、",
    "そうですね、",
    "ええっと、"
]
THINKING_INTERVAL = 0.5  # セリフとセリフの間の無音（秒）

DISPLAY_WIDTH = 1200
DISPLAY_HEIGHT = 700

# ── 顔検知によるモード切替 ────────────────────────────────────────────────────
FACE_CAMERA_ID           = 1     # カメラデバイスID
FACE_TRIGGER_SECONDS     = 1.5   # 正面検知が継続したら対話モードに切替
FACE_MIN_AREA_RATIO      = 0.01  # 顔面積/画面面積の最小値（遠すぎる人を無視）
FACE_CENTER_MARGIN       = 0.40  # 顔中心X座標が画面中心から許容するずれ（0.5が端）
FACE_SYMMETRY_THRESH     = 0.08  # 鼻が目の中間からずれる許容量（顔幅比）
FACE_EYE_LEVEL_THRESH    = 0.10  # 左右目の高さ差の許容量（顔高さ比）
FACE_STABILITY_FRAMES    = 8     # 安定判定に使う過去フレーム数
FACE_STABILITY_MAX_MOVE  = 0.05  # 安定とみなす最大移動量（画面幅比）
FACE_LOOP_INTERVAL       = 0.10  # 顔検知ループの間隔（秒）≒10fps

# ── 企画場所の座標（マップ画像全体を1.0とした相対座標）────────────────────────
# x,y  : ハイライト丸の中心（マップ全体を1.0とした相対座標）
# bx,by : 吹き出し矢印の先端（省略時は x,y を使用）
EXHIBIT_LOCATIONS = {
    # 3階
    "ジャングル・スコープ":              {"x": 0.3930, "y": 0.0860, "bx": 0.3810, "by": 0.0160},
    "現実拡張空間":                  {"x": 0.5390, "y": 0.0980, "bx": 0.5270, "by": 0.0190},
    "精密メディアアート":               {"x": 0.9210, "y": 0.0950, "bx": 0.9180, "by": 0.0100},
    "バルーンロボット":                {"x": 0.3240, "y": 0.3350, "bx": 0.3000, "by": 0.2820},
    "せいみつスイッチ":                {"x": 0.5240, "y": 0.3560, "bx": 0.5210, "by": 0.2790},
    "ロボットアーム":                 {"x": 0.7080, "y": 0.3500, "bx": 0.6960, "by": 0.2730},
    "スーパーロボットサッカー":            {"x": 0.9100, "y": 0.3440, "bx": 0.8830, "by": 0.2760},
    # 1階
    "AI精密ラボ":                  {"x": 0.3100, "y": 0.6260, "bx": 0.2950, "by": 0.5790},
    "立体四目並べ":                  {"x": 0.4250, "y": 0.6290, "bx": 0.4280, "by": 0.5670},
    "せいみつPONG!":               {"x": 0.5580, "y": 0.6290, "bx": 0.5760, "by": 0.5760},
    "お絵描きシューティング":             {"x": 0.8560, "y": 0.6170, "bx": 0.8470, "by": 0.5550},
    "ARタンク":                   {"x": 0.5590, "y": 0.8500, "bx": 0.5440, "by": 0.7820},
    "スーパー倒立振子":                   {"x": 0.7020, "y": 0.8290, "bx": 0.6900, "by": 0.7700},
    "AI着せ替えカメラ":               {"x": 0.8570, "y": 0.8380, "bx": 0.8780, "by": 0.7850},
    "じゃんけんAI":                 {"x": 0.1360, "y": 0.8530, "bx": 0.1120, "by": 0.7820},
}



# ── TTS エンジン ──────────────────────────────────────────────────────────────

class TTSEngine:
    def synthesize(self, text: str) -> str:
        """テキストを合成してWAVファイルパスを返す"""
        raise NotImplementedError
    def close(self): pass


class GttsTTS(TTSEngine):
    """gTTS (Google翻訳TTS) — 無料・要インターネット"""
    TLD_LIST = ["com", "co.jp", "co.uk", "com.au", "ca"]

    # 発音調整用テキスト置換辞書
    WORD_FIXES: dict[str, str] = {
        "工学": "こう学",
    }

    def __init__(self):
        self.speed = 1.1
        self.pitch = 1.5
        self._tld_index = 0

    @property
    def tld(self) -> str:
        return self.TLD_LIST[self._tld_index]

    def next_tld(self) -> str:
        self._tld_index = (self._tld_index + 1) % len(self.TLD_LIST)
        return self.tld

    def synthesize(self, text: str) -> str:
        from gtts import gTTS
        for word, reading in self.WORD_FIXES.items():
            text = text.replace(word, reading)
        mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        mp3.close()
        gTTS(text, lang="ja", tld=self.tld).save(mp3.name)
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()
        # atempo: 速度調整 / asetrate+aresample: ピッチ調整（速度に影響しない）
        af = f"atempo={self.speed},asetrate=24000*{self.pitch},aresample=24000"
        subprocess.run(
            ["ffmpeg", "-y", "-i", mp3.name,
             "-filter:a", af,
             "-ac", "1", "-sample_fmt", "s16",
             wav.name],
            check=True, capture_output=True
        )
        os.unlink(mp3.name)
        return wav.name


def create_tts_engine() -> TTSEngine:
    return GttsTTS()


# ── 顔検知チューニングパラメータ定義 ─────────────────────────────────────────
# (FaceDetector の属性名, 表示名, 変化量, 最小値, 最大値)
MIC_PARAMS = [
    # (EntranceRobotの属性名, 表示名, 変化量, 最小値, 最大値, int?)
    ("_mic_threshold",   "音声認識閾値        ［背景ノイズより少し高い値に。低すぎると常時反応、高すぎると無視される］", 10, 50, 2000, True),
    ("_listen_timeout",  "音声待機タイムアウト ［この秒数内に声が入らないと対話モードを終了して待機に戻る］",             1,  3,  30,   True),
]

FACE_PARAMS = [
    ("trigger_seconds",    "起動秒数         ［正面を向き続けると対話モードへ移行するまでの秒数］", 0.1,  0.3, 5.0),
    ("min_area_ratio",     "最小顔面積比      ［顔が画面面積の何割以上なら有効か（小さいと遠い人を無視）］", 0.01, 0.01, 0.30),
    ("center_margin",      "中央許容ずれ幅    ［顔の中心X座標が画面中心からどれだけずれていいか（0.5=端まで許容）］", 0.05, 0.05, 0.50),
    ("symmetry_thresh",    "正面対称閾値      ［鼻が左右目の中間からずれていい量（小さいほど真正面のみ検知）］", 0.01, 0.01, 0.30),
    ("eye_level_thresh",   "目の高さ差閾値    ［左右の目の高さ差の許容量（小さいほど傾き補正が厳しい）］", 0.01, 0.01, 0.30),
    ("stability_max_move", "安定移動量閾値    ［フレーム間の顔の移動量がこれ以下なら安定とみなす（画面幅比）］", 0.01, 0.01, 0.20),
    ("stability_frames",   "安定フレーム数    ［安定判定に使う過去フレーム数（多いほど動きへの感度が下がる）］", 1,    2,    20),
]

# ── 顔検知クラス ───────────────────────────────────────────────────────────────

class FaceDetector:
    """
    MediaPipe Face Detection を使って「話しかけようとしている人」を検知する。
    条件: 顔が正面を向いている + 十分近い + 画面中央付近 + 位置が安定
    各パラメータは実行時に動的変更可能（キーボードチューニング対応）
    """

    def __init__(self) -> None:
        self._mp_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=0,          # 0=2m以内の近距離モデル
            min_detection_confidence=0.6,
        )
        self._cap = cv2.VideoCapture(FACE_CAMERA_ID)
        self._history: collections.deque = collections.deque(maxlen=FACE_STABILITY_FRAMES)

        # チューニング可能パラメータ（実行時に変更可・保存/ロード対応）
        self.trigger_seconds    = FACE_TRIGGER_SECONDS
        self.min_area_ratio     = FACE_MIN_AREA_RATIO
        self.center_margin      = FACE_CENTER_MARGIN
        self.symmetry_thresh    = FACE_SYMMETRY_THRESH
        self.eye_level_thresh   = FACE_EYE_LEVEL_THRESH
        self.stability_max_move = FACE_STABILITY_MAX_MOVE
        self.stability_frames   = FACE_STABILITY_FRAMES

        # デバッグ表示用（メインスレッドから参照するためロック管理）
        self._debug_lock   = threading.Lock()
        self._debug_frame  = None   # RGB numpy array
        self._debug_bb     = None   # 最良候補のbounding box（検知条件通過したもの）
        self._debug_intent: bool = False
        self._debug_reason: str   = "未初期化"  # 最後の判定結果の理由
        self._debug_frame_ratio: float = 0.0   # フレーム蓄積率 0.0〜1.0
        self._load_tune()

    def _load_tune(self) -> None:
        """保存済みチューニング値をJSONから読み込む"""
        if not FACE_TUNE_SAVE_PATH.exists():
            return
        try:
            data = json.loads(FACE_TUNE_SAVE_PATH.read_text(encoding="utf-8"))
            for attr, *_ in FACE_PARAMS:
                if attr in data:
                    setattr(self, attr, data[attr])
            print(f"  [顔検知チューニング] 保存済み値をロード: {FACE_TUNE_SAVE_PATH}")
        except Exception as e:
            print(f"  [顔検知チューニング] ロード失敗: {e}")

    def save_tune(self) -> None:
        """現在のチューニング値をJSONに保存する"""
        data = {attr: getattr(self, attr) for attr, *_ in FACE_PARAMS}
        FACE_TUNE_SAVE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [顔検知チューニング] 保存しました: {FACE_TUNE_SAVE_PATH}")

    def is_intent_detected(self) -> bool:
        """話しかけようとしている人を検知したら True を返す"""
        ret, frame = self._cap.read()
        if not ret:
            return False

        # stability_frames が動的変更された場合にdequeサイズを追従
        if self._history.maxlen != self.stability_frames:
            old = list(self._history)
            self._history = collections.deque(old[-self.stability_frames:], maxlen=self.stability_frames)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._mp_detection.process(rgb)

        _bb_candidate = None
        _result = False
        _reason = "顔なし"
        try:
            if not results.detections:
                self._history.clear()
                return False

            # 候補の中で「面積×中央寄り度」が最高の顔を選ぶ
            best = None
            best_score = -1.0
            _reason = f"候補なし(顔{len(results.detections)}件・面積/中央条件を満たさず)"
            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                area = bb.width * bb.height
                cx = bb.xmin + bb.width / 2
                if area < self.min_area_ratio:
                    _reason = f"面積不足({area:.3f}<{self.min_area_ratio:.3f})"
                    continue
                if abs(cx - 0.5) > self.center_margin:
                    _reason = f"中央外れ(cx={cx:.2f},余裕={self.center_margin:.2f})"
                    continue
                score = area * (1.0 - abs(cx - 0.5))
                if score > best_score:
                    best_score = score
                    best = (det, bb, cx, bb.ymin + bb.height / 2)

            if best is None:
                self._history.clear()
                return False

            det, bb, cx, cy = best
            _bb_candidate = bb  # 条件通過した顔をデバッグ用に記録

            # キーポイントで正面向き判定
            # 順序: 0=右目, 1=左目, 2=鼻先, 3=口中央, 4=右耳, 5=左耳
            kps = det.location_data.relative_keypoints
            right_eye, left_eye, nose = kps[0], kps[1], kps[2]

            eye_mid_x = (right_eye.x + left_eye.x) / 2
            face_w = bb.width if bb.width > 1e-6 else 1e-6
            face_h = bb.height if bb.height > 1e-6 else 1e-6

            nose_offset    = abs(nose.x - eye_mid_x) / face_w
            eye_level_diff = abs(right_eye.y - left_eye.y) / face_h

            if nose_offset > self.symmetry_thresh:
                _reason = f"横向き(nose_offset={nose_offset:.3f}>{self.symmetry_thresh:.3f})"
                self._history.clear()
                return False
            if eye_level_diff > self.eye_level_thresh:
                _reason = f"目傾き(eye_diff={eye_level_diff:.3f}>{self.eye_level_thresh:.3f})"
                self._history.clear()
                return False

            # 位置の安定性チェック（歩いて通り過ぎる人を弾く）
            self._history.append((cx, cy))
            if len(self._history) < self.stability_frames:
                _reason = f"フレーム蓄積中({len(self._history)}/{self.stability_frames})"
                return False

            xs = [p[0] for p in self._history]
            ys = [p[1] for p in self._history]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            if x_range > self.stability_max_move:
                _reason = f"横移動不安定(dx={x_range:.3f}>{self.stability_max_move:.3f})"
                return False
            if y_range > self.stability_max_move:
                _reason = f"縦移動不安定(dy={y_range:.3f}>{self.stability_max_move:.3f})"
                return False

            _reason = "検知OK"
            _result = True
            return True
        finally:
            with self._debug_lock:
                self._debug_frame      = rgb
                self._debug_bb         = _bb_candidate
                self._debug_intent     = _result
                self._debug_reason     = _reason
                self._debug_frame_ratio = min(1.0, len(self._history) / max(self.stability_frames, 1))

    def get_debug_surface(self, width: int = 200) -> "pygame.Surface | None":
        """デバッグ用カメラ映像をpygame Surfaceで返す。検知状態をBBの色で示す。"""
        with self._debug_lock:
            if self._debug_frame is None:
                return None
            frame  = self._debug_frame.copy()
            bb     = self._debug_bb
            intent = self._debug_intent

        h, w = frame.shape[:2]
        if bb is not None:
            x1 = int(bb.xmin * w)
            y1 = int(bb.ymin * h)
            x2 = int((bb.xmin + bb.width) * w)
            y2 = int((bb.ymin + bb.height) * h)
            color = (0, 230, 80) if intent else (255, 180, 0)  # 緑=検知OK / 橙=条件不足
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        scale   = width / w
        new_h   = int(h * scale)
        small   = cv2.resize(frame, (width, new_h))
        return pygame.surfarray.make_surface(small.swapaxes(0, 1))

    def release(self) -> None:
        self._cap.release()
        self._mp_detection.close()


class RobotState(Enum):
    STANDBY = "standby"
    INTERACTION = "interaction"


# ── ロボット本体 ───────────────────────────────────────────────────────────────

class EntranceRobot:
    def __init__(self, serial_port: str | None = None, mic_index: int | None = None) -> None:
        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.state = RobotState.STANDBY
        self._stop_event = threading.Event()
        self._current_gen: int = 0  # _switch_toのたびに増加。旧スレッドが自身の失効を検知する

        # ── Arduino シリアル接続 ──────────────────────────────────
        self._arduino: _serial.Serial | None = None
        self._is_speaking: bool = False  # デフォルト: 非発話(1)
        if serial_port:
            try:
                self._arduino = _serial.Serial(serial_port, 115200, timeout=1)
                time.sleep(2)  # Arduino リセット待ち（接続時にArduinoがリセットするため必須）
                print(f"  [Arduino] 接続完了: {serial_port}")
                self._start_serial_sender()
                self._start_serial_reader()
            except Exception as e:
                print(f"  [Arduino] 接続失敗: {e}")
                self._arduino = None
        self._current_thread: threading.Thread | None = None

        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("精密ラボ. 案内マップ")
        self._show_cam_debug = True   # 対話モード中は非表示
        self._scan_progress: float | None = None  # None=非検知 / 0.0〜1.0=カウント中

        self.map_images = self._load_map_images()
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        logo_pil = PILImage.open(logo_path).convert("RGBA")
        logo_arr = pygame.image.fromstring(logo_pil.tobytes(), logo_pil.size, "RGBA").convert_alpha()
        self._logo_surf = logo_arr

        # 左右それぞれのサーフェスを管理
        self._surface_lock = threading.Lock()
        self._map_surface: pygame.Surface = self._pil_to_surface(self.map_images[0])
        self._right_surface: pygame.Surface | None = None
        self._highlight_loc: dict | None = None  # アニメーション用ハイライト座標

        self._tts: TTSEngine = create_tts_engine()
        self._standby_wavs: list[str] = self._presynth_standby_messages()
        self._greeting_wav: str = self._presynth_wav("こんにちは！精密ラボの企画についてなんでもお答えいたします！")
        self._thinking_wavs: list[str] = [self._presynth_wav(m) for m in THINKING_MESSAGES]
        self._thinking_channel = pygame.mixer.Channel(1)  # 専用チャンネル（GIL対策）
        self._thinking_loop_sound = self._make_thinking_loop_sound()  # 結合ループ音声
        self._farewell_wavs: list[str] = [self._presynth_wav(m) for m in FAREWELL_RESPONSES]

        self._resynth_lock = threading.Lock()  # 再合成の多重実行防止
        self._subtitle: str = ""               # 字幕テキスト（対話モード中の発話内容）

        self._mic_rms: float = 0.0
        self._is_listening: bool = False
        self._video_generation: int = 0
        self._video_frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._start_volume_monitor()
        self._bubble_font_main = self._load_jp_font(18, weight=6)
        noto_bold = "/Users/yoshidakouji/Library/Fonts/NotoSansCJKjp-Bold.otf"
        self._side_font       = pygame.font.Font(noto_bold, 42)
        self._side_font_mid   = pygame.font.Font(noto_bold, 32)
        self._side_font_small = pygame.font.Font(noto_bold, 30)
        self._listen_font = self._load_jp_font(32)
        self._label_font = self._load_jp_font(22)
        self._face_detect_font = self._load_jp_font(26, weight=7)
        self._subtitle_font = self._load_jp_font(30, weight=7)

        self._congestion: dict = {}  # 混雑状況キャッシュ
        self._start_firebase_poller()
        self._start_face_watcher()

        self._face_tuning_visible = False  # チューニングパネル表示フラグ
        self._mic_index      = mic_index        # マイクデバイスインデックス（None=デフォルト）
        self._mic_threshold  = MIC_THRESHOLD   # 音声認識エネルギー閾値
        self._listen_timeout = LISTEN_TIMEOUT  # 音声待機タイムアウト（秒）
        self._load_mic_tune()
        self._face_tuning_idx = 0          # 選択中のパラメータインデックス
        self._face_tune_font = self._load_jp_font(15)

    # ── ユーティリティ ────────────────────────────────────────

    def _ts(self, label: str) -> None:
        """タイムスタンプ付きでログ出力"""
        t = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        print(f"  [{t}] {label}")

    def _set_speaking(self, speaking: bool) -> None:
        """発話状態を設定し、即座にArduinoへ送信（バッファをflushして古いデータを破棄）"""
        self._is_speaking = speaking
        val = 0 if speaking else 1
        label = "発話中" if speaking else "非発話"
        if self._arduino and self._arduino.is_open:
            try:
                self._arduino.reset_input_buffer()   # Arduino受信バッファの古いデータを捨てる
                self._arduino.reset_output_buffer()  # 未送信データも捨てる
                self._arduino.write(f"{val}\n".encode())
                print(f"  [Arduino] 送信: {val} ({label})")
            except Exception as e:
                print(f"  [Arduino] 送信エラー: {e}")

    def _start_serial_sender(self) -> None:
        """5秒ごとに現在の状態をArduinoへ再送するハートビートスレッド"""
        def _heartbeat():
            while not self._stop_event.is_set():
                time.sleep(5)
                if self._arduino and self._arduino.is_open:
                    val = 0 if self._is_speaking else 1
                    try:
                        self._arduino.write(f"{val}\n".encode())
                    except Exception:
                        pass
        threading.Thread(target=_heartbeat, daemon=True).start()

    def _start_serial_reader(self) -> None:
        """Arduinoからの受信データを常時モニタしてターミナルに出力するスレッドを起動"""
        def _read():
            print("  [Arduino] 受信モニタ 開始")
            last_text = None
            while not self._stop_event.is_set():
                if not (self._arduino and self._arduino.is_open):
                    time.sleep(0.5)
                    continue
                try:
                    if self._arduino.in_waiting > 0:
                        data = self._arduino.read(self._arduino.in_waiting)
                        # デコード試行 → 失敗時はhex表示
                        try:
                            text = data.decode("utf-8").strip()
                        except UnicodeDecodeError:
                            text = f"[hex] {data.hex()}"
                        if text and text != last_text:
                            print(f"  [Arduino] 受信: {text}")
                            last_text = text
                    else:
                        time.sleep(0.05)
                except Exception as e:
                    print(f"  [Arduino] 受信エラー: {e}")
                    time.sleep(0.5)
        threading.Thread(target=_read, daemon=True).start()

    def _load_jp_font(self, size: int, weight: int = 3) -> pygame.font.Font:
        candidates = (
            f"/System/Library/Fonts/ヒラギノ角ゴシック W{weight}.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        )
        for path in candidates:
            try:
                return pygame.font.Font(path, size)
            except Exception:
                pass
        return pygame.font.SysFont(None, size)

    def _pil_to_surface(self, img: PILImage.Image) -> pygame.Surface:
        import numpy as np
        return pygame.surfarray.make_surface(np.array(img).swapaxes(0, 1))

    # ── マップ読み込み ────────────────────────────────────────

    def _load_map_images(self) -> list:
        png_path = os.path.join(os.path.dirname(__file__), "案内図", "facility_map.png")
        img = PILImage.open(png_path).convert("RGB")
        print(f"  [マップ] PNG読み込み完了")
        return [img]

    # ── Firebase 混雑状況ポーリング ───────────────────────────

    def _start_firebase_poller(self) -> None:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred = credentials.Certificate(str(_FIREBASE_CREDENTIAL))
        firebase_admin.initialize_app(cred)
        db = firestore.client()

        def _poll():
            while True:
                try:
                    docs = db.collection("tickets").stream()
                    result = {}
                    for doc in docs:
                        key = doc.id
                        exhibit = EXHIBIT_KEY_MAP.get(key)
                        if exhibit is None:
                            continue
                        data = doc.to_dict()
                        now_serving = data.get("nowServing", 0)
                        current_number = data.get("currentNumber", 0)
                        time_per = data.get("timePerPerson", 3)  # 1人あたりの体験時間（分）
                        waiting = max(now_serving - current_number, 0)
                        estimated_minutes = waiting * time_per + QUEUE_BUFFER_MINUTES
                        result[exhibit] = {"waiting": waiting, "minutes": estimated_minutes}
                    self._congestion = result
                    print(f"  [Firebase] 混雑状況更新: {result}")
                except Exception as e:
                    print(f"  [Firebase エラー] {e}")
                time.sleep(FIREBASE_POLL_INTERVAL)

        threading.Thread(target=_poll, daemon=True).start()

    def _build_congestion_text(self) -> str:
        if not self._congestion:
            return ""
        lines = ["【現在の整理券待ち状況】"]
        for exhibit, info in self._congestion.items():
            waiting = info["waiting"]
            minutes = info["minutes"]
            if waiting == 0:
                lines.append(f"- {exhibit}: 10分以内で体験できます")
            else:
                lines.append(f"- {exhibit}: 約{minutes}分待ち（{waiting}人待ち）")
        return "\n".join(lines)

    # ── 音量モニター ──────────────────────────────────────────

    def _start_volume_monitor(self) -> None:
        def _monitor():
            import struct
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                             input=True, frames_per_buffer=1024)
            try:
                while True:
                    data = stream.read(1024, exception_on_overflow=False)
                    shorts = struct.unpack('<' + 'h' * (len(data) // 2), data)
                    self._mic_rms = (sum(s * s for s in shorts) / len(shorts)) ** 0.5
            finally:
                stream.stop_stream()
                stream.close()
                pa.terminate()
        threading.Thread(target=_monitor, daemon=True).start()

    # ── 呼び込み音声の事前合成 ────────────────────────────────

    def _presynth_wav(self, text: str) -> str:
        return self._tts.synthesize(text)

    def _make_thinking_loop_sound(self) -> pygame.mixer.Sound:
        """ThinkingメッセージWAVを結合してループ再生用Soundを作成"""
        import wave, io
        frames_list = []
        params = None
        silence_sec = THINKING_INTERVAL
        for wav_path in self._thinking_wavs:
            with wave.open(wav_path) as wf:
                if params is None:
                    params = wf.getparams()
                frames_list.append(wf.readframes(wf.getnframes()))
                # セリフ間の無音
                silence_frames = int(wf.getframerate() * silence_sec) * wf.getnchannels() * wf.getsampwidth()
                frames_list.append(b'\x00' * silence_frames)
        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setparams(params)
            for f in frames_list:
                wf.writeframes(f)
        buf.seek(0)
        return pygame.mixer.Sound(buf)

    def _presynth_standby_messages(self) -> list[str]:
        print("  [事前合成] 呼び込みメッセージを合成中...")
        paths = [self._presynth_wav(msg) for msg in STANDBY_MESSAGES]
        print(f"  [事前合成] {len(paths)}件完了")
        return paths

    def _resynth_all(self) -> None:
        """音声パラメータ変更後に全事前合成音声を再生成する（バックグラウンド実行）"""
        if not self._resynth_lock.acquire(blocking=False):
            print("  [再合成] 既に実行中のためスキップ")
            return

        def _worker():
            try:
                print("  [再合成] 開始...")
                standby = [self._presynth_wav(msg) for msg in STANDBY_MESSAGES]
                greeting = self._presynth_wav("こんにちは！精密ラボの企画についてなんでもお答えいたします！")
                thinking = [self._presynth_wav(m) for m in THINKING_MESSAGES]
                farewell = [self._presynth_wav(m) for m in FAREWELL_RESPONSES]
                # thinking_loop_soundはSoundオブジェクトなのでwav差し替え後に再構築
                self._thinking_wavs = thinking
                loop_sound = self._make_thinking_loop_sound()
                # アトミックに差し替え
                self._standby_wavs = standby
                self._greeting_wav = greeting
                self._thinking_loop_sound = loop_sound
                self._farewell_wavs = farewell
                print("  [再合成] 完了")
            finally:
                self._resynth_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _draw_volume_bar(self) -> None:
        sw, sh = self.screen.get_size()
        bar_h = 28
        bar_y = sh - bar_h
        max_rms = 3000.0

        pygame.draw.rect(self.screen, (30, 30, 30), (0, bar_y, sw, bar_h))

        ratio = min(self._mic_rms / max_rms, 1.0)
        bar_w = int(sw * ratio)
        if ratio < 0.5:
            color = (0, 200, 80)
        elif ratio < 0.8:
            color = (230, 200, 0)
        else:
            color = (220, 50, 50)
        if bar_w > 0:
            pygame.draw.rect(self.screen, color, (0, bar_y + 4, bar_w, bar_h - 8))

        thresh_x = int(sw * min(MIC_THRESHOLD / max_rms, 1.0))
        pygame.draw.line(self.screen, (255, 255, 255), (thresh_x, bar_y), (thresh_x, sh), 2)

        font = pygame.font.SysFont(None, 20)
        rms_surf = font.render(
            f"RMS: {int(self._mic_rms)}  mic: {MIC_THRESHOLD}(白線)",
            True, (200, 200, 200)
        )
        self.screen.blit(rms_surf, (4, bar_y + 7))

    # ── 動画再生（待機モード）────────────────────────────────

    def _start_video_player(self) -> None:
        movies_dir = Path(__file__).parent / "movies"
        movies_dir.mkdir(exist_ok=True)

        self._video_generation += 1
        my_generation = self._video_generation

        def _play():
            exts = ("*.mp4", "*.mov", "*.avi", "*.mkv")
            while not self._stop_event.is_set() and self._video_generation == my_generation:
                files = []
                for ext in exts:
                    files.extend(sorted(movies_dir.glob(ext)))
                if not files:
                    time.sleep(1)
                    continue
                for video_path in files:
                    if self._stop_event.is_set() or self._video_generation != my_generation:
                        return
                    cap = cv2.VideoCapture(str(video_path))
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30
                    frame_interval = 1.0 / fps
                    while not self._stop_event.is_set() and self._video_generation == my_generation:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        try:
                            self._video_frame_queue.put_nowait(frame_rgb)
                        except queue.Full:
                            pass
                        time.sleep(frame_interval)
                    cap.release()

        threading.Thread(target=_play, daemon=True).start()

    # ── 音声合成 ──────────────────────────────────────────────

    def _get_response_wav(self, user_input: str, history: list):
        """Gemini回答生成 + VOICEVOX合成 + 画像読み込みをまとめて行う"""
        speech, exhibit, should_continue = self.get_response(user_input, history)
        self._ts("TTS合成 開始")
        wav_path = self._presynth_wav(speech)
        self._ts("TTS合成 完了")
        photo_pil = self._load_exhibit_photo(exhibit) if exhibit and EXHIBIT_LOCATIONS.get(exhibit) else None
        return wav_path, exhibit, speech, photo_pil, should_continue

    def speak(self, text: str, interruptible: bool = False) -> bool:
        self._ts("TTS合成 開始")
        tmp_path = self._tts.synthesize(text)
        self._ts("TTS合成 完了")

        interrupted = threading.Event()

        if interruptible:
            def _mic_monitor():
                import struct
                import pyaudio
                time.sleep(BARGE_IN_DELAY)
                pa = pyaudio.PyAudio()
                stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000,
                                 input=True, frames_per_buffer=1024)
                try:
                    while pygame.mixer.music.get_busy() and not self._stop_event.is_set():
                        data = stream.read(1024, exception_on_overflow=False)
                        shorts = struct.unpack('<' + 'h' * (len(data) // 2), data)
                        rms = (sum(s * s for s in shorts) / len(shorts)) ** 0.5
                        if rms > BARGE_IN_THRESHOLD:
                            interrupted.set()
                            break
                finally:
                    stream.stop_stream()
                    stream.close()
                    pa.terminate()
            threading.Thread(target=_mic_monitor, daemon=True).start()

        if self.state == RobotState.INTERACTION:
            self._subtitle = text
        pygame.mixer.music.load(tmp_path)
        self._set_speaking(True)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if self._stop_event.is_set() or interrupted.is_set():
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)
        self._set_speaking(False)
        self._subtitle = ""
        os.unlink(tmp_path)
        return not interrupted.is_set()

    # ── 音声認識 ──────────────────────────────────────────────

    def listen(self) -> tuple[str | None, bool]:
        """(認識テキスト or None, タイムアウトか) を返す"""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = self._mic_threshold
        recognizer.dynamic_energy_threshold = False

        self._is_listening = True
        try:
            with sr.Microphone(device_index=self._mic_index) as source:
                self._ts("マイク待機 開始")
                try:
                    audio = recognizer.listen(
                        source, timeout=self._listen_timeout, phrase_time_limit=10
                    )
                    self._ts("録音 完了")
                except sr.WaitTimeoutError:
                    return None, True  # タイムアウト
        finally:
            self._is_listening = False

        try:
            self._ts("Google STT 開始")
            text = recognizer.recognize_google(audio, language="ja-JP")
            self._ts(f"Google STT 完了 → 「{text}」")
            return text.strip() or None, False
        except sr.UnknownValueError:
            print("  [音声認識] 聞き取れず、再試行")
            return None, False  # 認識失敗（タイムアウトではない）
        except Exception as e:
            print(f"  [音声認識エラー] {e}")
            return None, False

    # ── LLM ──────────────────────────────────────────────────

    def get_response(self, user_input: str, history: list) -> tuple[str, str | None]:
        prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "system_prompt.txt")
        with open(prompt_path, encoding="utf-8") as f:
            system_prompt = f.read()

        # 混雑状況をsystem_instructionに付加
        congestion_text = self._build_congestion_text()
        full_system = system_prompt + (f"\n\n{congestion_text}" if congestion_text else "")

        messages = [{"role": "system", "content": full_system}]
        messages += list(history)
        messages.append({"role": "user", "content": user_input})

        try:
            self._ts("OpenAI 開始")
            response = self.client.chat.completions.create(
                model="gpt-5.4-mini",
                messages=messages,
                response_format={"type": "json_object"},
            )
            self._ts("OpenAI 完了")
            raw = response.choices[0].message.content
            print(f"  [OpenAI JSON] {raw}")
            data = json.loads(raw)
            exhibit_key = data.get("exhibit") or None
            exhibit = EXHIBIT_KEY_MAP.get(exhibit_key) if exhibit_key else None
            print(f"  [exhibit] key={exhibit_key} → name={exhibit}")
            return data.get("speech", ""), exhibit, data.get("continue", True)
        except json.JSONDecodeError:
            print(f"  [OpenAI JSONパースエラー] {raw}")
            return raw, None, True
        except Exception as e:
            print(f"  [Gemini エラー] {e}")
            return "すみません、もう一度お願いします", None, True

    # ── 企画写真 ──────────────────────────────────────────────

    def _load_exhibit_photo(self, exhibit_name: str) -> PILImage.Image:
        photo_dir = os.path.join(os.path.dirname(__file__), "assets", "photos")
        for ext in ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG"):
            path = os.path.join(photo_dir, f"{exhibit_name}.{ext}")
            if os.path.exists(path):
                return PILImage.open(path).convert("RGB")

        # ポスターPDFを試みる（最初のページをレンダリング）
        pdf_name = EXHIBIT_POSTER_MAP.get(exhibit_name)
        if pdf_name:
            pdf_path = os.path.join(photo_dir, pdf_name)
            if os.path.exists(pdf_path):
                try:
                    doc = fitz.open(pdf_path)
                    page = doc[0]
                    mat = fitz.Matrix(2.0, 2.0)
                    pix = page.get_pixmap(matrix=mat)
                    doc.close()
                    return PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except Exception as e:
                    print(f"  [PDF読込エラー] {pdf_name}: {e}")

        # ダミー画像生成
        img = PILImage.new("RGB", (640, 480), (220, 220, 220))
        draw = PILImageDraw.Draw(img)
        draw.rectangle([10, 10, 629, 469], outline=(180, 180, 180), width=4)
        try:
            font = PILImageFont.truetype(
                "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", size=40
            )
            draw.text((320, 240), exhibit_name, fill=(100, 100, 100), font=font, anchor="mm")
        except Exception:
            draw.text((320, 240), exhibit_name, fill=(100, 100, 100))
        return img

    # ── 吹き出し描画 ──────────────────────────────────────────

    def _draw_congestion_bubbles(self, map_x: int, map_y: int, mw: int, mh: int) -> None:
        """マップ上の各企画に待ち時間吹き出しを描画する"""
        queue_exhibits = set(EXHIBIT_KEY_MAP[k] for k in ("truck","space","switch","arm","soccer","pong","shooting","tank") if k in EXHIBIT_KEY_MAP)

        for exhibit, loc in EXHIBIT_LOCATIONS.items():
            bx = map_x + int(loc.get("bx", loc["x"]) * mw)
            by = map_y + int(loc.get("by", loc["y"]) * mh)

            if exhibit in self._congestion:
                info = self._congestion[exhibit]
                waiting = info["waiting"]
                minutes = info["minutes"]
                if waiting == 0:
                    text = "10分以内"
                    color = (20, 150, 65)
                elif minutes <= 10:
                    text = f"{minutes}分待ち"
                    color = (190, 120, 0)
                else:
                    text = f"{minutes}分待ち"
                    color = (180, 30, 50)
            elif exhibit in queue_exhibits:
                continue  # Firebase接続前は表示しない
            else:
                text = "整理券不要"
                color = (60, 90, 160)

            self._draw_bubble(text, bx, by, color)

    def _draw_bubble(self, text: str, cx: int, cy: int, color: tuple) -> None:
        stroke = 2
        pad_x, pad_y = 10, 8
        arrow_w = 10
        arrow_h = 10
        r_outer = 10
        r_inner = r_outer - stroke

        text_surf = self._bubble_font_main.render(text, True, color)
        tw, th = text_surf.get_size()

        inner_w = tw + pad_x * 2
        inner_h = th + pad_y * 2
        outer_w = inner_w + stroke * 2
        outer_h = inner_h + stroke * 2
        total_h = outer_h + arrow_h

        surf = pygame.Surface((outer_w, total_h), pygame.SRCALPHA)
        mid = outer_w // 2

        # 枠（ボーダー色）
        pygame.draw.rect(surf, (*color, 220), (0, 0, outer_w, outer_h), border_radius=r_outer)
        pygame.draw.polygon(surf, (*color, 220), [
            (mid - arrow_w, outer_h - 2),
            (mid + arrow_w, outer_h - 2),
            (mid, total_h),
        ])

        # 内側（白）—— 三角の底辺を内側矩形の中にめり込ませて境目の線を消す
        pygame.draw.rect(surf, (255, 255, 255, 248), (stroke, stroke, inner_w, inner_h), border_radius=r_inner)
        pygame.draw.polygon(surf, (255, 255, 255, 248), [
            (mid - arrow_w + stroke, outer_h - stroke - 1),
            (mid + arrow_w - stroke, outer_h - stroke - 1),
            (mid, total_h - stroke),
        ])

        # テキスト
        surf.blit(text_surf, (stroke + pad_x, stroke + pad_y))

        self.screen.blit(surf, (cx - outer_w // 2, cy - total_h))

    # ── マップ表示 ────────────────────────────────────────────

    def show_map_with_highlight(self, exhibit_name: str) -> None:
        exhibit_name = EXHIBIT_ALIASES.get(exhibit_name, exhibit_name)
        loc = EXHIBIT_LOCATIONS.get(exhibit_name)
        if loc is None:
            print(f"  [座標未登録] {exhibit_name}")
            return

        # 企画写真
        photo_img = self._load_exhibit_photo(exhibit_name)

        new_map = self._pil_to_surface(self.map_images[0])
        new_right = self._pil_to_surface(photo_img)
        with self._surface_lock:
            self._map_surface = new_map
            self._right_surface = new_right
            self._highlight_loc = loc

    # ── モードループ ──────────────────────────────────────────

    def _standby_loop(self) -> None:
        print("[待機モード] 開始")
        self._show_cam_debug = True
        # マップをリセット（ハイライトなし）、動画再生開始
        new_map = self._pil_to_surface(self.map_images[0])
        with self._surface_lock:
            self._map_surface = new_map
            self._right_surface = None
            self._highlight_loc = None
        self._start_video_player()

        my_gen = self._current_gen
        def _stale(): return self._stop_event.is_set() or self._current_gen != my_gen

        msg_index = 0
        while not _stale():
            idx = msg_index % len(self._standby_wavs)
            msg_index += 1
            print(f"  [呼び込み] {STANDBY_MESSAGES[idx]}")
            # 合成済みWAVを直接再生（合成処理なし）
            pygame.mixer.music.load(self._standby_wavs[idx])
            self._set_speaking(True)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if _stale():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
            self._set_speaking(False)
            if _stale():
                return
            interval = random.uniform(5, 7)
            for _ in range(int(interval * 10)):
                if _stale():
                    return
                time.sleep(0.1)

    def _interaction_loop(self) -> None:
        print("[対話モード] 開始")
        my_gen = self._current_gen
        def _stale(): return self._stop_event.is_set() or self._current_gen != my_gen

        # 1秒後にカメラデバッグビューを非表示
        def _hide_cam():
            time.sleep(1.0)
            if not _stale():
                self._show_cam_debug = False
        threading.Thread(target=_hide_cam, daemon=True).start()
        # 右パネルをクリア（動画停止後）
        with self._surface_lock:
            self._right_surface = None
        self._subtitle = "こんにちは！精密ラボの企画についてなんでもお答えいたします！"
        pygame.mixer.music.load(self._greeting_wav)
        self._set_speaking(True)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if _stale():
                pygame.mixer.music.stop()
                self._set_speaking(False)
                self._subtitle = ""
                return
            time.sleep(0.05)
        self._set_speaking(False)
        self._subtitle = ""
        history: list = []  # 対話モード中の会話履歴（モード終了で破棄）
        while not _stale():
            # 音声認識開始前にポスターを非表示に戻す
            with self._surface_lock:
                self._right_surface = None
                self._highlight_loc = None
            user_input, timed_out = self.listen()
            if _stale():
                return
            if timed_out:
                print("  [タイムアウト] 待機モードへ戻ります")
                self._switch_to(RobotState.STANDBY)
                return
            if user_input is None:
                continue  # 聞き取れなかっただけ → 再度マイク待機
            t_user_end = time.time()
            print(f"  [ユーザー] {user_input}")

            # 別れの言葉はGeminiを介さず即座に返答して待機モードへ
            if any(kw in user_input for kw in FAREWELL_KEYWORDS):
                idx = random.randrange(len(FAREWELL_RESPONSES))
                farewell_text = FAREWELL_RESPONSES[idx]
                wav = self._farewell_wavs[idx]
                self._subtitle = farewell_text
                pygame.mixer.music.load(wav)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if _stale():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.05)
                self._subtitle = ""
                time.sleep(1)
                self._switch_to(RobotState.STANDBY)
                return

            # Gemini + VOICEVOX合成をバックグラウンドで実行しながら考え中セリフを再生
            # loops=-1で無限ループ再生 → SDL(C)が管理するためVOICEVOXのGIL保持中も継続
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._get_response_wav, user_input, history)
                self._set_speaking(True)
                self._thinking_channel.play(self._thinking_loop_sound, loops=-1)
                while not future.done():
                    if _stale():
                        self._thinking_channel.stop()
                        self._set_speaking(False)
                        return
                    time.sleep(0.05)
                self._thinking_channel.stop()
                self._set_speaking(False)
                if _stale():
                    return
                wav_path, exhibit, speech, photo_pil, should_continue = future.result()

            # 会話履歴に追記（次回のOpenAI呼び出しに引き継ぐ）
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant", "content": json.dumps({"speech": speech, "exhibit": exhibit}, ensure_ascii=False)})

            # Surface変換はロックの外で行い、代入だけロック内で（render()のブロック防止）
            if exhibit and photo_pil:
                loc = EXHIBIT_LOCATIONS.get(exhibit)
                new_map = self._pil_to_surface(self.map_images[0])
                new_right = self._pil_to_surface(photo_pil)
                with self._surface_lock:
                    self._map_surface = new_map
                    self._right_surface = new_right
                    self._highlight_loc = loc
            elif exhibit:
                print(f"  [座標未登録] {exhibit}")
            else:
                # 企画に無関係な質問 → 写真とマーカーをクリア
                with self._surface_lock:
                    self._right_surface = None
                    self._highlight_loc = None
            print(f"  [ロボット] {speech}")
            self._ts("発話 開始")
            # 合成済みWAVを直接再生
            self._subtitle = speech
            pygame.mixer.music.load(wav_path)
            self._set_speaking(True)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if _stale():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
            self._set_speaking(False)
            self._subtitle = ""
            os.unlink(wav_path)

            if not should_continue:
                print("  [会話終了] 待機モードへ戻ります")
                time.sleep(1)
                self._switch_to(RobotState.STANDBY)
                return

    # ── 状態遷移 ──────────────────────────────────────────────

    def _switch_to(self, new_state: RobotState) -> None:
        self._stop_event.set()
        self._is_listening = False   # listen()がブロック中でもインジケーターを即消す
        self._video_generation += 1  # 動画プレーヤースレッドを確実に停止
        self._current_gen += 1       # 旧スレッドを失効させる
        if (self._current_thread and self._current_thread.is_alive()
                and self._current_thread != threading.current_thread()):
            self._current_thread.join(timeout=2)
        self._stop_event.clear()
        # 動画キューを空にして古いフレームが写真を上書きしないようにする
        while not self._video_frame_queue.empty():
            try:
                self._video_frame_queue.get_nowait()
            except queue.Empty:
                break
        self.state = new_state
        target = (
            self._standby_loop
            if new_state == RobotState.STANDBY
            else self._interaction_loop
        )
        self._current_thread = threading.Thread(target=target, daemon=True)
        self._current_thread.start()

    def toggle(self) -> None:
        next_state = (
            RobotState.INTERACTION
            if self.state == RobotState.STANDBY
            else RobotState.STANDBY
        )
        print(f"[モード切替] {self.state.value} → {next_state.value}")
        self._switch_to(next_state)

    # ── 描画 ──────────────────────────────────────────────────

    def _render(self) -> None:
        sw, sh = self.screen.get_size()
        bar_h = 28
        content_h = sh - bar_h

        with self._surface_lock:
            map_surf = self._map_surface

        self.screen.fill((35, 10, 20))

        # マップを全画面に拡張（上部に吹き出し用の余白を確保）
        top_h = 50
        avail_h = content_h - top_h
        scale_m = min(sw / map_surf.get_width(), avail_h / map_surf.get_height())
        mw = int(map_surf.get_width() * scale_m)
        mh = int(map_surf.get_height() * scale_m)
        map_x = sw // 2 - mw // 2
        map_y = top_h + avail_h // 2 - mh // 2
        scaled_map = pygame.transform.smoothscale(map_surf, (mw, mh))
        self.screen.blit(scaled_map, (map_x, map_y))

        # 吹き出し（待ち時間 / 整理券不要）
        self._draw_congestion_bubbles(map_x, map_y, mw, mh)

        # 右上ロゴ
        margin_w = sw - (map_x + mw)
        logo_w = margin_w - 10
        logo_h = int(self._logo_surf.get_height() * logo_w / self._logo_surf.get_width())
        logo_scaled = pygame.transform.smoothscale(self._logo_surf, (logo_w, logo_h))
        logo_x = map_x + mw + (margin_w - logo_w) // 2
        logo_y = map_y
        self.screen.blit(logo_scaled, (logo_x, logo_y))

        # サイドラベル（縦書き）
        label_cy = top_h + avail_h // 2
        left_cx  = map_x // 2
        right_cx = map_x + mw + (sw - map_x - mw) // 2
        # groups: グループのリスト。各グループは (text, vertical, font) のリスト
        # グループ間には大きめの余白を入れる
        def _build_items(groups):
            """グループリストからサーフェスリスト（グループ境界にNoneを挿入）を返す"""
            result = []
            for i, group in enumerate(groups):
                if i > 0:
                    result.append(None)  # グループ間の区切り
                for text, vertical, font in group:
                    if vertical:
                        result += [font.render(c, True, (255, 255, 255)) for c in text]
                    else:
                        s = font.render(text, True, (255, 255, 255))
                        result.append(pygame.transform.rotozoom(s, -90, 1.0))
            return result

        char_gap = 2
        right_cx_center = map_x + mw + margin_w // 2
        right_col_l = right_cx_center - 22   # 左列: （精密工学科企画）
        right_col_r = right_cx_center + 22   # 右列: →精密ラボ

        char_h_small = self._side_font_small.get_height()
        for groups, cx, y_offset in [
            ([[("←都市工学科企画", True, self._side_font_mid)]], left_cx, -60),
            ([[("→精密ラボ", True, self._side_font)]], right_col_r, 0),
            ([[("（", False, self._side_font_small), ("精密工学科企画", True, self._side_font_small), ("）", False, self._side_font_small)]], right_col_l, char_h_small * 3),
        ]:
            items = _build_items(groups)
            total_h = sum(s.get_height() for s in items) + char_gap * (len(items) - 1)
            y = label_cy - total_h // 2 + y_offset
            for s in items:
                self.screen.blit(s, (cx - s.get_width() // 2, y))
                y += s.get_height() + char_gap

        # 企画ポスター（対話モードで企画が特定されたとき画面半分に全画面表示）
        with self._surface_lock:
            photo_surf = self._right_surface
            hloc = self._highlight_loc
        if photo_surf is not None:
            # 企画がマップ左側 → ポスターを画面右半分、右側 → 左半分
            ex = hloc["x"] if hloc is not None else 0.0
            on_left = ex <= 0.5
            area_x = sw // 2 if on_left else 0
            area_w = sw - sw // 2 if on_left else sw // 2
            subtitle_reserve = 160  # 字幕ボックス分の余白
            area_h = content_h - subtitle_reserve
            scale_p = min(area_w / photo_surf.get_width(), area_h / photo_surf.get_height())
            pw = int(photo_surf.get_width() * scale_p)
            ph = int(photo_surf.get_height() * scale_p)
            px = area_x + (area_w - pw) // 2
            py = (area_h - ph) // 2
            scaled_photo = pygame.transform.smoothscale(photo_surf, (pw, ph))
            self.screen.blit(scaled_photo, (px, py))

        # ハイライト丸アニメーション
        with self._surface_lock:
            loc = self._highlight_loc
        if loc is not None:
            t = time.time()
            pulse = math.sin(t * 4) * 0.25 + 1.0   # 0.75〜1.25 で脈動
            ripple = (math.sin(t * 3) + 1) / 2      # 0〜1 で波紋
            base_r = max(20, int(mw * 0.045))
            cx = map_x + int(loc["x"] * mw)
            cy = map_y + int(loc["y"] * mh)

            # 波紋（外側に広がる半透明円）
            ripple_surf = pygame.Surface((sw, content_h), pygame.SRCALPHA)
            ripple_r = int(base_r * (1.2 + ripple * 0.8))
            ripple_alpha = int(180 * (1 - ripple))
            pygame.draw.circle(ripple_surf, (255, 50, 50, ripple_alpha), (cx, cy), ripple_r, 5)
            self.screen.blit(ripple_surf, (0, 0))

            # メイン丸（脈動）
            r = int(base_r * pulse)
            pygame.draw.circle(self.screen, (255, 255, 255), (cx, cy), r + 5, 5)  # 白縁
            pygame.draw.circle(self.screen, (230, 30, 30), (cx, cy), r, 6)         # 赤丸

        # マイク聞き取り中インジケーター
        if self._is_listening:
            t = time.time()
            ix = sw // 2
            iy = content_h // 2

            # 半透明の暗いオーバーレイ
            overlay = pygame.Surface((sw, content_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 160))
            self.screen.blit(overlay, (0, 0))

            # RMSに反応する外側の波紋
            rms_ratio = min(self._mic_rms / 3000.0, 1.0)
            ripple_r = int(90 + rms_ratio * 60 + math.sin(t * 8) * 8)
            ripple_surf = pygame.Surface((sw, content_h), pygame.SRCALPHA)
            pygame.draw.circle(ripple_surf, (80, 180, 255, 80), (ix, iy), ripple_r)
            self.screen.blit(ripple_surf, (0, 0))

            # メインの青い円
            base_r = int(70 + rms_ratio * 40)
            pygame.draw.circle(self.screen, (40, 130, 255), (ix, iy), base_r)
            pygame.draw.circle(self.screen, (120, 200, 255), (ix, iy), base_r, 4)

            # テキスト
            label = self._listen_font.render("音声認識中...", True, (255, 255, 255))
            self.screen.blit(label, (ix - label.get_width() // 2, iy + base_r + 20))

        # デバッグカメラビュー（左下固定、対話モード中は非表示）
        if self._show_cam_debug:
            cam_surf = self._face_detector.get_debug_surface(width=200)
            if cam_surf is not None:
                self.screen.blit(cam_surf, (10, sh - cam_surf.get_height() - 10))

        # 字幕（対話モード中の発話内容）
        if self.state == RobotState.INTERACTION and self._subtitle:
            # TTS読み仮名 → 表示用テキストに変換
            _tts_to_display = [
                ("エーアイ", "AI"), ("エーアール", "AR"),
                ("ブイアール", "VR"), ("せいみつポン", "せいみつPONG!"),
            ]
            chars = self._subtitle
            for src, dst in _tts_to_display:
                chars = chars.replace(src, dst)
            # 40文字で折り返し
            lines = [chars[i:i+40] for i in range(0, len(chars), 40)]
            pad_x, pad_y, line_gap = 20, 10, 6
            line_surfs = [self._subtitle_font.render(l, True, (255, 255, 255)) for l in lines]
            box_w = max(s.get_width() for s in line_surfs) + pad_x * 2
            line_h = line_surfs[0].get_height()
            box_h = line_h * len(line_surfs) + line_gap * (len(line_surfs) - 1) + pad_y * 2
            bx = sw // 2 - box_w // 2
            by = sh - bar_h - box_h - 10
            bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            self.screen.blit(bg, (bx, by))
            for i, s in enumerate(line_surfs):
                self.screen.blit(s, (bx + pad_x, by + pad_y + i * (line_h + line_gap)))

        self._draw_volume_bar()

    # ── 顔検知プログレスUI ───────────────────────────────────

    def _render_face_detect_ui(self, progress: float) -> None:
        """
        顔検知中に画面中央に表示する顔マーク＋円形プログレス。
        progress: 0.0（フレーム蓄積開始）〜 1.0（対話モード切替直前）
        フェーズ1(0〜0.5): フレーム蓄積, フェーズ2(0.5〜1.0): intentカウント
        """
        sw, sh = self.screen.get_size()
        cx = sw // 2
        cy = sh // 2 - 20

        ring_r  = 90   # 外周リングの半径
        ring_w  = 14   # リングの太さ
        face_r  = 68   # 顔円の半径

        # progress に応じて水色→緑にシフト
        r = 0
        g = int(180 + progress * 75)
        b = int(255 - progress * 200)
        arc_color = (r, g, b)

        # 背景リング（暗めのグレー）
        pygame.draw.circle(self.screen, (50, 50, 60), (cx, cy), ring_r, ring_w)

        # 円形プログレスアーク（12時を終点に、時計回りに progress 分だけ塗る）
        if progress > 0.01:
            arc_rect = pygame.Rect(cx - ring_r, cy - ring_r, ring_r * 2, ring_r * 2)
            start_angle = math.pi / 2 - progress * 2 * math.pi
            stop_angle  = math.pi / 2
            pygame.draw.arc(self.screen, arc_color, arc_rect, start_angle, stop_angle, ring_w)

        # 顔の背景円
        pygame.draw.circle(self.screen, (30, 35, 55), (cx, cy), face_r)
        pygame.draw.circle(self.screen, (70, 80, 110), (cx, cy), face_r, 2)

        # 目（左右）
        eye_y  = cy - 16
        eye_dx = 22
        eye_r  = 8
        pygame.draw.circle(self.screen, arc_color, (cx - eye_dx, eye_y), eye_r)
        pygame.draw.circle(self.screen, arc_color, (cx + eye_dx, eye_y), eye_r)

        # 口（弧）
        mouth_rect = pygame.Rect(cx - 22, cy + 8, 44, 28)
        pygame.draw.arc(self.screen, arc_color, mouth_rect, math.pi, 0, 4)

        # テキスト「顔検知中」（縁取り：暗い色でずらして描いてから本体色を重ねる）
        label = self._face_detect_font.render("顔検知中", True, arc_color)
        outline = self._face_detect_font.render("顔検知中", True, (0, 0, 0))
        tx = cx - label.get_width() // 2
        ty = cy + ring_r + 12
        outline_w = 3
        for dx in range(-outline_w, outline_w + 1):
            for dy in range(-outline_w, outline_w + 1):
                if dx != 0 or dy != 0:
                    self.screen.blit(outline, (tx + dx, ty + dy))
        self.screen.blit(label, (tx, ty))

    # ── 顔検知チューニング ────────────────────────────────────

    def _load_mic_tune(self) -> None:
        if not FACE_TUNE_SAVE_PATH.exists():
            return
        try:
            data = json.loads(FACE_TUNE_SAVE_PATH.read_text(encoding="utf-8"))
            for attr, *_ in MIC_PARAMS:
                if attr in data:
                    setattr(self, attr, data[attr])
        except Exception as e:
            print(f"  [マイクチューニング] ロード失敗: {e}")

    def _save_mic_tune(self) -> None:
        try:
            data = json.loads(FACE_TUNE_SAVE_PATH.read_text(encoding="utf-8")) if FACE_TUNE_SAVE_PATH.exists() else {}
        except Exception:
            data = {}
        for attr, *_ in MIC_PARAMS:
            data[attr] = getattr(self, attr)
        FACE_TUNE_SAVE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [マイクチューニング] 保存しました: {FACE_TUNE_SAVE_PATH}")

    def _handle_face_tune_key(self, key) -> None:
        total = len(FACE_PARAMS) + len(MIC_PARAMS)
        if key == pygame.K_UP:
            self._face_tuning_idx = (self._face_tuning_idx - 1) % total
        elif key == pygame.K_DOWN:
            self._face_tuning_idx = (self._face_tuning_idx + 1) % total
        elif key in (pygame.K_LEFT, pygame.K_RIGHT):
            idx = self._face_tuning_idx
            if idx < len(FACE_PARAMS):
                attr, label, step, minv, maxv = FACE_PARAMS[idx]
                obj = self._face_detector
                is_int = (attr == "stability_frames")
            else:
                attr, label, step, minv, maxv, is_int = MIC_PARAMS[idx - len(FACE_PARAMS)]
                obj = self
            cur = getattr(obj, attr)
            delta = step if key == pygame.K_RIGHT else -step
            new_val = int(max(minv, min(maxv, cur + delta))) if is_int else round(max(minv, min(maxv, cur + delta)), 4)
            setattr(obj, attr, new_val)
            print(f"  [チューニング] {label.split('［')[0].strip()} = {new_val}")

    def _render_face_tuning_panel(self, above_h: int = 0) -> None:
        fd = self._face_detector
        font = self._face_tune_font
        line_h = font.get_height() + 4
        pad = 10
        face_labels = [
            (f"  {'→' if i == self._face_tuning_idx else ' '} "
             f"{label}: {getattr(fd, attr):.2f}" if attr != "stability_frames"
             else f"  {'→' if i == self._face_tuning_idx else ' '} {label}: {getattr(fd, attr)}")
            for i, (attr, label, *_) in enumerate(FACE_PARAMS)
        ]
        mic_offset = len(FACE_PARAMS)
        mic_labels = [
            f"  {'→' if (mic_offset + i) == self._face_tuning_idx else ' '} {label}: {int(getattr(self, attr))}"
            for i, (attr, label, *_) in enumerate(MIC_PARAMS)
        ]
        header = "[ 顔検知チューニング ]"
        mic_header = "  --- 音声認識 ---"
        footer = "F:閉じる  ↑↓:選択  ←→:変更  S:保存"
        lines = [header] + face_labels + [mic_header] + mic_labels + [footer]
        panel_w = max(font.size(l)[0] for l in lines) + pad * 2
        panel_h = line_h * len(lines) + pad * 2

        sw, sh = self.screen.get_size()
        px, py = 10, sh - panel_h - 10 - above_h

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        self.screen.blit(panel, (px, py))

        # lines の構成: [header, *face_labels, mic_header, *mic_labels, footer]
        # インデックス→選択中パラメータの対応
        face_end = 1 + len(FACE_PARAMS)          # face_labels の終わり（exclusive）
        mic_start = face_end + 1                  # mic_labels の開始（mic_headerの次）
        for i, line in enumerate(lines):
            if i == 0 or i == len(lines) - 1:
                color = (255, 220, 80)            # ヘッダー・フッター
            elif i == face_end:
                color = (180, 180, 255)           # --- 音声認識 --- セクション区切り
            elif 1 <= i < face_end:
                color = (100, 255, 150) if (i - 1) == self._face_tuning_idx else (220, 220, 220)
            else:
                mic_i = i - mic_start
                color = (100, 255, 150) if (len(FACE_PARAMS) + mic_i) == self._face_tuning_idx else (220, 220, 220)
            surf = font.render(line, True, color)
            self.screen.blit(surf, (px + pad, py + pad + i * line_h))

    # ── 顔検知によるモード自動切替 ────────────────────────────

    def _start_face_watcher(self) -> None:
        self._face_watcher_stop = threading.Event()
        self._face_detector = FaceDetector()
        threading.Thread(target=self._face_watcher_loop, daemon=True).start()

    def _face_watcher_loop(self) -> None:
        intent_start: float | None = None
        _last_reason: str = ""
        while not self._face_watcher_stop.is_set():
            if self.state != RobotState.STANDBY:
                intent_start = None
                _last_reason = ""
                time.sleep(0.2)
                continue
            try:
                detected = self._face_detector.is_intent_detected()
            except Exception as e:
                print(f"  [顔検知エラー] {e}")
                time.sleep(0.2)
                continue

            reason = self._face_detector._debug_reason
            if reason != _last_reason:
                elapsed = f" ({time.time()-intent_start:.1f}s/{self._face_detector.trigger_seconds}s)" if intent_start else ""
                print(f"  [顔検知] {reason}{elapsed}")
                _last_reason = reason

            fd = self._face_detector

            if detected:
                if intent_start is None:
                    intent_start = time.time()
                elapsed = time.time() - intent_start
                self._scan_progress = min(elapsed / fd.trigger_seconds, 1.0)
                if elapsed >= fd.trigger_seconds:
                    print("  [顔検知] → 対話モードへ切替")
                    intent_start = None
                    _last_reason = ""
                    self._scan_progress = None
                    self._switch_to(RobotState.INTERACTION)
                    time.sleep(1.0)  # 切替直後の再トリガー防止
            else:
                intent_start = None
                self._scan_progress = None

            time.sleep(FACE_LOOP_INTERVAL)

    # ── 起動 ──────────────────────────────────────────────────

    def run(self) -> None:
        print("=" * 40)
        print("  精密ラボ 受付ロボット 起動")
        print("  Space : モード切替 / Ctrl+C : 終了")
        print("=" * 40)

        self._switch_to(RobotState.STANDBY)

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.toggle()
                    elif event.key == pygame.K_f:
                        self._face_tuning_visible = not self._face_tuning_visible
                    elif self._face_tuning_visible and event.key in (
                        pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT
                    ):
                        self._handle_face_tune_key(event.key)
                    elif self._face_tuning_visible and event.key == pygame.K_s:
                        self._face_detector.save_tune()
                        self._save_mic_tune()
                    elif isinstance(self._tts, GttsTTS):
                        changed = True
                        if event.key == pygame.K_UP:
                            self._tts.pitch = round(self._tts.pitch + 0.05, 2)
                        elif event.key == pygame.K_DOWN:
                            self._tts.pitch = round(max(0.5, self._tts.pitch - 0.05), 2)
                        elif event.key == pygame.K_RIGHT:
                            self._tts.speed = round(self._tts.speed + 0.05, 2)
                        elif event.key == pygame.K_LEFT:
                            self._tts.speed = round(max(0.5, self._tts.speed - 0.05), 2)
                        elif event.key == pygame.K_t:
                            self._tts.next_tld()
                        else:
                            changed = False
                        if changed:
                            print(f"  [TTS] speed={self._tts.speed}  pitch={self._tts.pitch}  tld={self._tts.tld}")
                            self._resynth_all()

            self._render()

            # 顔検知プログレスUI（待機モードで顔が見つかっているときのみ）
            if self.state == RobotState.STANDBY and self._scan_progress is not None:
                self._render_face_detect_ui(self._scan_progress)

            if self._face_tuning_visible:
                self._render_face_tuning_panel()

            pygame.display.flip()
            clock.tick(30)

        self._stop_event.set()
        self._face_watcher_stop.set()
        self._face_detector.release()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="精密ラボ 受付ロボット")
    parser.add_argument(
        "--serial",
        default=None,
        metavar="PORT",
        help="ArduinoのシリアルポートURL (例: /dev/tty.usbmodem1101)",
    )
    parser.add_argument(
        "--mic",
        type=int,
        default=None,
        metavar="INDEX",
        help="マイクのデバイスインデックス (例: 0)",
    )
    args = parser.parse_args()
    EntranceRobot(serial_port=args.serial, mic_index=args.mic).run()
