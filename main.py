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

import cv2
import math
import multiprocessing
import tempfile
import threading
import time
from enum import Enum
from pathlib import Path

import argparse
import subprocess

import fitz  # PyMuPDF
from google import genai
from google.genai import types
import PIL.Image as PILImage
import PIL.ImageDraw as PILImageDraw
import PIL.ImageFont as PILImageFont
import pygame
import speech_recognition as sr

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
    "room": "現実拡張空間",
    "media":    "精密メディアアート",
    "balloon":  "バルーンロボット",
    "switch":   "せいみつスイッチ",
    "arm":      "ロボットアーム",
    "chess":    "ロボットチェス",
    "soccer":   "スーパーロボットサッカー",
    # 1階
    "lab":      "AI精密ラボ",
    "connect4": "立体四目並べ",
    "pong":     "せいみつPONG",
    "shooting": "お絵描きシューティング",
    "tank":     "ARタンク",
    "balance":  "3軸制御バランスキューブ",
    "dress":    "AI着せ替えカメラ",
    "janken":   "じゃんけんAI",
}
FIREBASE_POLL_INTERVAL = 30  # 混雑状況の更新間隔（秒）
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

# ── 企画場所の座標（マップ画像全体を1.0とした相対座標）────────────────────────
# x,y  : ハイライト丸の中心（マップ全体を1.0とした相対座標）
# bx,by : 吹き出し矢印の先端（省略時は x,y を使用）
EXHIBIT_LOCATIONS = {
    # 3階
    "ジャングル・スコープ":              {"x": 0.3930, "y": 0.0860, "bx": 0.3810, "by": 0.0160},
    "現実拡張空間":                  {"x": 0.5390, "y": 0.0980, "bx": 0.5270, "by": 0.0190},
    "精密メディアアート":               {"x": 0.9210, "y": 0.0950, "bx": 0.9180, "by": 0.0100},
    "バルーンロボット":                {"x": 0.3240, "y": 0.3350, "bx": 0.3000, "by": 0.2820},
    "せいみつスイッチ":                {"x": 0.5090, "y": 0.3530, "bx": 0.4940, "by": 0.2790},
    "ロボットアーム":                 {"x": 0.6510, "y": 0.3500, "bx": 0.6330, "by": 0.2730},
    "ロボットチェス":                 {"x": 0.7890, "y": 0.3470, "bx": 0.7680, "by": 0.2730},
    "スーパーロボットサッカー":            {"x": 0.9220, "y": 0.3410, "bx": 0.8980, "by": 0.2760},
    # 1階
    "AI精密ラボ":                  {"x": 0.3100, "y": 0.6260, "bx": 0.2950, "by": 0.5790},
    "立体四目並べ":                  {"x": 0.4250, "y": 0.6290, "bx": 0.4280, "by": 0.5670},
    "せいみつPONG":                {"x": 0.5580, "y": 0.6290, "bx": 0.5760, "by": 0.5760},
    "お絵描きシューティング":             {"x": 0.8560, "y": 0.6170, "bx": 0.8470, "by": 0.5550},
    "ARタンク":                   {"x": 0.5590, "y": 0.8500, "bx": 0.5440, "by": 0.7820},
    "3軸制御バランスキューブ":            {"x": 0.7020, "y": 0.8290, "bx": 0.6900, "by": 0.7700},
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


class RobotState(Enum):
    STANDBY = "standby"
    INTERACTION = "interaction"


# ── ロボット本体 ───────────────────────────────────────────────────────────────

class EntranceRobot:
    def __init__(self) -> None:
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.state = RobotState.STANDBY
        self._stop_event = threading.Event()
        self._current_thread: threading.Thread | None = None

        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("精密ラボ. 案内マップ")

        self.map_images = self._load_map_images()

        # 左右それぞれのサーフェスを管理
        self._surface_lock = threading.Lock()
        self._map_surface: pygame.Surface = self._pil_to_surface(self.map_images[0])
        self._right_surface: pygame.Surface | None = None
        self._highlight_loc: dict | None = None  # アニメーション用ハイライト座標

        self._tts: TTSEngine = create_tts_engine()
        self._standby_wavs: list[str] = self._presynth_standby_messages()
        self._greeting_wav: str = self._presynth_wav("こんにちは！何かご質問はありますか？")
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
        self._bubble_font_main = self._load_jp_font(14, weight=6)
        self._listen_font = self._load_jp_font(32)
        self._label_font = self._load_jp_font(22)
        self._subtitle_font = self._load_jp_font(30, weight=7)

        self._congestion: dict = {}  # 混雑状況キャッシュ
        self._start_firebase_poller()

    # ── ユーティリティ ────────────────────────────────────────

    def _ts(self, label: str) -> None:
        """タイムスタンプ付きでログ出力"""
        t = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        print(f"  [{t}] {label}")

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
        png_path = os.path.join(os.path.dirname(__file__), "案内図", "館内図（吹き出しなし）.png")
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
                greeting = self._presynth_wav("こんにちは！何かご質問はありますか？")
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
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if self._stop_event.is_set() or interrupted.is_set():
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)
        self._subtitle = ""
        os.unlink(tmp_path)
        return not interrupted.is_set()

    # ── 音声認識 ──────────────────────────────────────────────

    def listen(self) -> tuple[str | None, bool]:
        """(認識テキスト or None, タイムアウトか) を返す"""
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = MIC_THRESHOLD
        recognizer.dynamic_energy_threshold = False

        self._is_listening = True
        try:
            with sr.Microphone() as source:
                self._ts("マイク待機 開始")
                try:
                    audio = recognizer.listen(
                        source, timeout=LISTEN_TIMEOUT, phrase_time_limit=10
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

        contents = list(history)
        contents.append({"role": "user", "parts": [{"text": user_input}]})

        try:
            self._ts("Gemini 開始")
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=full_system,
                    response_mime_type="application/json"
                ),
            )
            self._ts("Gemini 完了")
            raw = response.text
            data = json.loads(raw)
            exhibit_key = data.get("exhibit") or None
            exhibit = EXHIBIT_KEY_MAP.get(exhibit_key) if exhibit_key else None
            return data.get("speech", ""), exhibit, data.get("continue", True)
        except json.JSONDecodeError:
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
        queue_exhibits = set(EXHIBIT_KEY_MAP[k] for k in ("truck","room","switch","arm","chess","soccer","pong","shooting","tank") if k in EXHIBIT_KEY_MAP)

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
        pad_x, pad_y = 7, 5
        arrow_w = 8
        arrow_h = 8
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
        # マップをリセット（ハイライトなし）、動画再生開始
        new_map = self._pil_to_surface(self.map_images[0])
        with self._surface_lock:
            self._map_surface = new_map
            self._right_surface = None
            self._highlight_loc = None
        self._start_video_player()

        msg_index = 0
        while not self._stop_event.is_set():
            idx = msg_index % len(self._standby_wavs)
            msg_index += 1
            print(f"  [呼び込み] {STANDBY_MESSAGES[idx]}")
            # 合成済みWAVを直接再生（合成処理なし）
            pygame.mixer.music.load(self._standby_wavs[idx])
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    return
                time.sleep(0.05)
            interval = random.uniform(5, 7)
            for _ in range(int(interval * 10)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

    def _interaction_loop(self) -> None:
        print("[対話モード] 開始")
        # 右パネルをクリア（動画停止後）
        with self._surface_lock:
            self._right_surface = None
        self._subtitle = "こんにちは！何かご質問はありますか？"
        pygame.mixer.music.load(self._greeting_wav)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if self._stop_event.is_set():
                pygame.mixer.music.stop()
                self._subtitle = ""
                return
            time.sleep(0.05)
        self._subtitle = ""
        history: list = []  # 対話モード中の会話履歴（モード終了で破棄）
        while not self._stop_event.is_set():
            user_input, timed_out = self.listen()
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
                    if self._stop_event.is_set():
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
                self._thinking_channel.play(self._thinking_loop_sound, loops=-1)
                while not future.done():
                    if self._stop_event.is_set():
                        self._thinking_channel.stop()
                        return
                    time.sleep(0.05)
                self._thinking_channel.stop()
                wav_path, exhibit, speech, photo_pil, should_continue = future.result()

            # 会話履歴に追記（次回のGemini呼び出しに引き継ぐ）
            history.append({"role": "user",  "parts": [{"text": user_input}]})
            history.append({"role": "model", "parts": [{"text": json.dumps({"speech": speech, "exhibit": exhibit}, ensure_ascii=False)}]})

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
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
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

        # 動画フレームをメインスレッドでサーフェスに変換
        try:
            frame_rgb = self._video_frame_queue.get_nowait()
            with self._surface_lock:
                self._right_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        except queue.Empty:
            pass

        with self._surface_lock:
            map_surf = self._map_surface
            right_surf = self._right_surface

        self.screen.fill((20, 20, 20))
        half_w = sw // 2

        # 左半分: マップ（常に画面の半分）
        scale_m = min(half_w / map_surf.get_width(), content_h / map_surf.get_height())
        mw = int(map_surf.get_width() * scale_m)
        mh = int(map_surf.get_height() * scale_m)
        map_x = half_w // 2 - mw // 2
        map_y = content_h // 2 - mh // 2
        scaled_map = pygame.transform.scale(map_surf, (mw, mh))
        self.screen.blit(scaled_map, (map_x, map_y))

        # 吹き出し（待ち時間 / 整理券不要）
        self._draw_congestion_bubbles(map_x, map_y, mw, mh)

        # 右半分: 動画 or 写真
        if right_surf is not None:
            scale_r = min(half_w / right_surf.get_width(), content_h / right_surf.get_height())
            rw = int(right_surf.get_width() * scale_r)
            rh = int(right_surf.get_height() * scale_r)
            scaled_right = pygame.transform.scale(right_surf, (rw, rh))
            self.screen.blit(scaled_right, (half_w + half_w // 2 - rw // 2, content_h // 2 - rh // 2))

        # パネルラベル（右パネルの上に重ねて表示）
        if self.state == RobotState.STANDBY:
            for text, cx in [("フロアマップ", half_w // 2), ("企画紹介動画", half_w + half_w // 2)]:
                surf = self._label_font.render(text, True, (255, 255, 255))
                tw, th = surf.get_size()
                pad_x, pad_y = 12, 5
                bg = pygame.Surface((tw + pad_x * 2, th + pad_y * 2), pygame.SRCALPHA)
                bg.fill((0, 0, 0, 160))
                bx = cx - (tw + pad_x * 2) // 2
                self.screen.blit(bg, (bx, 6))
                self.screen.blit(surf, (bx + pad_x, 6 + pad_y))

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

        # 字幕（対話モード中の発話内容）
        if self.state == RobotState.INTERACTION and self._subtitle:
            # 40文字で折り返し
            chars = self._subtitle
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
            pygame.display.flip()
            clock.tick(30)

        self._stop_event.set()
        pygame.quit()


if __name__ == "__main__":
    EntranceRobot().run()
