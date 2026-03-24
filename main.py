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

VOICEVOX_SPEAKER = 8        # 1: ずんだもん(ノーマル), 3: ずんだもん(あまあま), 8: 春日部つむぎ
BARGE_IN_THRESHOLD = 10000  # バージイン検知の閾値
BARGE_IN_DELAY = 0.5        # 再生開始後、この秒数は監視しない（エコー対策）
_VOICEVOX_DIR = Path(__file__).parent / "voicevox_core"
_FIREBASE_CREDENTIAL = Path(__file__).parent / "firebase_admin.json"

# 企画key → 企画名マッピング（Geminiの出力・Firebase両方で使用）
EXHIBIT_KEY_MAP = {
    # 3階
    "soccer":   "ロボットサッカー",
    "chess":    "ロボットチェス",
    "arm":      "ワームホールロボットアーム",
    "switch":   "せいみつスイッチ",
    "dress":    "AI着替えカメラ",
    "shooting": "お絵描きシューティング",
    "media":    "メディアアート",
    "space":    "自己投影空間",
    "truck":    "トロッコVR",
    # 1階
    "pendulum": "スーパー倒立振子",
    "tank":     "ARタンク",
    "pong":     "せいみつPONG",
    "balloon":  "バルーンロボット",
    "connect4": "立体四目並べ",
    "handwrite":"筆談ロボット",
    "janken":   "じゃんけんAI",
    "reception":"受付ロボット",
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
EXHIBIT_LOCATIONS = {
    # 3階
    "ロボットサッカー":           {"x": 0.13, "y": 0.14},
    "ロボットチェス":             {"x": 0.27, "y": 0.14},
    "ワームホールロボットアーム":   {"x": 0.40, "y": 0.14},
    "せいみつスイッチ":           {"x": 0.52, "y": 0.14},
    "AI着替えカメラ":             {"x": 0.62, "y": 0.13},
    "お絵描きシューティング":      {"x": 0.7, "y": 0.13},
    "メディアアート":             {"x": 0.13, "y": 0.36},
    "自己投影空間":               {"x": 0.64, "y": 0.3},
    "トロッコVR":                 {"x": 0.64, "y": 0.39},
    # 1階
    "スーパー倒立振子":           {"x": 0.27, "y": 0.7},
    "ARタンク":                  {"x": 0.35, "y": 0.73},
    "せいみつPONG":              {"x": 0.35, "y": 0.81},
    "バルーンロボット":           {"x": 0.15, "y": 0.82},
    "立体四目並べ":               {"x": 0.45, "y": 0.82},
    "筆談ロボット":               {"x": 0.55, "y": 0.76},
    "じゃんけんAI":              {"x": 0.78, "y": 0.7},
    "受付ロボット":               {"x": 0.9, "y": 0.7},
}



# ── TTS エンジン ──────────────────────────────────────────────────────────────

class TTSEngine:
    def synthesize(self, text: str) -> str:
        """テキストを合成してWAVファイルパスを返す"""
        raise NotImplementedError
    def close(self): pass


class VoicevoxTTS(TTSEngine):
    def __init__(self):
        from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
        import warnings; warnings.filterwarnings("ignore")
        print("  [VOICEVOX] 初期化中...")
        ort_path = _VOICEVOX_DIR / "onnxruntime" / "lib" / Onnxruntime.LIB_VERSIONED_FILENAME
        ort = Onnxruntime.load_once(filename=str(ort_path))
        dict_path = _VOICEVOX_DIR / "dict" / "open_jtalk_dic_utf_8-1.11"
        self._synth = Synthesizer(
            ort, OpenJtalk(dict_path),
            acceleration_mode="CPU",
            cpu_num_threads=max(multiprocessing.cpu_count() // 2, 1),
        )
        vvm_path = _VOICEVOX_DIR / "models" / "vvms" / "0.vvm"
        with VoiceModelFile.open(vvm_path) as model:
            self._synth.load_voice_model(model)
        print("  [VOICEVOX] 初期化完了")

    def synthesize(self, text: str) -> str:
        query = self._synth.create_audio_query(text, VOICEVOX_SPEAKER)
        wav = self._synth.synthesis(query, VOICEVOX_SPEAKER)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(wav); f.close()
        return f.name


class SayTTS(TTSEngine):
    """macOS 組み込み say コマンド（オフライン・低遅延）"""
    VOICE = "Kyoko"  # 日本語音声

    def synthesize(self, text: str) -> str:
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.close()
        subprocess.run(
            ["say", "-v", self.VOICE, "-o", f.name,
             "--file-format=WAVE", "--data-format=LEI16@24000", text],
            check=True
        )
        return f.name


class GttsTTS(TTSEngine):
    """gTTS (Google翻訳TTS) — 無料・要インターネット"""
    def synthesize(self, text: str) -> str:
        from gtts import gTTS
        mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        mp3.close()
        gTTS(text, lang="ja").save(mp3.name)
        # afconvert (macOS) で WAV に変換
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", mp3.name, wav.name],
            check=True
        )
        os.unlink(mp3.name)
        return wav.name


class GoogleCloudTTS(TTSEngine):
    """Google Cloud Text-to-Speech — 高品質・有料"""
    def __init__(self):
        from google.cloud import texttospeech
        self._client = texttospeech.TextToSpeechClient()
        self._voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP", name="ja-JP-Neural2-B"
        )
        self._audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=24000,
        )
        print("  [Google Cloud TTS] 初期化完了")

    def synthesize(self, text: str) -> str:
        from google.cloud import texttospeech
        response = self._client.synthesize_speech(
            input=texttospeech.SynthesisInput(text=text),
            voice=self._voice,
            audio_config=self._audio_config,
        )
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(response.audio_content); f.close()
        return f.name


class OpenAITTS(TTSEngine):
    """OpenAI TTS — 高品質・有料"""
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        print("  [OpenAI TTS] 初期化完了")

    def synthesize(self, text: str) -> str:
        mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        mp3.close()
        response = self._client.audio.speech.create(
            model="tts-1", voice="nova", input=text
        )
        response.stream_to_file(mp3.name)
        wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav.close()
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@24000", mp3.name, wav.name],
            check=True
        )
        os.unlink(mp3.name)
        return wav.name


def create_tts_engine(name: str) -> TTSEngine:
    engines = {
        "voicevox": VoicevoxTTS,
        "say":      SayTTS,
        "gtts":     GttsTTS,
        "google":   GoogleCloudTTS,
        "openai":   OpenAITTS,
    }
    cls = engines.get(name)
    if cls is None:
        raise ValueError(f"Unknown TTS engine: {name}")
    return cls()


class RobotState(Enum):
    STANDBY = "standby"
    INTERACTION = "interaction"


# ── ロボット本体 ───────────────────────────────────────────────────────────────

class EntranceRobot:
    def __init__(self, tts_name: str = "voicevox") -> None:
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

        self._tts: TTSEngine = create_tts_engine(tts_name)
        print(f"  [TTS] エンジン: {tts_name}")
        self._standby_wavs: list[str] = self._presynth_standby_messages()
        self._greeting_wav: str = self._presynth_wav("こんにちは！何かご質問はありますか？")
        self._thinking_wavs: list[str] = [self._presynth_wav(m) for m in THINKING_MESSAGES]
        self._thinking_channel = pygame.mixer.Channel(1)  # 専用チャンネル（GIL対策）
        self._thinking_loop_sound = self._make_thinking_loop_sound()  # 結合ループ音声
        self._farewell_wavs: list[str] = [self._presynth_wav(m) for m in FAREWELL_RESPONSES]

        self._mic_rms: float = 0.0
        self._is_listening: bool = False
        self._video_generation: int = 0
        self._video_frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._start_volume_monitor()
        self._bubble_font = self._load_jp_font(15)
        self._listen_font = self._load_jp_font(32)
        self._label_font = self._load_jp_font(22)

        self._congestion: dict = {}  # 混雑状況キャッシュ
        self._start_firebase_poller()

    # ── ユーティリティ ────────────────────────────────────────

    def _ts(self, label: str) -> None:
        """タイムスタンプ付きでログ出力"""
        t = time.strftime("%H:%M:%S") + f".{int(time.time() * 1000) % 1000:03d}"
        print(f"  [{t}] {label}")

    def _load_jp_font(self, size: int) -> pygame.font.Font:
        for path in (
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
            "/System/Library/Fonts/Hiragino Sans GB.ttc",
        ):
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
        pdf_path = os.path.join(os.path.dirname(__file__), "assets", "企画場所.pdf")
        doc = fitz.open(pdf_path)
        images = []
        for page in doc:
            pix = page.get_pixmap(dpi=150)
            img = PILImage.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        print(f"  [マップ] {len(images)}ページ読み込み完了")
        return images

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

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if self._stop_event.is_set() or interrupted.is_set():
                pygame.mixer.music.stop()
                break
            time.sleep(0.05)
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
                model="gemini-2.5-flash",
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
        queue_exhibits = set(EXHIBIT_KEY_MAP[k] for k in ("soccer","chess","arm","switch","shooting","space","truck","tank","pong"))

        no_bubble_exhibits = {"受付ロボット"}

        for exhibit, loc in EXHIBIT_LOCATIONS.items():
            if exhibit in no_bubble_exhibits:
                continue
            cx = map_x + int(loc["x"] * mw)
            cy = map_y + int(loc["y"] * mh)

            if exhibit in self._congestion:
                info = self._congestion[exhibit]
                waiting = info["waiting"]
                minutes = info["minutes"]
                if waiting == 0:
                    text = "10分以内"
                    bg = (30, 180, 80)
                elif minutes <= 10:
                    text = f"約{minutes}分"
                    bg = (210, 170, 0)
                else:
                    text = f"約{minutes}分"
                    bg = (200, 60, 40)
            elif exhibit in queue_exhibits:
                continue  # Firebase接続前は表示しない
            else:
                text = "整理券不要"
                bg = (80, 100, 160)

            self._draw_bubble(text, cx, cy, bg)

    def _draw_bubble(self, text: str, cx: int, cy: int, bg: tuple) -> None:
        font = self._bubble_font
        text_surf = font.render(text, True, (255, 255, 255))
        tw, th = text_surf.get_size()
        pad_x, pad_y = 6, 3
        bw = tw + pad_x * 2
        bh = th + pad_y * 2
        arrow_h = 6

        bx = cx - bw // 2
        by = cy - bh - arrow_h - 2

        surf = pygame.Surface((bw, bh + arrow_h), pygame.SRCALPHA)
        pygame.draw.rect(surf, (*bg, 210), (0, 0, bw, bh), border_radius=5)
        pygame.draw.polygon(surf, (*bg, 210), [
            (bw // 2 - 5, bh),
            (bw // 2 + 5, bh),
            (bw // 2,     bh + arrow_h),
        ])
        self.screen.blit(surf, (bx, by))
        self.screen.blit(text_surf, (bx + pad_x, by + pad_y))

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
        pygame.mixer.music.load(self._greeting_wav)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if self._stop_event.is_set():
                pygame.mixer.music.stop()
                return
            time.sleep(0.05)
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
                wav = random.choice(self._farewell_wavs)
                pygame.mixer.music.load(wav)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self._stop_event.is_set():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.05)
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
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
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

            self._render()
            pygame.display.flip()
            clock.tick(30)

        self._stop_event.set()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="精密ラボ 受付ロボット")
    parser.add_argument(
        "--tts",
        choices=["voicevox", "say", "gtts", "google", "openai"],
        default="voicevox",
        help="音声合成エンジン (default: voicevox)",
    )
    args = parser.parse_args()
    EntranceRobot(tts_name=args.tts).run()
