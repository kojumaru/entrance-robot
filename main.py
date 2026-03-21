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

import fitz  # PyMuPDF
from google import genai
from google.genai import types
import PIL.Image as PILImage
import PIL.ImageDraw as PILImageDraw
import PIL.ImageFont as PILImageFont
import pygame
import speech_recognition as sr
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

# ── 定数 ──────────────────────────────────────────────────────────────────────

MIC_THRESHOLD = 500
LISTEN_TIMEOUT = 8

VOICEVOX_SPEAKER = 8        # 1: ずんだもん(ノーマル), 3: ずんだもん(あまあま), 8: 春日部つむぎ
BARGE_IN_THRESHOLD = 10000  # バージイン検知の閾値
BARGE_IN_DELAY = 0.5        # 再生開始後、この秒数は監視しない（エコー対策）
_VOICEVOX_DIR = Path(__file__).parent / "voicevox_core"
_FIREBASE_CREDENTIAL = Path(__file__).parent / "firebase_admin.json"

# FirebaseのドキュメントID → 企画名マッピング
FIREBASE_EXHIBIT_MAP = {
    "arm":      "ワームホールロボットアーム",
    "chess":    "ロボットチェス",
    "pong":     "せいみつPONG",
    "shooting": "お絵描きシューティング",
    "soccer":   "ロボットサッカー",
    "space":    "自己投影空間",
    "switch":   "せいみつスイッチ",
    "tank":     "ARタンク",
    "truck":    "トロッコVR",
}
FIREBASE_POLL_INTERVAL = 30  # 混雑状況の更新間隔（秒）

STANDBY_MESSAGES = [
    "精密ラボへようこそ！1階右と3階で工学のさまざまな企画を展示しています！",
    "こんにちは！精密ラボの展示をご案内します！",
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
    "お絵描きシューティング":      {"x": 0.72, "y": 0.14},
    "メディアアート":             {"x": 0.11, "y": 0.40},
    "自己投影空間":               {"x": 0.62, "y": 0.33},
    "トロッコVR":                 {"x": 0.62, "y": 0.43},
    # 1階
    "スーパー倒立振子":           {"x": 0.22, "y": 0.68},
    "ARタンク":                  {"x": 0.30, "y": 0.73},
    "せいみつPONG":              {"x": 0.29, "y": 0.81},
    "バルーンロボット":           {"x": 0.13, "y": 0.82},
    "立体四目並べ":               {"x": 0.41, "y": 0.82},
    "筆談ロボット":               {"x": 0.52, "y": 0.76},
    "じゃんけんAI":              {"x": 0.76, "y": 0.67},
    "受付ロボット":               {"x": 0.87, "y": 0.67},
}

# Geminiが誤変換した場合の正規名への変換マップ
EXHIBIT_ALIASES = {
    "精密スイッチ":   "せいみつスイッチ",
    "精密PONG":      "せいみつPONG",
    "精密ポン":      "せいみつPONG",
    "せいみつポン":  "せいみつPONG",
}


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

        self.synthesizer = self._init_voicevox()
        self._standby_wavs: list[str] = self._presynth_standby_messages()
        self._greeting_wav: str = self._presynth_wav("こんにちは！何かご質問はありますか？")
        self._thinking_wavs: list[str] = [self._presynth_wav(m) for m in THINKING_MESSAGES]

        self._mic_rms: float = 0.0
        self._video_generation: int = 0
        self._video_frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self._start_volume_monitor()

        self._congestion: dict = {}  # 混雑状況キャッシュ
        self._start_firebase_poller()

    # ── VOICEVOX 初期化 ───────────────────────────────────────

    def _init_voicevox(self) -> Synthesizer:
        import warnings; warnings.filterwarnings("ignore")
        print("  [VOICEVOX] 初期化中...")
        ort_path = _VOICEVOX_DIR / "onnxruntime" / "lib" / Onnxruntime.LIB_VERSIONED_FILENAME
        ort = Onnxruntime.load_once(filename=str(ort_path))
        dict_path = _VOICEVOX_DIR / "dict" / "open_jtalk_dic_utf_8-1.11"
        synthesizer = Synthesizer(
            ort,
            OpenJtalk(dict_path),
            acceleration_mode="CPU",
            cpu_num_threads=max(multiprocessing.cpu_count() // 2, 1),
        )
        vvm_path = _VOICEVOX_DIR / "models" / "vvms" / "0.vvm"
        with VoiceModelFile.open(vvm_path) as model:
            synthesizer.load_voice_model(model)
        print("  [VOICEVOX] 初期化完了")
        return synthesizer

    # ── ユーティリティ ────────────────────────────────────────

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
                        exhibit = FIREBASE_EXHIBIT_MAP.get(key)
                        if exhibit is None:
                            continue
                        data = doc.to_dict()
                        now_serving = data.get("nowServing", 0)
                        current_number = data.get("currentNumber", 0)
                        waiting = max(now_serving - current_number, 0)
                        result[exhibit] = waiting
                    self._congestion = result
                    print(f"  [Firebase] 混雑状況更新: {result}")
                except Exception as e:
                    print(f"  [Firebase エラー] {e}")
                time.sleep(FIREBASE_POLL_INTERVAL)

        threading.Thread(target=_poll, daemon=True).start()

    def _build_congestion_text(self) -> str:
        if not self._congestion:
            return ""
        lines = ["【現在の整理券待ち人数】"]
        for exhibit, waiting in self._congestion.items():
            lines.append(f"- {exhibit}: {waiting}人待ち")
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
        audio_query = self.synthesizer.create_audio_query(text, VOICEVOX_SPEAKER)
        wav = self.synthesizer.synthesis(audio_query, VOICEVOX_SPEAKER)
        f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        f.write(wav)
        f.close()
        return f.name

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
        speech, exhibit = self.get_response(user_input, history)
        wav_path = self._presynth_wav(speech)
        exhibit = EXHIBIT_ALIASES.get(exhibit, exhibit) if exhibit else None
        photo_pil = self._load_exhibit_photo(exhibit) if exhibit and EXHIBIT_LOCATIONS.get(exhibit) else None
        return wav_path, exhibit, speech, photo_pil

    def speak(self, text: str, interruptible: bool = False) -> bool:
        t_voicevox_start = time.time()
        audio_query = self.synthesizer.create_audio_query(text, VOICEVOX_SPEAKER)
        wav = self.synthesizer.synthesis(audio_query, VOICEVOX_SPEAKER)
        print(f"  [計測] VOICEVOX合成: {time.time() - t_voicevox_start:.2f}s ({len(text)}文字)")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav)
            tmp_path = f.name

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

        with sr.Microphone() as source:
            print("  [マイク待機中...]")
            try:
                t0 = time.time()
                audio = recognizer.listen(
                    source, timeout=LISTEN_TIMEOUT, phrase_time_limit=10
                )
                t_recorded = time.time()
                print(f"  [計測] 録音完了: {t_recorded - t0:.2f}s")
            except sr.WaitTimeoutError:
                return None, True  # タイムアウト

        try:
            t_stt_start = time.time()
            text = recognizer.recognize_google(audio, language="ja-JP")
            print(f"  [計測] Google STT: {time.time() - t_stt_start:.2f}s → 「{text}」")
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
            t_gemini_start = time.time()
            response = self.client.models.generate_content(
                model="gemini-2.5-flash",
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=full_system,
                    response_mime_type="application/json"
                ),
            )
            print(f"  [計測] Gemini: {time.time() - t_gemini_start:.2f}s")
            raw = response.text
            data = json.loads(raw)
            return data.get("speech", ""), data.get("exhibit") or None
        except json.JSONDecodeError:
            return raw, None
        except Exception as e:
            print(f"  [Gemini エラー] {e}")
            return "すみません、もう一度お願いします", None

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

            # Gemini + VOICEVOX合成をバックグラウンドで実行しながら考え中セリフを再生
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._get_response_wav, user_input, history)
                remaining = list(self._thinking_wavs)
                random.shuffle(remaining)
                while not future.done():
                    if not remaining:
                        remaining = list(self._thinking_wavs)
                        random.shuffle(remaining)
                    pygame.mixer.music.load(remaining.pop())
                    pygame.mixer.music.play()
                    time.sleep(0.1)  # play()直後はget_busy()がFalseになる場合があるため待機
                    while pygame.mixer.music.get_busy() and not future.done():
                        if self._stop_event.is_set():
                            pygame.mixer.music.stop()
                            return
                        time.sleep(0.05)
                    # セリフ間のインターバル（future完了なら即抜ける）
                    interval_steps = int(THINKING_INTERVAL / 0.05)
                    for _ in range(interval_steps):
                        if future.done() or self._stop_event.is_set():
                            break
                        time.sleep(0.05)
                while pygame.mixer.music.get_busy():
                    time.sleep(0.05)
                wav_path, exhibit, speech, photo_pil = future.result()

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
            t_speak_start = time.time()
            print(f"  [ロボット] {speech}")
            print(f"  [計測] 発話終了→紬が喋り始めるまでの合計: {t_speak_start - t_user_end:.2f}s")
            # 合成済みWAVを直接再生
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
            os.unlink(wav_path)

    # ── 状態遷移 ──────────────────────────────────────────────

    def _switch_to(self, new_state: RobotState) -> None:
        self._stop_event.set()
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

        # 右半分: 動画 or 写真
        if right_surf is not None:
            scale_r = min(half_w / right_surf.get_width(), content_h / right_surf.get_height())
            rw = int(right_surf.get_width() * scale_r)
            rh = int(right_surf.get_height() * scale_r)
            scaled_right = pygame.transform.scale(right_surf, (rw, rh))
            self.screen.blit(scaled_right, (half_w + half_w // 2 - rw // 2, content_h // 2 - rh // 2))

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
    EntranceRobot().run()
