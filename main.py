"""
精密ラボ. 案内マップ表示
  - Firebase の混雑状況をリアルタイム表示
  - 背景動画をループ再生
  - Arduino へ状態をシリアル送信（--serial 指定時）
"""

import json
import os
import random
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import argparse
import math
import pygame
import PIL.Image as PILImage
import PIL.ImageDraw as PILImageDraw
import PIL.ImageFont as PILImageFont
import serial as _serial

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

_FIREBASE_CREDENTIAL = Path(__file__).parent / "firebase_admin.json"

FIREBASE_POLL_INTERVAL = 30
QUEUE_BUFFER_MINUTES = 10

DISPLAY_WIDTH  = 1200
DISPLAY_HEIGHT = 700

# 企画key → 企画名マッピング
EXHIBIT_KEY_MAP = {
    "truck":    "ジャングル・スコープ",
    "space":    "現実拡張空間",
    "media":    "精密メディアアート",
    "balloon":  "バルーンロボット",
    "switch":   "せいみつスイッチ",
    "arm":      "ロボットアーム",
    "soccer":   "スーパーロボットサッカー",
    "ai_lab":   "AI精密ラボ",
    "connect4": "立体四目並べ",
    "pong":     "せいみつPONG!",
    "shooting": "お絵描きシューティング",
    "tank":     "ARタンク",
    "pendulum": "スーパー倒立振子",
    "dress":    "AI着せ替えカメラ",
    "janken":   "じゃんけんAI",
}

# 企画場所の座標（マップ画像全体を1.0とした相対座標）
EXHIBIT_LOCATIONS = {
    "ジャングル・スコープ":      {"x": 0.3930, "y": 0.0860, "bx": 0.3810, "by": 0.0160},
    "現実拡張空間":              {"x": 0.5390, "y": 0.0980, "bx": 0.5270, "by": 0.0190},
    "精密メディアアート":        {"x": 0.9210, "y": 0.0950, "bx": 0.9180, "by": 0.0100},
    "バルーンロボット":          {"x": 0.3240, "y": 0.3350, "bx": 0.3000, "by": 0.2820},
    "せいみつスイッチ":          {"x": 0.5240, "y": 0.3560, "bx": 0.5210, "by": 0.2790},
    "ロボットアーム":            {"x": 0.7080, "y": 0.3500, "bx": 0.6960, "by": 0.2730},
    "スーパーロボットサッカー":  {"x": 0.9100, "y": 0.3440, "bx": 0.8830, "by": 0.2760},
    "AI精密ラボ":                {"x": 0.3100, "y": 0.6260, "bx": 0.2950, "by": 0.5790},
    "立体四目並べ":              {"x": 0.4250, "y": 0.6290, "bx": 0.4280, "by": 0.5670},
    "せいみつPONG!":             {"x": 0.5580, "y": 0.6290, "bx": 0.5760, "by": 0.5760},
    "お絵描きシューティング":    {"x": 0.8560, "y": 0.6170, "bx": 0.8470, "by": 0.5550},
    "ARタンク":                  {"x": 0.5590, "y": 0.8500, "bx": 0.5440, "by": 0.7820},
    "スーパー倒立振子":          {"x": 0.7020, "y": 0.8290, "bx": 0.6900, "by": 0.7700},
    "AI着せ替えカメラ":          {"x": 0.8570, "y": 0.8380, "bx": 0.8780, "by": 0.7850},
    "じゃんけんAI":              {"x": 0.1360, "y": 0.8530, "bx": 0.1120, "by": 0.7820},
}


STANDBY_MESSAGES = [
    "精密ラボへようこそ！1階右と3階で工学のさまざまな企画を展示しています！",
]

STANDBY_INTERVAL_RANGE = (10, 15)  # 呼び込み間隔（秒）の最小・最大

# ── TTS (gTTS) ─────────────────────────────────────────────────────────────────

_WORD_FIXES = {
    "工学": "こう学",
}

def _synthesize_gtts(text: str) -> str:
    """テキストをgTTSで合成してWAVファイルパスを返す"""
    from gtts import gTTS
    for word, reading in _WORD_FIXES.items():
        text = text.replace(word, reading)
    mp3 = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    mp3.close()
    gTTS(text, lang="ja", tld="co.jp").save(mp3.name)
    wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    wav.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", mp3.name,
         "-filter:a", "atempo=1.1,asetrate=24000*1.5,aresample=24000",
         "-ac", "1", "-sample_fmt", "s16", wav.name],
        check=True, capture_output=True,
    )
    os.unlink(mp3.name)
    return wav.name


# ── ロボット本体 ───────────────────────────────────────────────────────────────

class EntranceRobot:
    def __init__(self, serial_port: str | None = None) -> None:
        self._stop_event = threading.Event()

        # ── Arduino シリアル接続 ──────────────────────────────────
        self._arduino: _serial.Serial | None = None
        self._is_speaking: bool = False
        if serial_port:
            try:
                self._arduino = _serial.Serial(serial_port, 115200, timeout=1)
                time.sleep(2)
                print(f"  [Arduino] 接続完了: {serial_port}")
                self._start_serial_sender()
                self._start_serial_reader()
            except Exception as e:
                print(f"  [Arduino] 接続失敗: {e}")
                self._arduino = None

        pygame.init()
        pygame.mixer.init()
        self.screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("精密ラボ. 案内マップ")

        self.map_images = self._load_map_images()
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
        logo_pil = PILImage.open(logo_path).convert("RGBA")
        self._logo_surf = pygame.image.fromstring(logo_pil.tobytes(), logo_pil.size, "RGBA").convert_alpha()

        self._surface_lock = threading.Lock()
        self._map_surface: pygame.Surface = self._pil_to_surface(self.map_images[0])

        noto_bold = "/Users/yoshidakouji/Library/Fonts/NotoSansCJKjp-Bold.otf"
        self._bubble_font_main = self._load_jp_font(18, weight=6)
        self._side_font       = pygame.font.Font(noto_bold, 42)
        self._side_font_mid   = pygame.font.Font(noto_bold, 32)
        self._side_font_small = pygame.font.Font(noto_bold, 30)
        self._label_font = self._load_jp_font(22)

        self._congestion: dict = {}
        self._start_firebase_poller()

        self._subtitle: str = ""

        # 呼び込み音声を事前合成してバックグラウンドで再生開始
        self._standby_wavs: list[str] = []
        threading.Thread(target=self._presynth_and_start_calling, daemon=True).start()

    # ── Arduino シリアル ─────────────────────────────────────────

    def _set_speaking(self, speaking: bool) -> None:
        self._is_speaking = speaking

    def _start_serial_sender(self) -> None:
        def _heartbeat():
            last_val = None
            while not self._stop_event.is_set():
                if self._arduino and self._arduino.is_open:
                    val = 0 if self._is_speaking else 1
                    try:
                        self._arduino.write(f"{val}\n".encode())
                        if val != last_val:
                            label = "発話中" if val == 0 else "非発話"
                            print(f"  [Arduino] 送信: {val} ({label})")
                            last_val = val
                    except Exception as e:
                        print(f"  [Arduino] 送信エラー: {e}")
                time.sleep(5)
        threading.Thread(target=_heartbeat, daemon=True).start()

    def _start_serial_reader(self) -> None:
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

    # ── ユーティリティ ────────────────────────────────────────────

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

    # ── マップ読み込み ────────────────────────────────────────────

    def _load_map_images(self) -> list:
        png_path = os.path.join(os.path.dirname(__file__), "案内図", "facility_map.png")
        img = PILImage.open(png_path).convert("RGB")
        print(f"  [マップ] PNG読み込み完了")
        return [img]

    # ── Firebase 混雑状況ポーリング ───────────────────────────────

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
                        time_per = data.get("timePerPerson", 3)
                        waiting = max(now_serving - current_number, 0)
                        estimated_minutes = waiting * time_per + QUEUE_BUFFER_MINUTES
                        result[exhibit] = {"waiting": waiting, "minutes": estimated_minutes}
                    self._congestion = result
                    print(f"  [Firebase] 混雑状況更新: {result}")
                except Exception as e:
                    print(f"  [Firebase エラー] {e}")
                time.sleep(FIREBASE_POLL_INTERVAL)

        threading.Thread(target=_poll, daemon=True).start()

    # ── 待機モード呼び込み ────────────────────────────────────────

    def _presynth_and_start_calling(self) -> None:
        print("  [事前合成] 呼び込みメッセージを合成中...")
        try:
            wavs = [_synthesize_gtts(msg) for msg in STANDBY_MESSAGES]
            self._standby_wavs = wavs
            print(f"  [事前合成] {len(wavs)}件完了")
        except Exception as e:
            print(f"  [事前合成] 失敗: {e}")
            return
        self._calling_loop()

    def _calling_loop(self) -> None:
        print("[呼び込み] ループ開始")
        msg_index = 0
        while not self._stop_event.is_set():
            idx = msg_index % len(self._standby_wavs)
            msg_index += 1
            print(f"  [呼び込み] {STANDBY_MESSAGES[idx]}")
            self._subtitle = STANDBY_MESSAGES[idx]
            try:
                pygame.mixer.music.load(self._standby_wavs[idx])
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    if self._stop_event.is_set():
                        pygame.mixer.music.stop()
                        self._subtitle = ""
                        return
                    time.sleep(0.05)
            except Exception as e:
                print(f"  [呼び込み] 再生エラー: {e}")
            self._subtitle = ""
            interval = random.uniform(*STANDBY_INTERVAL_RANGE)
            for _ in range(int(interval * 10)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

    # ── 吹き出し描画 ──────────────────────────────────────────────

    def _draw_congestion_bubbles(self, map_x: int, map_y: int, mw: int, mh: int) -> None:
        queue_exhibits = set(
            EXHIBIT_KEY_MAP[k] for k in
            ("truck", "space", "switch", "arm", "soccer", "pong", "shooting", "tank")
            if k in EXHIBIT_KEY_MAP
        )

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
                continue
            else:
                text = "整理券不要"
                color = (60, 90, 160)

            self._draw_bubble(text, bx, by, color)

    def _draw_bubble(self, text: str, cx: int, cy: int, color: tuple) -> None:
        stroke = 2
        pad_x, pad_y = 10, 8
        arrow_w, arrow_h = 10, 10
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

        pygame.draw.rect(surf, (*color, 220), (0, 0, outer_w, outer_h), border_radius=r_outer)
        pygame.draw.polygon(surf, (*color, 220), [
            (mid - arrow_w, outer_h - 2),
            (mid + arrow_w, outer_h - 2),
            (mid, total_h),
        ])
        pygame.draw.rect(surf, (255, 255, 255, 248), (stroke, stroke, inner_w, inner_h), border_radius=r_inner)
        pygame.draw.polygon(surf, (255, 255, 255, 248), [
            (mid - arrow_w + stroke, outer_h - stroke - 1),
            (mid + arrow_w - stroke, outer_h - stroke - 1),
            (mid, total_h - stroke),
        ])
        surf.blit(text_surf, (stroke + pad_x, stroke + pad_y))
        self.screen.blit(surf, (cx - outer_w // 2, cy - total_h))

    # ── 描画 ──────────────────────────────────────────────────────

    def _render(self) -> None:
        sw, sh = self.screen.get_size()

        with self._surface_lock:
            map_surf = self._map_surface

        self.screen.fill((35, 10, 20))

        top_h = 50
        avail_h = sh - top_h
        scale_m = min(sw / map_surf.get_width(), avail_h / map_surf.get_height())
        mw = int(map_surf.get_width() * scale_m)
        mh = int(map_surf.get_height() * scale_m)
        map_x = sw // 2 - mw // 2
        map_y = top_h + avail_h // 2 - mh // 2
        scaled_map = pygame.transform.smoothscale(map_surf, (mw, mh))
        self.screen.blit(scaled_map, (map_x, map_y))

        self._draw_congestion_bubbles(map_x, map_y, mw, mh)

        # 右上ロゴ
        margin_w = sw - (map_x + mw)
        logo_w = margin_w - 10
        logo_h = int(self._logo_surf.get_height() * logo_w / self._logo_surf.get_width())
        logo_scaled = pygame.transform.smoothscale(self._logo_surf, (logo_w, logo_h))
        logo_x = map_x + mw + (margin_w - logo_w) // 2
        self.screen.blit(logo_scaled, (logo_x, map_y))

        # サイドラベル（縦書き）
        label_cy = top_h + avail_h // 2
        char_gap = 2
        right_cx_center = map_x + mw + margin_w // 2
        right_col_l = right_cx_center - 22
        right_col_r = right_cx_center + 22
        char_h_small = self._side_font_small.get_height()

        def _render_vertical(items, cx, y_offset):
            surfs = []
            for text, vertical, font in items:
                if vertical:
                    surfs += [font.render(c, True, (255, 255, 255)) for c in text]
                else:
                    s = font.render(text, True, (255, 255, 255))
                    surfs.append(pygame.transform.rotozoom(s, -90, 1.0))
            total_h = sum(s.get_height() for s in surfs) + char_gap * (len(surfs) - 1)
            y = label_cy - total_h // 2 + y_offset
            for s in surfs:
                self.screen.blit(s, (cx - s.get_width() // 2, y))
                y += s.get_height() + char_gap

        _render_vertical([("←都市工学科企画", True, self._side_font_mid)], map_x // 2, -60)
        _render_vertical([("→精密ラボ", True, self._side_font)], right_col_r, 0)
        _render_vertical([
            ("（", False, self._side_font_small),
            ("精密工学科企画", True, self._side_font_small),
            ("）", False, self._side_font_small),
        ], right_col_l, char_h_small * 3)

        # 字幕（呼び込み中のみ表示）
        subtitle = self._subtitle
        if subtitle:
            self._draw_subtitle(subtitle, sw, sh)

    def _draw_subtitle(self, text: str, sw: int, sh: int) -> None:
        pad_x, pad_y = 20, 10
        font = self._bubble_font_main
        text_surf = font.render(text, True, (255, 255, 255))
        tw, th = text_surf.get_size()
        bg_w = tw + pad_x * 2
        bg_h = th + pad_y * 2
        bg_x = (sw - bg_w) // 2
        bg_y = sh - bg_h - 16
        bg_surf = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg_surf.fill((0, 0, 0, 180))
        self.screen.blit(bg_surf, (bg_x, bg_y))
        self.screen.blit(text_surf, (bg_x + pad_x, bg_y + pad_y))

    # ── 起動 ──────────────────────────────────────────────────────

    def run(self) -> None:
        print("=" * 40)
        print("  精密ラボ 案内マップ 起動")
        print("  ESC : 終了")
        print("=" * 40)

        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

            self._render()
            pygame.display.flip()
            clock.tick(30)

        self._stop_event.set()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="精密ラボ 案内マップ")
    parser.add_argument(
        "--serial",
        default=None,
        metavar="PORT",
        help="ArduinoのシリアルポートURL (例: /dev/cu.usbserial-110)",
    )
    args = parser.parse_args()
    EntranceRobot(serial_port=args.serial).run()
