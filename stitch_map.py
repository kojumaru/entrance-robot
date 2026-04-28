"""
見開きマップ結合ツール

右ページの位置を調整して、2ページを綺麗に1枚に結合する。

操作:
  矢印キー        : 右ページを移動 (±2px)
  Shift + 矢印    : 大きく移動 (±20px)
  R               : リセット（初期位置に戻す）
  Shift+S         : festival_map.png として保存
  Q / Esc         : 終了
"""

import os
import pygame
import sys
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = "/Users/yoshidakouji/Downloads/第98回五月祭公式パンフレット.pdf"
OUT_PATH = os.path.join(_HERE, "assets", "festival_map.png")

# ── ページ読み込み ────────────────────────────────────────────────
import fitz
doc = fitz.open(PDF_PATH)
def _load_page(idx: int, dpi: int = 150) -> pygame.Surface:
    pix = doc[idx].get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return pygame.image.fromstring(img.tobytes(), img.size, "RGB"), img

surf_l, pil_l = _load_page(3)  # ページ4（0-indexed: 3）左
surf_r, pil_r = _load_page(4)  # ページ5（0-indexed: 4）右

PW, PH = surf_l.get_size()  # 各ページのピクセルサイズ

# ── pygame 初期化 ────────────────────────────────────────────────
WIN_W = min(PW * 2, 1400)
WIN_H = min(PH + 80, 900)
SCALE = WIN_W / (PW * 2)  # 画面に収めるスケール

pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("見開きマップ結合ツール  |  矢印:移動  Shift+S:保存  R:リセット")
clock = pygame.time.Clock()

def load_jp_font(size):
    for path in (
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ):
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)

font = load_jp_font(16)

# 右ページのオフセット（左ページに対する相対位置、ピクセル単位・元解像度）
offset_x: int = PW   # デフォルト：左ページの右端に配置
offset_y: int = 0

def render():
    screen.fill((40, 40, 40))

    # キャンバスサイズ = 左ページ幅 + 右ページ幅（オフセット考慮）
    canvas_w = max(PW, offset_x + PW)
    canvas_h = max(PH, offset_y + PH)

    # スケール（画面高に収める）
    sc = min(WIN_W / canvas_w, (WIN_H - 60) / canvas_h)
    cw = int(canvas_w * sc)
    ch = int(canvas_h * sc)
    cx = (WIN_W - cw) // 2
    cy = (WIN_H - 60 - ch) // 2

    # 左ページ
    sl = pygame.transform.scale(surf_l, (int(PW * sc), int(PH * sc)))
    screen.blit(sl, (cx, cy))

    # 右ページ（半透明でオーバーレイ）
    sr = pygame.transform.scale(surf_r, (int(PW * sc), int(PH * sc)))
    rx = cx + int(offset_x * sc)
    ry = cy + int(offset_y * sc)
    screen.blit(sr, (rx, ry))

    # 繋ぎ目を示す縦線
    seam_x = cx + int(PW * sc)
    pygame.draw.line(screen, (255, 80, 80), (seam_x, cy), (seam_x, cy + ch), 1)

    # 情報パネル
    info_y = WIN_H - 55
    pygame.draw.rect(screen, (20, 20, 20), (0, info_y, WIN_W, 55))
    screen.blit(font.render(
        f"右ページ offset: x={offset_x}px  y={offset_y}px  "
        f"（左ページ幅={PW}px）  重なり={PW - offset_x}px",
        True, (200, 220, 255)
    ), (10, info_y + 6))
    screen.blit(font.render(
        "矢印:移動(2px)  Shift+矢印:移動(20px)  R:リセット  Shift+S:保存  Q:終了",
        True, (120, 120, 120)
    ), (10, info_y + 28))

    pygame.display.flip()

def save():
    """現在のオフセットで2ページを結合してPNG保存"""
    canvas_w = max(pil_l.width, offset_x + pil_r.width)
    canvas_h = max(pil_l.height, offset_y + pil_r.height)
    out = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    out.paste(pil_l, (0, 0))
    out.paste(pil_r, (offset_x, offset_y))
    out.save(OUT_PATH)
    print(f"保存: {OUT_PATH}  ({canvas_w}x{canvas_h})")
    return True

status = ""
status_timer = 0

while True:
    dt = clock.tick(30)
    status_timer = max(0, status_timer - dt)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit(); sys.exit()

        elif event.type == pygame.KEYDOWN:
            mods = pygame.key.get_mods()
            step = 20 if (mods & pygame.KMOD_SHIFT) else 2

            if event.key == pygame.K_LEFT:    offset_x -= step
            elif event.key == pygame.K_RIGHT: offset_x += step
            elif event.key == pygame.K_UP:    offset_y -= step
            elif event.key == pygame.K_DOWN:  offset_y += step
            elif event.key == pygame.K_r:
                offset_x, offset_y = PW, 0
            elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                pygame.quit(); sys.exit()
            elif event.key == pygame.K_s and (mods & pygame.KMOD_SHIFT):
                if save():
                    status = "保存しました！"
                    status_timer = 2500

    render()
