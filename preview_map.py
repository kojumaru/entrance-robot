"""
インタラクティブな座標エディタ

操作:
  企画キー入力 + Enter : 企画を選択（例: jungle, chess, pong ...）
  Tab                  : 丸 / 吹き出し モード切替
  矢印キー             : 選択中の位置を微移動 (±0.005)
  Shift + 矢印         : 大きく移動 (±0.02)
  S                    : main.py の EXHIBIT_LOCATIONS を上書き保存
  Escape               : 入力クリア
"""

import ast
import re
import os
import math
import copy
import pygame
import sys

# ── main.py からデータ抽出 ────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_MAIN = os.path.join(_HERE, "main.py")

with open(_MAIN, encoding="utf-8") as f:
    _src = f.read()

def _extract(name: str) -> dict:
    m = re.search(rf"{name}\s*=\s*(\{{.*?\}})\n", _src, re.DOTALL)
    if not m:
        raise RuntimeError(f"{name} が main.py から見つかりません")
    return ast.literal_eval(m.group(1))

EXHIBIT_KEY_MAP: dict  = _extract("EXHIBIT_KEY_MAP")
_raw_locations          = _extract("EXHIBIT_LOCATIONS")

# 編集用コピー（float で持つ）
locations: dict = copy.deepcopy(_raw_locations)

# ── 定数 ─────────────────────────────────────────────────────────
WIN_W, WIN_H = 1300, 820
INFO_H       = 140          # 下部情報パネルの高さ
MAP_H        = WIN_H - INFO_H
STEP_FINE    = 0.003
STEP_COARSE  = 0.015
MAP_IMG_PATH = os.path.join(_HERE, "案内図", "館内図（吹き出しなし）.png")

# ── pygame 初期化 ─────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("座標エディタ  |  S: 保存  Tab: モード切替  矢印: 移動")
clock = pygame.time.Clock()

def load_jp_font(size: int) -> pygame.font.Font:
    for path in (
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ):
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)

font_sm  = load_jp_font(14)
font_md  = load_jp_font(18)
font_lg  = load_jp_font(22)

# マップ画像をロード
_pil_img = pygame.image.load(MAP_IMG_PATH)

# ── 状態 ─────────────────────────────────────────────────────────
selected_key: str | None = None   # EXHIBIT_KEY_MAP のキー
mode: str = "circle"               # "circle" or "bubble"
input_buf: str = ""
status_msg: str = ""               # 一時メッセージ

# ── 描画ヘルパー ──────────────────────────────────────────────────

def draw_bubble(surface: pygame.Surface, text: str, cx: int, cy: int,
                bg: tuple, font: pygame.font.Font) -> None:
    """吹き出しを描画（矢印の先端が cx,cy に来るよう配置）"""
    text_surf = font.render(text, True, (255, 255, 255))
    tw, th = text_surf.get_size()
    pad_x, pad_y = 7, 4
    bw = tw + pad_x * 2
    bh = th + pad_y * 2
    arrow_h = 8

    bx = cx - bw // 2
    by = cy - bh - arrow_h - 2

    surf = pygame.Surface((bw, bh + arrow_h), pygame.SRCALPHA)
    pygame.draw.rect(surf, (*bg, 220), (0, 0, bw, bh), border_radius=6)
    pygame.draw.polygon(surf, (*bg, 220), [
        (bw // 2 - 6, bh),
        (bw // 2 + 6, bh),
        (bw // 2,     bh + arrow_h),
    ])
    surface.blit(surf, (bx, by))
    surface.blit(text_surf, (bx + pad_x, by + pad_y))


def draw_circle_marker(surface: pygame.Surface, cx: int, cy: int,
                        r: int, color: tuple, selected: bool) -> None:
    if selected:
        pygame.draw.circle(surface, (255, 255, 100), (cx, cy), r + 6, 3)
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), r + 3, 4)
    pygame.draw.circle(surface, color, (cx, cy), r, 5)


def draw_bubble_marker(surface: pygame.Surface, bx: int, by: int,
                        selected: bool) -> None:
    """吹き出し先端位置を示す小さなひし形"""
    size = 7
    pts = [(bx, by - size), (bx + size, by), (bx, by + size), (bx - size, by)]
    color = (255, 240, 50) if selected else (100, 180, 255)
    pygame.draw.polygon(surface, color, pts)
    pygame.draw.polygon(surface, (0, 0, 0), pts, 1)


def render_map(mw: int, mh: int, map_x: int, map_y: int) -> pygame.Surface:
    scaled = pygame.transform.scale(_pil_img, (mw, mh))
    surf = pygame.Surface((WIN_W, MAP_H))
    surf.fill((30, 30, 30))
    surf.blit(scaled, (map_x, map_y))

    sel_name = EXHIBIT_KEY_MAP.get(selected_key) if selected_key else None

    for name, loc in locations.items():
        cx = map_x + int(loc["x"] * mw)
        cy = map_y + int(loc["y"] * mh)
        bx = map_x + int(loc.get("bx", loc["x"]) * mw)
        by = map_y + int(loc.get("by", loc["y"]) * mh)

        is_sel = (name == sel_name)
        circle_color = (220, 50, 50)

        # 吹き出し
        if is_sel and mode == "bubble":
            bubble_bg = (30, 120, 220)
        else:
            bubble_bg = (180, 60, 60)
        draw_bubble(surf, name, bx, by, bubble_bg, font_sm)

        # 吹き出し先端マーカー
        draw_bubble_marker(surf, bx, by, is_sel and mode == "bubble")

        # 丸
        draw_circle_marker(surf, cx, cy, 14, circle_color, is_sel and mode == "circle")

    return surf


def render_info(mw: int, mh: int, map_x: int, map_y: int) -> pygame.Surface:
    surf = pygame.Surface((WIN_W, INFO_H))
    surf.fill((20, 20, 20))

    sel_name = EXHIBIT_KEY_MAP.get(selected_key) if selected_key else None

    # 企画キー一覧（小さく）
    keys_text = "  ".join(EXHIBIT_KEY_MAP.keys())
    surf.blit(font_sm.render(f"キー: {keys_text}", True, (120, 120, 120)), (10, 6))

    # 選択中情報
    if sel_name and sel_name in locations:
        loc = locations[sel_name]
        cx, cy = loc["x"], loc["y"]
        bx = loc.get("bx", loc["x"])
        by = loc.get("by", loc["y"])

        mode_color_c = (255, 230, 50) if mode == "circle" else (160, 160, 160)
        mode_color_b = (255, 230, 50) if mode == "bubble" else (160, 160, 160)

        surf.blit(font_lg.render(f"選択: {selected_key}  →  {sel_name}", True, (255, 255, 255)), (10, 28))
        surf.blit(font_md.render(f"丸   x={cx:.4f}  y={cy:.4f}", True, mode_color_c), (10, 58))
        surf.blit(font_md.render(f"吹出 bx={bx:.4f} by={by:.4f}", True, mode_color_b), (10, 82))
        mode_label = "[ 丸を移動中 ]" if mode == "circle" else "[ 吹き出しを移動中 ]"
        surf.blit(font_md.render(mode_label, True, (255, 230, 50)), (400, 58))
    else:
        surf.blit(font_lg.render("企画キーを入力して Enter", True, (180, 180, 180)), (10, 28))

    # 入力バッファ
    surf.blit(font_md.render(f"入力: {input_buf}_", True, (200, 220, 255)), (10, 108))

    # ステータス
    if status_msg:
        surf.blit(font_md.render(status_msg, True, (100, 255, 100)), (450, 108))

    # 操作ガイド
    guide = "矢印:移動  Shift+矢印:大移動  Tab:丸↔吹出  Shift+S:保存  Esc:入力クリア"
    surf.blit(font_sm.render(guide, True, (100, 100, 100)), (450, 28))

    return surf


def save_to_main() -> None:
    """EXHIBIT_LOCATIONS を main.py に上書き"""
    lines = ["# x,y  : ハイライト丸の中心（マップ全体を1.0とした相対座標）\n",
             "# bx,by : 吹き出し矢印の先端（省略時は x,y を使用）\n",
             "EXHIBIT_LOCATIONS = {\n"]
    sections = {"# 3階": [], "# 1階": []}
    for name, loc in locations.items():
        # 3階か1階か元のソースで判定
        section = "# 3階" if f'"{name}"' in _src.split("# 1階")[0] else "# 1階"
        sections[section].append((name, loc))

    for sec, items in sections.items():
        lines.append(f"    {sec}\n")
        for name, loc in items:
            x  = loc["x"]
            y  = loc["y"]
            bx = loc.get("bx", loc["x"])
            by = loc.get("by", loc["y"])
            lines.append(
                f'    "{name}":{" " * max(1, 24 - len(name))}'
                f'{{"x": {x:.4f}, "y": {y:.4f}, "bx": {bx:.4f}, "by": {by:.4f}}},\n'
            )
    lines.append("}\n")
    new_block = "".join(lines)

    new_src = re.sub(
        r"# x,y.*?EXHIBIT_LOCATIONS\s*=\s*\{.*?\}\n",
        new_block,
        _src,
        flags=re.DOTALL,
    )
    with open(_MAIN, "w", encoding="utf-8") as f:
        f.write(new_src)
    global status_msg
    status_msg = "保存しました！"


# ── メインループ ──────────────────────────────────────────────────
def main() -> None:
    global selected_key, mode, input_buf, status_msg

    status_timer = 0

    while True:
        dt = clock.tick(30)
        status_timer += dt
        if status_timer > 2500:
            status_msg = ""

        # マップ描画サイズ計算
        img_w, img_h = _pil_img.get_size()
        scale = min(WIN_W / img_w, MAP_H / img_h)
        mw = int(img_w * scale)
        mh = int(img_h * scale)
        map_x = (WIN_W - mw) // 2
        map_y = (MAP_H - mh) // 2

        # イベント処理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                step = STEP_COARSE if (mods & pygame.KMOD_SHIFT) else STEP_FINE

                # 矢印キー: 選択企画の座標を移動
                sel_name = EXHIBIT_KEY_MAP.get(selected_key) if selected_key else None
                if sel_name and sel_name in locations:
                    loc = locations[sel_name]
                    dx = dy = 0.0
                    if event.key == pygame.K_LEFT:  dx = -step
                    elif event.key == pygame.K_RIGHT: dx = step
                    elif event.key == pygame.K_UP:   dy = -step
                    elif event.key == pygame.K_DOWN:  dy = step

                    if dx or dy:
                        if mode == "circle":
                            loc["x"] = round(max(0.0, min(1.0, loc["x"] + dx)), 4)
                            loc["y"] = round(max(0.0, min(1.0, loc["y"] + dy)), 4)
                        else:
                            loc["bx"] = round(max(0.0, min(1.0, loc.get("bx", loc["x"]) + dx)), 4)
                            loc["by"] = round(max(0.0, min(1.0, loc.get("by", loc["y"]) + dy)), 4)
                        continue

                if event.key == pygame.K_TAB:
                    mode = "bubble" if mode == "circle" else "circle"

                elif event.key == pygame.K_RETURN:
                    key = input_buf.strip()
                    if key in EXHIBIT_KEY_MAP:
                        selected_key = key
                        status_msg = f"{key} を選択"
                        status_timer = 0
                    else:
                        status_msg = f"不明なキー: {key}"
                        status_timer = 0
                    input_buf = ""

                elif event.key == pygame.K_ESCAPE:
                    input_buf = ""
                    selected_key = None

                elif event.key == pygame.K_BACKSPACE:
                    input_buf = input_buf[:-1]

                elif event.key == pygame.K_s and (mods & pygame.KMOD_SHIFT):
                    save_to_main()
                    status_timer = 0

                elif event.unicode and event.unicode.isprintable():
                    input_buf += event.unicode

        # 描画
        screen.blit(render_map(mw, mh, map_x, map_y), (0, 0))
        screen.blit(render_info(mw, mh, map_x, map_y), (0, MAP_H))
        pygame.display.flip()


if __name__ == "__main__":
    main()
