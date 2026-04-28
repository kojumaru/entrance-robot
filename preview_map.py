"""
インタラクティブな座標エディタ

操作:
  M                    : マップ切替（精密ラボ ↔ 全体マップ）
  企画/建物キー + Enter : 対象を選択
  Tab                  : 丸 / 吹き出し モード切替（精密ラボモードのみ）
  矢印キー             : 位置を微移動 (±0.003)
  Shift + 矢印         : 大きく移動 (±0.015)
  Shift+S              : main.py に保存
  Escape               : 入力クリア・選択解除
"""

import ast
import re
import os
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

EXHIBIT_KEY_MAP: dict   = _extract("EXHIBIT_KEY_MAP")
BUILDING_NAME_MAP: dict = _extract("BUILDING_NAME_MAP")
exhibit_locs: dict      = copy.deepcopy(_extract("EXHIBIT_LOCATIONS"))
building_locs: dict     = copy.deepcopy(_extract("BUILDING_LOCATIONS"))

# ── 定数 ─────────────────────────────────────────────────────────
WIN_W, WIN_H = 1400, 860
INFO_H       = 140
MAP_H        = WIN_H - INFO_H
STEP_FINE    = 0.003
STEP_COARSE  = 0.015

SEIMITSU_MAP = os.path.join(_HERE, "案内図", "館内図（吹き出しなし）.png")
FESTIVAL_MAP = os.path.join(_HERE, "assets", "festival_map.png")

# ── pygame 初期化 ─────────────────────────────────────────────────
pygame.init()
screen = pygame.display.set_mode((WIN_W, WIN_H))
pygame.display.set_caption("座標エディタ  |  M:マップ切替  Shift+S:保存  矢印:移動")
clock = pygame.time.Clock()

def load_jp_font(size: int) -> pygame.font.Font:
    for path in (
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ):
        if os.path.exists(path):
            return pygame.font.Font(path, size)
    return pygame.font.SysFont(None, size)

font_sm = load_jp_font(14)
font_md = load_jp_font(18)
font_lg = load_jp_font(22)

img_seimitsu = pygame.image.load(SEIMITSU_MAP)
img_festival = pygame.image.load(FESTIVAL_MAP)

# ── 状態 ─────────────────────────────────────────────────────────
map_mode: str        = "festival"   # "seimitsu" or "festival"
selected_key: str | None = None
submode: str         = "circle"     # "circle" or "bubble"（精密ラボのみ）
input_buf: str       = ""
status_msg: str      = ""

# ── 描画ヘルパー ──────────────────────────────────────────────────

def draw_bubble(surface, text, cx, cy, bg, font):
    ts = font.render(text, True, (255, 255, 255))
    tw, th = ts.get_size()
    px, py = 7, 4
    bw, bh, ah = tw + px*2, th + py*2, 8
    s = pygame.Surface((bw, bh + ah), pygame.SRCALPHA)
    pygame.draw.rect(s, (*bg, 220), (0, 0, bw, bh), border_radius=6)
    pygame.draw.polygon(s, (*bg, 220), [(bw//2-6, bh),(bw//2+6, bh),(bw//2, bh+ah)])
    surface.blit(s, (cx - bw//2, cy - bh - ah - 2))
    surface.blit(ts, (cx - bw//2 + px, cy - bh - ah - 2 + py))

def draw_circle_marker(surface, cx, cy, r, color, selected):
    if selected:
        pygame.draw.circle(surface, (255, 255, 100), (cx, cy), r + 6, 3)
    pygame.draw.circle(surface, (255, 255, 255), (cx, cy), r + 3, 4)
    pygame.draw.circle(surface, color, (cx, cy), r, 5)

def draw_diamond(surface, bx, by, selected):
    sz = 7
    pts = [(bx, by-sz),(bx+sz, by),(bx, by+sz),(bx-sz, by)]
    pygame.draw.polygon(surface, (255,240,50) if selected else (100,180,255), pts)
    pygame.draw.polygon(surface, (0,0,0), pts, 1)


def render_seimitsu(mw, mh, mx, my) -> pygame.Surface:
    surf = pygame.Surface((WIN_W, MAP_H))
    surf.fill((30, 30, 30))
    surf.blit(pygame.transform.scale(img_seimitsu, (mw, mh)), (mx, my))

    sel_name = EXHIBIT_KEY_MAP.get(selected_key) if selected_key else None

    for name, loc in exhibit_locs.items():
        cx = mx + int(loc["x"] * mw)
        cy = my + int(loc["y"] * mh)
        bx = mx + int(loc.get("bx", loc["x"]) * mw)
        by = my + int(loc.get("by", loc["y"]) * mh)
        is_sel = (name == sel_name)

        bubble_bg = (30, 120, 220) if (is_sel and submode == "bubble") else (180, 60, 60)
        draw_bubble(surf, name, bx, by, bubble_bg, font_sm)
        draw_diamond(surf, bx, by, is_sel and submode == "bubble")
        draw_circle_marker(surf, cx, cy, 14, (220, 50, 50), is_sel and submode == "circle")

    return surf


def render_festival(mw, mh, mx, my) -> pygame.Surface:
    surf = pygame.Surface((WIN_W, MAP_H))
    surf.fill((30, 30, 30))
    surf.blit(pygame.transform.scale(img_festival, (mw, mh)), (mx, my))

    for key, loc in building_locs.items():
        cx = mx + int(loc["x"] * mw)
        cy = my + int(loc["y"] * mh)
        is_sel = (key == selected_key)
        name = BUILDING_NAME_MAP.get(key, key)

        draw_bubble(surf, name, cx, cy, (30, 120, 220) if is_sel else (60, 130, 60), font_sm)
        draw_circle_marker(surf, cx, cy, 12, (50, 160, 50), is_sel)

    return surf


def render_info() -> pygame.Surface:
    surf = pygame.Surface((WIN_W, INFO_H))
    surf.fill((20, 20, 20))

    mode_label = "[ 精密ラボマップ ]" if map_mode == "seimitsu" else "[ 全体キャンパスマップ ]"
    surf.blit(font_lg.render(mode_label, True, (255, 200, 50)), (10, 6))

    if map_mode == "seimitsu":
        keys_text = "  ".join(EXHIBIT_KEY_MAP.keys())
        surf.blit(font_sm.render(f"キー: {keys_text}", True, (100, 100, 100)), (300, 10))

        sel_name = EXHIBIT_KEY_MAP.get(selected_key) if selected_key else None
        if sel_name and sel_name in exhibit_locs:
            loc = exhibit_locs[sel_name]
            cx, cy = loc["x"], loc["y"]
            bx = loc.get("bx", loc["x"]); by = loc.get("by", loc["y"])
            cc = (255,230,50) if submode=="circle" else (160,160,160)
            bc = (255,230,50) if submode=="bubble" else (160,160,160)
            surf.blit(font_lg.render(f"選択: {selected_key} → {sel_name}", True, (255,255,255)), (10, 34))
            surf.blit(font_md.render(f"丸   x={cx:.4f}  y={cy:.4f}", True, cc), (10, 64))
            surf.blit(font_md.render(f"吹出 bx={bx:.4f} by={by:.4f}", True, bc), (10, 88))
            surf.blit(font_md.render("[ 丸を移動中 ]" if submode=="circle" else "[ 吹き出しを移動中 ]",
                                     True, (255,230,50)), (420, 64))
        else:
            surf.blit(font_lg.render("企画キーを入力して Enter", True, (180,180,180)), (10, 34))

    else:
        keys_text = "  ".join(BUILDING_NAME_MAP.keys())
        surf.blit(font_sm.render(f"キー: {keys_text}", True, (100, 100, 100)), (300, 10))

        if selected_key and selected_key in building_locs:
            loc = building_locs[selected_key]
            name = BUILDING_NAME_MAP.get(selected_key, selected_key)
            surf.blit(font_lg.render(f"選択: {selected_key} → {name}", True, (255,255,255)), (10, 34))
            surf.blit(font_md.render(f"x={loc['x']:.4f}  y={loc['y']:.4f}", True, (255,230,50)), (10, 64))
        else:
            surf.blit(font_lg.render("建物キーを入力して Enter", True, (180,180,180)), (10, 34))

    surf.blit(font_md.render(f"入力: {input_buf}_", True, (200,220,255)), (10, 112))
    if status_msg:
        surf.blit(font_md.render(status_msg, True, (100,255,100)), (500, 112))
    guide = "M:マップ切替  矢印:移動  Shift+矢印:大移動  Tab:丸↔吹出(精密のみ)  Shift+S:保存  Esc:クリア"
    surf.blit(font_sm.render(guide, True, (80,80,80)), (500, 34))

    return surf


def save_to_main() -> None:
    global status_msg, _src
    with open(_MAIN, encoding="utf-8") as f:
        src = f.read()

    # EXHIBIT_LOCATIONS を更新
    lines = ["# x,y  : ハイライト丸の中心（マップ全体を1.0とした相対座標）\n",
             "# bx,by : 吹き出し矢印の先端（省略時は x,y を使用）\n",
             "EXHIBIT_LOCATIONS = {\n"]
    sections = {"# 3階": [], "# 1階": []}
    for name, loc in exhibit_locs.items():
        sec = "# 3階" if f'"{name}"' in src.split("# 1階")[0] else "# 1階"
        sections[sec].append((name, loc))
    for sec, items in sections.items():
        lines.append(f"    {sec}\n")
        for name, loc in items:
            x=loc["x"]; y=loc["y"]
            bx=loc.get("bx",x); by=loc.get("by",y)
            pad = " " * max(1, 26 - len(name))
            lines.append(f'    "{name}":{pad}{{"x": {x:.4f}, "y": {y:.4f}, "bx": {bx:.4f}, "by": {by:.4f}}},\n')
    lines.append("}\n")
    src = re.sub(r"# x,y.*?EXHIBIT_LOCATIONS\s*=\s*\{.*?\}\n", "".join(lines), src, flags=re.DOTALL)

    # BUILDING_LOCATIONS を更新
    blines = ["BUILDING_LOCATIONS = {\n"]
    for key, loc in building_locs.items():
        name = BUILDING_NAME_MAP.get(key, key)
        pad = " " * max(1, 12 - len(key))
        blines.append(f'    "{key}":{pad}{{"x": {loc["x"]:.4f}, "y": {loc["y"]:.4f}}},  # {name}\n')
    blines.append("}\n")
    src = re.sub(r"BUILDING_LOCATIONS\s*=\s*\{.*?\}\n", "".join(blines), src, flags=re.DOTALL)

    with open(_MAIN, "w", encoding="utf-8") as f:
        f.write(src)
    _src = src
    status_msg = "保存しました！"


# ── メインループ ──────────────────────────────────────────────────
def main() -> None:
    global map_mode, selected_key, submode, input_buf, status_msg

    status_timer = 0
    hold_timer = 0  # 長押し用タイマー（ms）

    while True:
        dt = clock.tick(30)
        status_timer += dt
        if status_timer > 2500:
            status_msg = ""

        # 長押し連続移動（150ms後に開始、30fpsで毎フレーム）
        keys = pygame.key.get_pressed()
        mods = pygame.key.get_mods()
        any_arrow = keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]
        if any_arrow:
            hold_timer += dt
        else:
            hold_timer = 0

        if hold_timer > 150:
            step = STEP_COARSE if (mods & pygame.KMOD_SHIFT) else STEP_FINE
            dx = (keys[pygame.K_RIGHT] - keys[pygame.K_LEFT]) * step
            dy = (keys[pygame.K_DOWN]  - keys[pygame.K_UP])   * step
            if dx or dy:
                if map_mode == "seimitsu":
                    sel_name = EXHIBIT_KEY_MAP.get(selected_key) if selected_key else None
                    if sel_name and sel_name in exhibit_locs:
                        loc = exhibit_locs[sel_name]
                        if submode == "circle":
                            loc["x"] = round(max(0.0, min(1.0, loc["x"] + dx)), 4)
                            loc["y"] = round(max(0.0, min(1.0, loc["y"] + dy)), 4)
                        else:
                            loc["bx"] = round(max(0.0, min(1.0, loc.get("bx", loc["x"]) + dx)), 4)
                            loc["by"] = round(max(0.0, min(1.0, loc.get("by", loc["y"]) + dy)), 4)
                else:
                    if selected_key and selected_key in building_locs:
                        loc = building_locs[selected_key]
                        loc["x"] = round(max(0.0, min(1.0, loc["x"] + dx)), 4)
                        loc["y"] = round(max(0.0, min(1.0, loc["y"] + dy)), 4)

        img = img_seimitsu if map_mode == "seimitsu" else img_festival
        img_w, img_h = img.get_size()
        scale = min(WIN_W / img_w, MAP_H / img_h)
        mw = int(img_w * scale)
        mh = int(img_h * scale)
        mx = (WIN_W - mw) // 2
        my = (MAP_H - mh) // 2

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

            elif event.type == pygame.KEYDOWN:
                mods = pygame.key.get_mods()
                step = STEP_COARSE if (mods & pygame.KMOD_SHIFT) else STEP_FINE

                dx = dy = 0.0
                if event.key == pygame.K_LEFT:    dx = -step
                elif event.key == pygame.K_RIGHT: dx =  step
                elif event.key == pygame.K_UP:    dy = -step
                elif event.key == pygame.K_DOWN:  dy =  step

                if dx or dy:
                    if map_mode == "seimitsu":
                        sel_name = EXHIBIT_KEY_MAP.get(selected_key) if selected_key else None
                        if sel_name and sel_name in exhibit_locs:
                            loc = exhibit_locs[sel_name]
                            if submode == "circle":
                                loc["x"] = round(max(0.0, min(1.0, loc["x"] + dx)), 4)
                                loc["y"] = round(max(0.0, min(1.0, loc["y"] + dy)), 4)
                            else:
                                loc["bx"] = round(max(0.0, min(1.0, loc.get("bx", loc["x"]) + dx)), 4)
                                loc["by"] = round(max(0.0, min(1.0, loc.get("by", loc["y"]) + dy)), 4)
                    else:
                        if selected_key and selected_key in building_locs:
                            loc = building_locs[selected_key]
                            loc["x"] = round(max(0.0, min(1.0, loc["x"] + dx)), 4)
                            loc["y"] = round(max(0.0, min(1.0, loc["y"] + dy)), 4)
                    continue

                if event.key == pygame.K_m:
                    map_mode = "festival" if map_mode == "seimitsu" else "seimitsu"
                    selected_key = None; input_buf = ""
                    status_msg = "全体マップ" if map_mode == "festival" else "精密ラボマップ"
                    status_timer = 0

                elif event.key == pygame.K_TAB and map_mode == "seimitsu":
                    submode = "bubble" if submode == "circle" else "circle"

                elif event.key == pygame.K_RETURN:
                    key = input_buf.strip()
                    if map_mode == "seimitsu" and key in EXHIBIT_KEY_MAP:
                        selected_key = key
                        status_msg = f"{key} を選択"
                    elif map_mode == "festival" and key in BUILDING_NAME_MAP:
                        selected_key = key
                        status_msg = f"{key} を選択"
                    else:
                        status_msg = f"不明なキー: {key}"
                    status_timer = 0
                    input_buf = ""

                elif event.key == pygame.K_ESCAPE:
                    input_buf = ""; selected_key = None

                elif event.key == pygame.K_BACKSPACE:
                    input_buf = input_buf[:-1]

                elif event.key == pygame.K_s and (mods & pygame.KMOD_SHIFT):
                    save_to_main(); status_timer = 0

                elif event.unicode and event.unicode.isprintable():
                    input_buf += event.unicode

        if map_mode == "seimitsu":
            screen.blit(render_seimitsu(mw, mh, mx, my), (0, 0))
        else:
            screen.blit(render_festival(mw, mh, mx, my), (0, 0))
        screen.blit(render_info(), (0, MAP_H))
        pygame.display.flip()


if __name__ == "__main__":
    main()
