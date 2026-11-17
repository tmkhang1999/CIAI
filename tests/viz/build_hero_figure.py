#!/usr/bin/env python3
"""Hero figure for the project page and README: a five-column cut of ciai-qualitative.jpg.

The full grid (held-out MID scene everett_lobby3, three flash directions, seven methods)
is too dense to read at the top of a page. This keeps Input, GT albedo, Ours and the two
baselines whose albedo changes most with the light on this scene, ranked by the
per-scene C_mat in documents/results/mid_per_scene.json. The rule is applied here in
code, so the choice is not made by eye.

The model weights are not available any more, so the panels are cut from the saved grid
rather than re-rendered. Panel geometry was measured on the 1900 px grid.

Usage:
  python tests/viz/build_hero_figure.py
"""
from __future__ import annotations

import json
import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'docs', 'static', 'img', 'ciai-qualitative.jpg')
PER_SCENE = os.path.join(ROOT, 'documents', 'results', 'mid_per_scene.json')
OUTS = [os.path.join(ROOT, 'docs', 'static', 'img', 'ciai-hero.jpg'),
        os.path.join(ROOT, 'documents', 'thesis', 'images', 'readme', 'ciai-hero.jpg')]

SCENE = 'everett_lobby3'
# Column order of the source grid, as drawn by build_all_models_light_matrix.py.
COLUMNS = ['Input', 'GT albedo', 'Ours', 'CRefNet', 'Marigold-App', 'Marigold-Light',
           'Ordinal Shading', 'CD-IID', 'RGB-X']
OURS_KEY = 'Ours (full)'
ROWS = ['light 0', 'light 9', 'light 18']
# Measured on the 1900x472 grid: first panel at x=108, pitch 199.2, panels 197x132.
X0, PITCH, PW = 108, 199.2, 197
ROW_Y, PH = (58, 192, 326), 132
SCALE, GAP, N_BASE = 2, 8, 2


def font(size: int, bold: bool = False):
    for n in (['DejaVuSans-Bold.ttf', 'Arial Bold.ttf'] if bold else ['DejaVuSans.ttf', 'Arial.ttf']):
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    try:
        from matplotlib import font_manager
        fp = font_manager.FontProperties(family='DejaVu Sans', weight='bold' if bold else 'normal')
        return ImageFont.truetype(font_manager.findfont(fp), size)
    except Exception as exc:  # last resort keeps the figure buildable
        print(f'[warn] no TrueType font found ({exc}); using the PIL default')
        return ImageFont.load_default()


def pick_baselines() -> list[str]:
    """Baselines with the largest per-scene C_mat (luminance change across lights)."""
    d = json.load(open(PER_SCENE))
    i = d['scenes'].index(SCENE)
    scores = {m: v['C_mat'][i] for m, v in d['methods'].items()
              if not m.startswith('Ours') and m in COLUMNS}
    if len(scores) < N_BASE:
        raise ValueError(f'only {len(scores)} baselines matched the grid columns')
    ranked = sorted(scores, key=scores.get, reverse=True)
    print('C_mat on', SCENE, {m: round(scores[m], 3) for m in ranked},
          '| ours', round(d['methods'][OURS_KEY]['C_mat'][i], 3))
    return ranked[:N_BASE]


def main():
    if not os.path.exists(SRC):
        raise FileNotFoundError(f'qualitative grid not found: {SRC}')
    grid = Image.open(SRC).convert('RGB')
    if grid.size != (1900, 472):
        raise ValueError(f'unexpected grid size {grid.size}; re-measure the panel geometry')

    cols = ['Input', 'GT albedo', 'Ours'] + pick_baselines()
    pw, ph = PW * SCALE, PH * SCALE
    head, lab_w = 64, 150
    canvas = Image.new('RGB', (lab_w + len(cols) * (pw + GAP) - GAP,
                               head + len(ROWS) * (ph + GAP) - GAP), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    for c, name in enumerate(cols):
        k = COLUMNS.index(name)
        sx = round(X0 + k * PITCH)
        x = lab_w + c * (pw + GAP)
        d.text((x + pw // 2, head // 2), name, fill=(20, 20, 20), anchor='mm',
               font=font(30, bold=name in ('GT albedo', 'Ours')))
        for r, sy in enumerate(ROW_Y):
            tile = grid.crop((sx, sy, sx + PW, sy + PH)).resize((pw, ph), Image.LANCZOS)
            canvas.paste(tile, (x, head + r * (ph + GAP)))
    for r, lab in enumerate(ROWS):
        d.text((lab_w // 2, head + r * (ph + GAP) + ph // 2), lab, fill=(20, 20, 20),
               anchor='mm', font=font(28, bold=True))
    for p in OUTS:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        canvas.save(p, quality=90, optimize=True)
        print('wrote', p, canvas.size)


if __name__ == '__main__':
    main()
