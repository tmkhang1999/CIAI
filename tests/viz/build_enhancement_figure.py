#!/usr/bin/env python3
"""Shading-lift and residual edits for the enhancement example (fig:enhance).

The motivating use case is photo enhancement of interiors: brighten dark regions and
tone down glare without changing the materials. This script applies the two
factor-space edits of Chapter 6 to the saved decomposition figure
(documents/thesis/images/ch6/decomposition.jpg: input, albedo, shading, residual), so
no model has to be run.

How the saved panels map back to the factors (see build_ch6_figures.py):
  input    = sRGB(I)                      -> I = panel ** 2.2
  shading  = sRGB(S / p99(S))             -> S up to scale; the lift gain only uses
                                             L(S) / p95, so the unknown scale cancels
  residual = sRGB(R), true magnitude      -> R = panel ** 2.2
The diffuse part is taken as D = I - R, so the edits are
  shading lift   I' = D * g(S) + R          (eq:illum_edit with eq:shadow_lift)
  naive lift     I' = I * g(I)              (the same gain computed on the image)

Usage:
  python tests/viz/build_enhancement_figure.py
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, 'documents', 'thesis', 'images', 'ch6', 'decomposition.jpg')
OUT_DIR = os.path.join(ROOT, 'documents', 'thesis', 'images', 'ch6')
WEB_OUT = os.path.join(ROOT, 'docs', 'static', 'img', 'ciai-enhance.jpg')
README_OUT = os.path.join(ROOT, 'documents', 'thesis', 'images', 'readme', 'ciai-enhance.jpg')
THESIS_OUT = os.path.join(OUT_DIR, 'enhance.jpg')
GAIN_OUT = os.path.join(OUT_DIR, 'enhance_gain.jpg')

# Panel geometry of decomposition.jpg: four 620 px panels with 10 px gaps, header above.
PANEL_W, GAP, TOP = 620, 10, 88
GAMMA, PCT, MAX_GAIN = 0.5, 95.0, 2.5   # same settings as the Chapter 6 edit
CROP = (150, 250, 470, 470)             # lamp head and its wall shadow, panel pixels


def panel(img: np.ndarray, k: int) -> np.ndarray:
    x0 = k * (PANEL_W + GAP)
    return img[TOP:, x0:x0 + PANEL_W].astype(np.float64) / 255.0


def lum709(x: np.ndarray) -> np.ndarray:
    return 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]


def lift_gain(x: np.ndarray) -> np.ndarray:
    """Per-pixel gain of eq:shadow_lift; values above the p95 reference are unchanged."""
    lum = lum709(x)
    pos = lum[lum > 1e-6]
    if pos.size == 0:
        print('[warn] empty luminance; gain set to 1')
        return np.ones_like(lum)
    rel = np.clip(lum / (np.percentile(pos, PCT) + 1e-8), 0.02, 1.0)
    return np.minimum(rel ** (GAMMA - 1.0), MAX_GAIN)


def to8(x: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(x, 0, 1) ** (1 / 2.2) * 255 + 0.5).astype(np.uint8))


def font(size: int, bold: bool = False):
    names = ['DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf',
             'Arial Bold.ttf' if bold else 'Arial.ttf']
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    if not os.path.exists(SRC):
        raise FileNotFoundError(f'decomposition figure not found: {SRC}')
    img = np.asarray(Image.open(SRC).convert('RGB'))
    if img.shape[1] != 4 * PANEL_W + 3 * GAP:
        raise ValueError(f'unexpected figure width {img.shape[1]}; panel geometry changed')

    I = panel(img, 0) ** 2.2
    S = panel(img, 2) ** 2.2
    R = panel(img, 3) ** 2.2
    D = np.clip(I - R, 0, None)

    g_img, g_shd = lift_gain(I), lift_gain(S)
    results = {
        'input': I,
        'naive': I * g_img[..., None],
        'shading': D * g_shd[..., None] + R,
    }

    # Labelled strip for the README and the project page: full frames on top, a 2x crop
    # of the lamp and the shadow it casts on the wall below, where the difference shows.
    labels = ['Input photograph', 'Brighten the whole image', 'Brighten the shading only']
    arrs = list(results.values())
    x0, y0, x1, y1 = CROP
    pw = 520
    ph = int(pw * I.shape[0] / I.shape[1])
    cw, ch = pw, int(pw * (y1 - y0) / (x1 - x0))
    head, gap = 54, 8
    canvas = Image.new('RGB', (3 * pw + 2 * gap, head + ph + gap + ch), (255, 255, 255))
    d = ImageDraw.Draw(canvas)
    sx = pw / I.shape[1]
    for k, (lab, arr) in enumerate(zip(labels, arrs)):
        x = k * (pw + gap)
        full = to8(arr)
        canvas.paste(full.resize((pw, ph), Image.LANCZOS), (x, head))
        canvas.paste(full.crop(CROP).resize((cw, ch), Image.LANCZOS), (x, head + ph + gap))
        d.rectangle([x + x0 * sx, head + y0 * sx, x + x1 * sx, head + y1 * sx],
                    outline=(220, 40, 40), width=3)
        d.rectangle([x, head + ph + gap, x + cw - 1, head + ph + gap + ch - 1],
                    outline=(220, 40, 40), width=3)
        d.text((x + pw // 2, head // 2), lab, fill=(25, 25, 25), anchor='mm',
               font=font(26, bold=k == 2))
    for p in (WEB_OUT, README_OUT, THESIS_OUT):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        canvas.save(p, quality=90)
        print('wrote', p)

    save_gain_maps(g_img, g_shd)


def save_gain_maps(g_img: np.ndarray, g_shd: np.ndarray):
    """Where each version of the lift acts: gain computed from the image vs the shading."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f'[warn] matplotlib unavailable, gain maps skipped: {exc}')
        return
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 5.6), constrained_layout=True)
    titles = ['Gain from the image', 'Gain from the predicted shading']
    for ax, g, t in zip(axes, (g_img, g_shd), titles):
        im = ax.imshow(g, cmap='magma', vmin=1.0, vmax=MAX_GAIN)
        ax.set_title(t, fontsize=12)
        ax.axis('off')
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label('brightness gain', fontsize=11)
    fig.savefig(GAIN_OUT, dpi=170)
    plt.close(fig)
    print('wrote', GAIN_OUT)


if __name__ == '__main__':
    main()
