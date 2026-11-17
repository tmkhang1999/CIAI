#!/usr/bin/env python3
"""Panels for fig:wb_lanes: how one MID training pair is pre-processed.

The MID loader (src/data/midintrinsic_dataset.py) builds a CIAI pair from two flash
directions a and b of one scene:
  frame a  white-balanced on its own grey probe; also carries the supervised albedo loss
           against the white-balanced pseudo-GT albedo (the "anchor"),
  frame b  left raw, so it keeps its own illuminant colour.
This script writes the three panels (frame a, frame b, anchor) with one shared exposure so
the brightness difference between the frames is real. The TikZ layout lives in
documents/thesis/chapters/Chapter3.tex.

Reads MID test EXRs only; no model is involved.

Usage:
  python tests/viz/build_wb_lanes_figure.py [--scene everett_dining1] [--a 7 --b 12]
"""
from __future__ import annotations

import argparse
import os
import sys
from itertools import permutations

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'data'))
from mid_illuminant import probe_chromaticity  # noqa: E402

MID_TEST = os.path.join(ROOT, 'data', 'multi_illumination_test_mip2_exr')
OUT_DIR = os.path.join(ROOT, 'documents', 'thesis', 'images', 'wb')
TRAIN_SKIP = {2, 3, 20, 21, 24}  # same as MIDIntrinsicDataset.skip_list
TRAIN_IDX = [i for i in range(25) if i not in TRAIN_SKIP]
WIDTH = 640


def read_exr(path: str) -> np.ndarray:
    im = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if im is None:
        raise OSError(f'cannot read {path} (is OPENCV_IO_ENABLE_OPENEXR set?)')
    return np.clip(im[:, :, ::-1].astype(np.float64), 0.0, None)


def white_balance(scene: str, idx: int) -> np.ndarray:
    """Same rule as MIDIntrinsicDataset._white_balance: scale R and B so the probe is grey."""
    img = read_exr(os.path.join(scene, f'dir_{idx}_mip2.exr'))
    prb = read_exr(os.path.join(scene, 'probes', f'dir_{idx}_gray256.exr'))
    msk = np.any(prb > 0.01, axis=-1)
    msk = cv2.erode(np.pad(msk, 1).astype(np.uint8), np.ones((11, 11), np.uint8))[1:-1, 1:-1] > 0
    if not msk.any():
        print(f'[warn] probe mask empty for dir {idx}; frame left unbalanced')
        return img
    med = np.median(prb[msk], axis=0)
    return img * np.array([med[1] / (med[0] + 1e-6), 1.0, med[1] / (med[2] + 1e-6)])


def lum(x: np.ndarray) -> np.ndarray:
    return 0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]


def to_srgb8(x: np.ndarray, scale: float) -> np.ndarray:
    y = np.clip(x * scale, 0.0, 1.0) ** (1 / 2.2)
    h, w = y.shape[:2]
    y = cv2.resize(y, (WIDTH, int(round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    return (y[:, :, ::-1] * 255 + 0.5).astype(np.uint8)


def pick_pair(scene: str):
    """b = the training direction whose raw illuminant is furthest from neutral (the colour
    change L_inv sees); a = the direction whose lighting differs most from b."""
    ch = {i: probe_chromaticity(scene, i) for i in TRAIN_IDX}
    ch = {i: c for i, c in ch.items() if c is not None}
    if len(ch) < 2:
        raise RuntimeError(f'fewer than two readable probes in {scene}')
    b = max(ch, key=lambda i: np.hypot(ch[i][0] - 1 / 3, ch[i][2] - 1 / 3))
    lb = np.log(lum(read_exr(os.path.join(scene, f'dir_{b}_mip2.exr')))[::8, ::8] + 1e-3)
    a = max((i for i in ch if i != b), key=lambda i: np.abs(
        np.log(lum(read_exr(os.path.join(scene, f'dir_{i}_mip2.exr')))[::8, ::8] + 1e-3) - lb).mean())
    return a, b, float(np.hypot(ch[b][0] - 1 / 3, ch[b][2] - 1 / 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='everett_dining1')
    ap.add_argument('--a', type=int, default=None)
    ap.add_argument('--b', type=int, default=None)
    args = ap.parse_args()

    scene = os.path.join(MID_TEST, args.scene)
    if not os.path.isdir(scene):
        raise FileNotFoundError(f'scene not found: {scene}')
    if args.a is None or args.b is None:
        a, b, gap = pick_pair(scene)
    else:
        a, b = args.a, args.b
        cb = probe_chromaticity(scene, b)
        gap = float(np.hypot(cb[0] - 1 / 3, cb[2] - 1 / 3)) if cb is not None else float('nan')

    fa = white_balance(scene, a)
    fb = read_exr(os.path.join(scene, f'dir_{b}_mip2.exr'))
    alb = read_exr(os.path.join(scene, 'albedo.exr'))

    # One exposure for both frames, so the brightness change between them is preserved.
    shared = 0.8 / (np.percentile(lum(np.concatenate([fa, fb], 0)), 90) + 1e-8)
    alb_scale = 0.9 / (np.percentile(alb, 99) + 1e-8)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, img, s in [('pair_frame_a_wb.jpg', fa, shared), ('pair_frame_b_raw.jpg', fb, shared),
                         ('pair_anchor.jpg', alb, alb_scale)]:
        out = os.path.join(OUT_DIR, name)
        if not cv2.imwrite(out, to_srgb8(img, s), [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise OSError(f'failed to write {out}')
        print('wrote', out)
    print(f'scene {args.scene}: a={a} (white-balanced), b={b} (raw), '
          f'raw illuminant offset from neutral {gap:.3f}')


if __name__ == '__main__':
    main()
