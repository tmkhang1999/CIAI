#!/usr/bin/env python3
"""Data-only figures of the report (no model needed), built from the MID test split.

  images/report/pair_mask.jpg     one CIAI pair: the two raw frames, the log luminance ratio
                                  that L_expl asks the shading to explain, and the validity
                                  mask Omega_v (15th to 97th luminance percentile in both
                                  frames, the rule of src/data/midintrinsic_dataset.py)
  images/report/mid_materials.jpg the photograph, MID's material segmentation and the
                                  MIDIntrinsics albedo used by C_mat, Cast_rel and Chroma_err

Needs data/multi_illumination_test_mip2_exr (MID test split with MIDIntrinsics albedo).

Run:
  python tests/viz/build_report_figures.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MID = ROOT / 'data' / 'multi_illumination_test_mip2_exr'
OUT = ROOT / 'documents' / 'thesis' / 'images' / 'report'
SCENE, DIR_A, DIR_B = 'everett_lobby19', 6, 12


def read_exr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if img is None:
        raise SystemExit(f'cannot read {path}')
    return np.clip(img[:, :, ::-1].astype(np.float32), 0.0, None)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def valid_mask(rgb: np.ndarray, specular_pct: float = 97.0, shadow_pct: float = 15.0) -> np.ndarray:
    """Same rule as MIDIntrinsicDataset._hdr_valid_mask."""
    lum = luminance(rgb)
    hi = float(np.percentile(lum, specular_pct))
    lo = max(float(np.percentile(lum, shadow_pct)), 5e-3)
    return (lum > lo) & (lum < hi)


def display(rgb: np.ndarray, scale: float) -> np.ndarray:
    return np.clip(rgb * scale, 0.0, 1.0) ** (1 / 2.2)


def pair_mask():
    scene = MID / SCENE
    a = read_exr(scene / f'dir_{DIR_A}_mip2.exr')
    b = read_exr(scene / f'dir_{DIR_B}_mip2.exr')
    # one shared exposure for both frames, so their brightness difference stays visible
    scale = 0.8 / float(np.percentile(np.mean(np.concatenate([a, b]), axis=2), 90))
    eps = 1e-3
    ratio = np.log((luminance(a) + eps) / (luminance(b) + eps))
    mask = valid_mask(a) & valid_mask(b)
    print(f'  valid pixels in both frames: {100 * mask.mean():.1f}%')
    fig, ax = plt.subplots(1, 4, figsize=(12, 2.35), dpi=200)
    ax[0].imshow(display(a, scale)); ax[0].set_title(f'$I_1$ (flash direction {DIR_A})', fontsize=9)
    ax[1].imshow(display(b, scale)); ax[1].set_title(f'$I_2$ (flash direction {DIR_B})', fontsize=9)
    lim = float(np.percentile(np.abs(ratio[mask]), 98))
    im = ax[2].imshow(np.where(mask, ratio, np.nan), cmap='coolwarm', vmin=-lim, vmax=lim)
    ax[2].set_title(r'$\log L(I_1) / L(I_2)$ on $\Omega_v$', fontsize=9)
    ax[3].imshow(mask, cmap='gray', vmin=0, vmax=1)
    ax[3].set_title(rf'$\Omega_v$ ({100 * mask.mean():.0f}% of pixels)', fontsize=9)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    cb = fig.colorbar(im, ax=ax[2], fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    return save(fig, 'pair_mask.jpg')


def mid_materials():
    scene = MID / SCENE
    a = read_exr(scene / f'dir_{DIR_A}_mip2.exr')
    albedo = read_exr(scene / 'albedo.exr')
    seg = cv2.imread(str(scene / 'materials_mip2.png'), cv2.IMREAD_UNCHANGED)
    if seg is None:
        raise SystemExit('materials_mip2.png missing')
    if seg.ndim == 3:
        seg = seg[..., 0]
    seg = cv2.resize(seg, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_NEAREST)
    ids = [v for v in np.unique(seg) if v != 0]
    rng = np.random.default_rng(0)
    colours = {v: rng.uniform(0.25, 1.0, 3) for v in ids}
    seg_rgb = np.zeros(a.shape, np.float32)
    for v, c in colours.items():
        seg_rgb[seg == v] = c
    scale = 0.8 / float(np.percentile(np.mean(a, axis=2), 90))
    fig, ax = plt.subplots(1, 3, figsize=(9, 2.35), dpi=200)
    ax[0].imshow(display(a, scale)); ax[0].set_title('Photograph (one flash direction)', fontsize=9)
    ax[1].imshow(seg_rgb); ax[1].set_title(f'Material segmentation ({len(ids)} materials)', fontsize=9)
    ax[2].imshow(np.clip(albedo, 0, 1) ** (1 / 2.2)); ax[2].set_title('MIDIntrinsics pseudo-GT albedo', fontsize=9)
    for x in ax:
        x.set_xticks([]); x.set_yticks([])
    fig.tight_layout()
    return save(fig, 'mid_materials.jpg')


def save(fig, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, bbox_inches='tight', pad_inches=0.02, pil_kwargs={'quality': 92})
    plt.close(fig)
    print(f'  {path.relative_to(ROOT)}')
    return path


def main():
    if not MID.is_dir():
        raise SystemExit(f'MID test split not found: {MID}')
    pair_mask()
    mid_materials()


if __name__ == '__main__':
    main()
