"""Colour analysis: dominant palette (k-means) + average colour."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .schemas import Colour, PaletteEntry


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(c))) for c in rgb])


def _kmeans(
    pixels: np.ndarray, k: int, max_iter: int = 20, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Tiny Lloyd k-means. pixels shape (n, 3)."""
    rng = np.random.default_rng(seed)
    n = pixels.shape[0]
    if n <= k:
        return pixels.astype(float), np.ones(n) / max(n, 1)
    centers = pixels[rng.choice(n, size=k, replace=False)].astype(float)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = pixels[mask].mean(axis=0)
    weights = np.bincount(labels, minlength=k).astype(float) / n
    return centers, weights


def analyse(img: Image.Image, k: int = 5, sample_size: int = 10_000) -> Colour:
    """Compute dominant palette, average colour, and weighted palette entries."""
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).reshape(-1, 3)
    if arr.shape[0] > sample_size:
        idx = np.random.default_rng(0).choice(arr.shape[0], size=sample_size, replace=False)
        arr = arr[idx]
    centers, weights = _kmeans(arr, k)
    # Sort palette by weight desc
    order = np.argsort(-weights)
    palette = [
        PaletteEntry(hex=_hex(tuple(centers[i])), weight=float(weights[i]))
        for i in order
        if weights[i] > 0
    ]
    dominant = [p.hex for p in palette[: min(3, len(palette))]]
    avg = arr.mean(axis=0)
    return Colour(dominant=dominant, average=_hex(tuple(avg)), palette=palette)
