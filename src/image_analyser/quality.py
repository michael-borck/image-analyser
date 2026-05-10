"""Quality signals: blur (Laplacian variance), exposure, brightness, contrast, noise."""

from __future__ import annotations

import struct

import numpy as np
from PIL import Image

from .schemas import Exposure, Quality


def _grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.mean(axis=2)
    return arr.astype(float)


def _laplacian_variance(arr: np.ndarray) -> float:
    """3x3 Laplacian filter (no scipy)."""
    g = _grayscale(arr)
    # |0  1 0|
    # |1 -4 1|
    # |0  1 0|
    lap = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _exposure(arr: np.ndarray) -> Exposure:
    g = _grayscale(arr)
    n = g.size
    underexposed = float(np.sum(g < 16) / n)
    overexposed = float(np.sum(g > 239) / n)
    clipping = float(np.sum((g <= 0) | (g >= 255)) / n)
    return Exposure(
        underexposed_pct=underexposed,
        overexposed_pct=overexposed,
        clipping_pct=clipping,
    )


def _noise_estimate(arr: np.ndarray) -> float:
    """Rough noise estimate: stddev of high-frequency residual (image - 3x3 mean blur)."""
    g = _grayscale(arr)
    # 3x3 mean
    blurred = (
        g[:-2, :-2] + g[:-2, 1:-1] + g[:-2, 2:] +
        g[1:-1, :-2] + g[1:-1, 1:-1] + g[1:-1, 2:] +
        g[2:, :-2] + g[2:, 1:-1] + g[2:, 2:]
    ) / 9.0
    residual = g[1:-1, 1:-1] - blurred
    return float(residual.std())


def _jpeg_quality_estimate(raw_bytes: bytes) -> int | None:
    """Heuristic estimate from the DQT marker(s). Returns None for non-JPEG.

    Method: read the first quantisation table; compute the mean of its values.
    Apply the standard inverse-quality formula used by libjpeg:
        quality = max(1, min(100, round(100 - mean(Q) * 0.5)))
    Approximate, but stable across lossless re-saves.
    """
    if len(raw_bytes) < 4 or raw_bytes[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(raw_bytes) - 1:
        if raw_bytes[i] != 0xFF:
            i += 1
            continue
        marker = raw_bytes[i + 1]
        if marker == 0xDB:                  # DQT
            length = struct.unpack(">H", raw_bytes[i + 2:i + 4])[0]
            table = raw_bytes[i + 5:i + 4 + length]   # skip Pq/Tq byte
            if len(table) < 64:
                return None
            mean_q = sum(table[:64]) / 64.0
            return max(1, min(100, round(100 - mean_q * 0.5)))
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        # Skip variable-length segments
        if i + 3 >= len(raw_bytes):
            break
        seg_len = struct.unpack(">H", raw_bytes[i + 2:i + 4])[0]
        i += 2 + seg_len
    return None


def analyse(img: Image.Image, raw_bytes: bytes) -> Quality:
    arr = np.asarray(img.convert("RGB"))
    blur = _laplacian_variance(arr)
    g = _grayscale(arr)
    brightness = float(g.mean()) / 255.0
    contrast = float(g.std()) / 255.0
    return Quality(
        blur_score=blur,
        exposure=_exposure(arr),
        brightness=brightness,
        contrast=contrast,
        noise=_noise_estimate(arr),
        jpeg_quality_estimate=_jpeg_quality_estimate(raw_bytes),
    )
