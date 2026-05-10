# tests/test_quality.py
"""Quality signal tests."""

import numpy as np
from PIL import Image

from image_analyser.quality import analyse


def test_uniform_image_has_low_blur_low_contrast():
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    q = analyse(img, raw_bytes=b"")
    assert q.blur_score < 1.0
    assert q.contrast < 1.0
    assert q.exposure.underexposed_pct == 0.0
    assert q.exposure.overexposed_pct == 0.0


def test_high_frequency_image_has_high_blur_score():
    arr = (np.random.default_rng(0).integers(0, 256, (100, 100, 3))).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    q = analyse(img, raw_bytes=b"")
    assert q.blur_score > 100.0  # noise → high Laplacian variance


def test_overexposed_image_reports_clipping():
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    q = analyse(img, raw_bytes=b"")
    assert q.exposure.overexposed_pct > 0.99
    assert q.exposure.clipping_pct > 0.99


def test_jpeg_quality_estimate_reasonable_for_q85_fixture(fixtures_dir):
    img = Image.open(fixtures_dir / "small.jpg")
    raw = (fixtures_dir / "small.jpg").read_bytes()
    q = analyse(img, raw_bytes=raw)
    assert q.jpeg_quality_estimate is not None
    assert 50 <= q.jpeg_quality_estimate <= 100
