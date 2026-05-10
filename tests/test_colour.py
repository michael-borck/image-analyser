# tests/test_colour.py
"""Colour analysis tests."""

import numpy as np
from PIL import Image

from image_analyser.colour import analyse


def test_solid_red_image_has_red_dominant_and_average():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = analyse(img)
    assert result.average == "#ff0000"
    assert result.dominant[0] == "#ff0000"
    # palette weights sum to ~1 (all red)
    assert abs(sum(p.weight for p in result.palette) - 1.0) < 1e-3


def test_two_band_image_has_two_dominant_colours():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50] = (255, 0, 0)
    arr[50:] = (0, 0, 255)
    img = Image.fromarray(arr, "RGB")
    result = analyse(img)
    palette_hexes = {p.hex for p in result.palette}
    # The two clusters should be near red and near blue
    assert any(h.startswith("#ff") for h in palette_hexes)
    assert any(h.startswith("#0000ff") or h.startswith("#0000fe") for h in palette_hexes)
