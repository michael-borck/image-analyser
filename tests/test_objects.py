# tests/test_objects.py
"""Object detection tests."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_analyser.objects import analyse, is_available


def test_skips_when_transformers_missing(monkeypatch):
    monkeypatch.setattr("image_analyser.objects._transformers_importable", lambda: False)
    objects, reason = analyse(Image.new("RGB", (10, 10)))
    assert objects is None
    assert reason == "ml extra not installed"


def test_filters_below_threshold(monkeypatch):
    monkeypatch.setattr("image_analyser.objects._transformers_importable", lambda: True)
    fake_pipeline = MagicMock(return_value=[
        {"label": "cat", "score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5}},
        {"label": "dog", "score": 0.3, "box": {"xmin": 6, "ymin": 6, "xmax": 9, "ymax": 9}},
    ])
    with patch("image_analyser.objects._get_pipeline", return_value=fake_pipeline):
        objects, reason = analyse(Image.new("RGB", (10, 10)), threshold=0.5)
    assert reason is None
    assert len(objects) == 1
    assert objects[0].label == "cat"
    assert objects[0].score == pytest.approx(0.9)


@pytest.mark.slow
def test_detr_real_model_finds_objects(fixtures_dir):
    if not is_available():
        pytest.skip("transformers not installed")
    img = Image.open(fixtures_dir / "small.jpg").convert("RGB")
    objects, reason = analyse(img, threshold=0.0)
    # DETR should always return at least one detection on a non-trivial image
    assert reason is None
    assert isinstance(objects, list)
