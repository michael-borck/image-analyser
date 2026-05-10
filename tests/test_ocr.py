# tests/test_ocr.py
"""OCR dispatcher tests."""

from unittest.mock import patch

import pytest
from PIL import Image

from image_analyser.ocr import analyse


def test_disabled_by_config(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_OCR_ENGINE", "none")
    res, reason = analyse(Image.new("RGB", (10, 10)))
    assert res is None
    assert reason == "disabled by config"


def test_auto_skips_when_neither_engine(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_OCR_ENGINE", "auto")
    with patch("image_analyser.ocr._tesseract_available", return_value=False), \
         patch("image_analyser.ocr._easyocr_available", return_value=False):
        res, reason = analyse(Image.new("RGB", (10, 10)))
    assert res is None
    assert reason == "no ocr engine available"


def test_explicit_tesseract_skips_when_missing(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_OCR_ENGINE", "tesseract")
    with patch("image_analyser.ocr._tesseract_available", return_value=False):
        res, reason = analyse(Image.new("RGB", (10, 10)))
    assert res is None
    assert reason == "tesseract not installed"


def test_explicit_easyocr_skips_when_missing(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_OCR_ENGINE", "easyocr")
    with patch("image_analyser.ocr._easyocr_available", return_value=False):
        res, reason = analyse(Image.new("RGB", (10, 10)))
    assert res is None
    assert reason == "easyocr not installed"


@pytest.mark.slow
def test_tesseract_extracts_text(fixtures_dir, monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_OCR_ENGINE", "tesseract")
    from image_analyser.ocr import _tesseract_available
    if not _tesseract_available():
        pytest.skip("tesseract not installed")
    img = Image.open(fixtures_dir / "text.png")
    res, reason = analyse(img)
    assert reason is None
    assert "IMAGE" in res.text.upper()
    assert res.engine == "tesseract"


def test_average_confidence_is_mean_of_blocks(monkeypatch):
    """OCR result's average_confidence equals the mean of per-block confidences."""
    monkeypatch.setenv("IMAGE_ANALYSER_OCR_ENGINE", "tesseract")

    fake_data = {
        "text": ["foo", "bar", "baz"],
        "left": [0, 10, 20],
        "top": [0, 0, 0],
        "width": [5, 5, 5],
        "height": [10, 10, 10],
        # tesseract reports confidences in 0..100; the dispatcher divides by 100
        "conf": [90.0, 70.0, 50.0],
    }

    fake_pyt = type(
        "FakePytesseract",
        (),
        {
            "image_to_data": staticmethod(lambda img, output_type=None: fake_data),
            "Output": type("Output", (), {"DICT": "dict"}),
        },
    )()

    with patch("image_analyser.ocr._tesseract_available", return_value=True), \
         patch.dict("sys.modules", {"pytesseract": fake_pyt}):
        res, reason = analyse(Image.new("RGB", (10, 10)))

    assert reason is None
    assert res is not None
    assert len(res.blocks) == 3
    # Mean of (0.9 + 0.7 + 0.5) / 3 = 0.7
    assert res.average_confidence == 0.7


def test_average_confidence_zero_when_no_blocks(monkeypatch):
    """When OCR finds no blocks, average_confidence should be 0.0."""
    monkeypatch.setenv("IMAGE_ANALYSER_OCR_ENGINE", "tesseract")

    fake_data = {
        "text": ["", "  ", ""],
        "left": [0, 10, 20],
        "top": [0, 0, 0],
        "width": [5, 5, 5],
        "height": [10, 10, 10],
        "conf": [-1.0, -1.0, -1.0],
    }

    fake_pyt = type(
        "FakePytesseract",
        (),
        {
            "image_to_data": staticmethod(lambda img, output_type=None: fake_data),
            "Output": type("Output", (), {"DICT": "dict"}),
        },
    )()

    with patch("image_analyser.ocr._tesseract_available", return_value=True), \
         patch.dict("sys.modules", {"pytesseract": fake_pyt}):
        res, reason = analyse(Image.new("RGB", (10, 10)))

    assert reason is None
    assert res is not None
    assert res.blocks == []
    assert res.average_confidence == 0.0
