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
