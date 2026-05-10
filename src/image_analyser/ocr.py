"""OCR dispatcher: tesseract / easyocr."""

from __future__ import annotations

import importlib.util
import logging
import os
import shutil
from functools import lru_cache
from typing import Any

from PIL import Image

from .schemas import BBox, Ocr, OcrBlock

logger = logging.getLogger(__name__)


def _tesseract_available() -> bool:
    return (
        importlib.util.find_spec("pytesseract") is not None
        and shutil.which("tesseract") is not None
    )


def _easyocr_available() -> bool:
    return importlib.util.find_spec("easyocr") is not None


@lru_cache(maxsize=1)
def _easyocr_reader() -> Any:
    import easyocr
    return easyocr.Reader(["en"], gpu=False)


def _ocr_tesseract(img: Image.Image) -> Ocr:
    import pytesseract
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    blocks: list[OcrBlock] = []
    for i, txt in enumerate(data["text"]):
        if not txt.strip():
            continue
        blocks.append(
            OcrBlock(
                text=txt,
                bbox=BBox(
                    x=int(data["left"][i]), y=int(data["top"][i]),
                    w=int(data["width"][i]), h=int(data["height"][i]),
                ),
                confidence=float(data["conf"][i]) / 100.0 if float(data["conf"][i]) >= 0 else 0.0,
            )
        )
    text = " ".join(b.text for b in blocks)
    return Ocr(text=text, blocks=blocks, engine="tesseract")


def _ocr_easyocr(img: Image.Image) -> Ocr:
    import numpy as np
    reader = _easyocr_reader()
    raw = reader.readtext(np.asarray(img.convert("RGB")))
    blocks: list[OcrBlock] = []
    for box, txt, conf in raw:
        xs = [int(p[0]) for p in box]
        ys = [int(p[1]) for p in box]
        blocks.append(
            OcrBlock(
                text=str(txt),
                bbox=BBox(x=min(xs), y=min(ys), w=max(xs) - min(xs), h=max(ys) - min(ys)),
                confidence=float(conf),
            )
        )
    text = " ".join(b.text for b in blocks)
    return Ocr(text=text, blocks=blocks, engine="easyocr")


def analyse(img: Image.Image) -> tuple[Ocr | None, str | None]:
    engine = os.getenv("IMAGE_ANALYSER_OCR_ENGINE", "auto")
    if engine == "none":
        return None, "disabled by config"
    if engine == "tesseract":
        if not _tesseract_available():
            return None, "tesseract not installed"
        return _ocr_tesseract(img), None
    if engine == "easyocr":
        if not _easyocr_available():
            return None, "easyocr not installed"
        return _ocr_easyocr(img), None
    # auto
    if _tesseract_available():
        return _ocr_tesseract(img), None
    if _easyocr_available():
        return _ocr_easyocr(img), None
    return None, "no ocr engine available"
