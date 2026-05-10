"""Barcode and QR-code detection via pyzbar."""

from __future__ import annotations

from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

from .schemas import Barcode, BBox


def analyse(img: Image.Image) -> list[Barcode]:
    """Detect barcodes and QR codes. Returns an empty list if none are present."""
    results: list[Barcode] = []
    for obj in zbar_decode(img):
        rect = obj.rect
        try:
            value = obj.data.decode("utf-8")
        except UnicodeDecodeError:
            value = obj.data.decode("latin-1", errors="replace")
        results.append(
            Barcode(
                type=obj.type,
                value=value,
                bbox=BBox(x=rect.left, y=rect.top, w=rect.width, h=rect.height),
            )
        )
    return results
