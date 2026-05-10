"""Barcode and QR-code detection via pyzbar.

`pyzbar` is imported lazily because it loads `libzbar` via ctypes at module
import time. On macOS Apple Silicon the Homebrew install path
(/opt/homebrew/lib) is not in the default dynamic-linker search path, so a
top-level import would crash `import image_analyser` for users who installed
zbar via brew but did not export `DYLD_LIBRARY_PATH`. Falling back to an
empty list keeps barcode detection a "do what you can" signal.
"""

from __future__ import annotations

import logging

from PIL import Image

from .schemas import Barcode, BBox

logger = logging.getLogger(__name__)


def analyse(img: Image.Image) -> list[Barcode]:
    """Detect barcodes and QR codes. Returns an empty list if pyzbar/libzbar are not loadable or no codes are present."""
    try:
        from pyzbar.pyzbar import decode as zbar_decode
    except (ImportError, OSError) as e:
        logger.warning("pyzbar/libzbar not loadable; barcode detection disabled: %s", e)
        return []
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
