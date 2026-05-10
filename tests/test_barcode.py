# tests/test_barcode.py
"""Barcode/QR detection tests."""

from PIL import Image

from image_analyser.barcode import analyse


def test_qr_payload_decoded(fixtures_dir):
    img = Image.open(fixtures_dir / "qr.png")
    barcodes = analyse(img)
    assert len(barcodes) == 1
    b = barcodes[0]
    assert b.type == "QRCODE"
    assert b.value == "image-analyser-test"
    assert b.bbox.w > 0 and b.bbox.h > 0


def test_no_barcode_returns_empty(fixtures_dir):
    img = Image.open(fixtures_dir / "1x1.png")
    assert analyse(img) == []
