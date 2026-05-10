# tests/test_barcode.py
"""Barcode/QR detection tests."""

from pathlib import Path

from PIL import Image

from image_analyser.barcode import analyse


def test_qr_payload_decoded(fixtures_dir):
    img = Image.open(fixtures_dir / "qr.png")
    barcodes, reason = analyse(img)
    assert reason is None
    assert barcodes is not None
    assert len(barcodes) == 1
    b = barcodes[0]
    assert b.type == "QRCODE"
    assert b.value == "image-analyser-test"
    assert b.bbox.w > 0 and b.bbox.h > 0


def test_no_barcode_returns_empty(fixtures_dir):
    img = Image.open(fixtures_dir / "1x1.png")
    barcodes, reason = analyse(img)
    assert reason is None
    assert barcodes == []


def test_skips_when_libzbar_not_loadable(monkeypatch):
    """When pyzbar can't load libzbar, barcode.analyse should return (None, reason)."""
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyzbar.pyzbar" or name.startswith("pyzbar."):
            raise OSError("Unable to find zbar shared library")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    img = Image.open(Path("tests/fixtures/qr.png"))
    barcodes, reason = analyse(img)
    assert barcodes is None
    assert reason == "libzbar not loadable"
