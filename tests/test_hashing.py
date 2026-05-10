# tests/test_hashing.py
"""Hashing tests."""

from PIL import Image

from image_analyser.hashing import analyse


def test_hash_of_1x1_png_is_stable(fixtures_dir):
    img = Image.open(fixtures_dir / "1x1.png")
    raw = (fixtures_dir / "1x1.png").read_bytes()
    result = analyse(img, raw)
    assert len(result.sha256) == 64
    assert all(c in "0123456789abcdef" for c in result.sha256)
    assert len(result.phash) == 16
    assert len(result.dhash) == 16


def test_same_bytes_same_hash(fixtures_dir):
    raw = (fixtures_dir / "small.jpg").read_bytes()
    img1 = Image.open(fixtures_dir / "small.jpg")
    img2 = Image.open(fixtures_dir / "small.jpg")
    h1 = analyse(img1, raw)
    h2 = analyse(img2, raw)
    assert h1.sha256 == h2.sha256
    assert h1.phash == h2.phash


def test_different_images_different_phash(fixtures_dir):
    raw1 = (fixtures_dir / "1x1.png").read_bytes()
    raw2 = (fixtures_dir / "small.jpg").read_bytes()
    h1 = analyse(Image.open(fixtures_dir / "1x1.png"), raw1)
    h2 = analyse(Image.open(fixtures_dir / "small.jpg"), raw2)
    assert h1.sha256 != h2.sha256
    assert h1.phash != h2.phash
