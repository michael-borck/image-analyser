# tests/test_metadata.py
"""Metadata tests: format/dims, EXIF, animation, C2PA."""

from PIL import Image

from image_analyser.metadata import analyse, basic


def test_basic_dims_for_1x1_png(fixtures_dir):
    img = Image.open(fixtures_dir / "1x1.png")
    raw = (fixtures_dir / "1x1.png").read_bytes()
    info = basic(img, raw)
    assert info["format"] == "PNG"
    assert info["resolution"] == (1, 1)
    assert info["megapixels"] == 0.0
    assert info["aspect_class"] == "square"
    assert info["colour_mode"] == "RGB"
    assert info["bit_depth"] == 8
    assert info["has_alpha"] is False


def test_landscape_classification():
    img = Image.new("RGB", (200, 100))
    raw = b""
    info = basic(img, raw)
    assert info["aspect_class"] == "landscape"


def test_portrait_classification():
    img = Image.new("RGB", (100, 200))
    info = basic(img, b"")
    assert info["aspect_class"] == "portrait"


def test_exif_roundtrip(fixtures_dir):
    img = Image.open(fixtures_dir / "small.jpg")
    raw = (fixtures_dir / "small.jpg").read_bytes()
    md = analyse(img, raw)
    assert md.exif is not None
    assert md.exif.camera == "TestCam Model 1"
    assert md.exif.iso == 400


def test_animation_detection_for_gif(fixtures_dir):
    img = Image.open(fixtures_dir / "animated.gif")
    raw = (fixtures_dir / "animated.gif").read_bytes()
    info = basic(img, raw)
    assert info["format"] == "GIF"
    # animation field is part of analyse(), tested via the animation helper:
    from image_analyser.metadata import animation_info
    a = animation_info(img)
    assert a is not None
    assert a.frame_count == 3


def test_no_c2pa_when_absent(fixtures_dir):
    img = Image.open(fixtures_dir / "1x1.png")
    raw = (fixtures_dir / "1x1.png").read_bytes()
    md = analyse(img, raw)
    # C2PA is None when not present — test_metadata.py asserts the absence-path
    assert md.c2pa is None or md.c2pa.present is False
