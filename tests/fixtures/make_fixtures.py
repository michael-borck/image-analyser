"""Generate fixture images used by tests. Run once after cloning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FIXTURES_DIR = Path(__file__).parent


def make_1x1_png() -> None:
    Image.new("RGB", (1, 1), color=(0, 0, 0)).save(FIXTURES_DIR / "1x1.png", "PNG")


def make_small_jpg() -> None:
    """200x200 JPEG with embedded EXIF (camera + ISO)."""
    img = Image.new("RGB", (200, 200))
    pixels = np.indices((200, 200)).sum(axis=0).astype(np.uint8)
    arr = np.stack([pixels, pixels // 2, 255 - pixels], axis=-1)
    img = Image.fromarray(arr, "RGB")
    # Pillow doesn't write EXIF directly; use piexif if available, else write minimal exif via Image.Exif.
    exif = Image.Exif()
    exif[271] = "TestCam"            # Make
    exif[272] = "Model 1"            # Model
    exif[34855] = 400                # ISOSpeedRatings
    img.save(FIXTURES_DIR / "small.jpg", "JPEG", quality=85, exif=exif.tobytes())


def make_animated_gif() -> None:
    frames = [
        Image.new("RGB", (50, 50), color=(255, 0, 0)),
        Image.new("RGB", (50, 50), color=(0, 255, 0)),
        Image.new("RGB", (50, 50), color=(0, 0, 255)),
    ]
    frames[0].save(
        FIXTURES_DIR / "animated.gif",
        save_all=True, append_images=frames[1:], duration=100, loop=0,
    )


def make_qr_png() -> None:
    """Render a QR code containing the literal payload 'image-analyser-test'."""
    import qrcode  # only used here; not a runtime dep
    img = qrcode.make("image-analyser-test")
    img.save(FIXTURES_DIR / "qr.png")


def make_text_png() -> None:
    """Plain white image with the word 'IMAGE' in black for OCR tests."""
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 60)
    except OSError:
        font = ImageFont.load_default()
    draw.text((20, 20), "IMAGE", fill=(0, 0, 0), font=font)
    img.save(FIXTURES_DIR / "text.png")


def main() -> None:
    make_1x1_png()
    make_small_jpg()
    make_animated_gif()
    try:
        make_qr_png()
    except ImportError:
        print("Skipping qr.png — `pip install qrcode[pil]` to enable.")
    make_text_png()
    print("Fixtures generated in", FIXTURES_DIR)


if __name__ == "__main__":
    main()
