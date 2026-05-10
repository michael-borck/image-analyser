"""Image hashing: SHA256 of raw bytes + perceptual pHash and dHash."""

from __future__ import annotations

import hashlib

import imagehash
from PIL import Image

from .schemas import Hash


def analyse(img: Image.Image, raw_bytes: bytes) -> Hash:
    """Return content + perceptual hashes for an image."""
    sha = hashlib.sha256(raw_bytes).hexdigest()
    phash = str(imagehash.phash(img))   # 16 hex chars
    dhash = str(imagehash.dhash(img))
    return Hash(sha256=sha, phash=phash, dhash=dhash)
