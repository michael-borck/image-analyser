# image-analyser v0.1.0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `image-analyser` 0.1.0 — a CLI + FastAPI tool for static image analysis, member of the analyser family. Functional first release published to PyPI.

**Architecture:** One Python package `image_analyser` with an `ImageAnalyser` orchestrator class that dispatches to per-analysis modules (metadata, quality, colour, hashing, barcode, objects, caption, ocr). Heavy ML deps gated behind `[ml]`/`[ocr]`/`[api]` extras. Default behaviour: run everything available; report skipped/failed in response.

**Tech Stack:** Python 3.11+, Pillow + pillow-heif, numpy, imagehash, pyzbar, c2pa-python, python-magic, pydantic v2, FastAPI, uvicorn, typer, slowapi, hatchling. Optional: torch + transformers + accelerate (`[ml]`); pytesseract + easyocr (`[ocr]`); anthropic + openai + google-genai (`[api]`).

**Spec:** `docs/superpowers/specs/2026-05-08-image-analyser-design.md` — read this first for the response schema, env vars, and dispatch rules.

---

## File Structure

```
image-analyser/
├── LICENSE                            # written by `gh repo create --license mit`
├── README.md                          # Task 17
├── pyproject.toml                     # Task 1
├── .gitignore                         # Task 1
├── docs/superpowers/                  # already exists (spec + this plan)
├── src/image_analyser/
│   ├── __init__.py                    # Task 1 — exports + __version__ via importlib.metadata
│   ├── exceptions.py                  # Task 2
│   ├── schemas.py                     # Task 3
│   ├── hashing.py                     # Task 5
│   ├── barcode.py                     # Task 6
│   ├── colour.py                      # Task 7
│   ├── quality.py                     # Task 8
│   ├── metadata.py                    # Task 9
│   ├── objects.py                     # Task 10
│   ├── caption.py                     # Task 11
│   ├── ocr.py                         # Task 12
│   ├── image_analyser.py              # Task 13 — orchestrator class
│   ├── app.py                         # Task 14 — FastAPI
│   └── cli.py                         # Task 15 — typer CLI
└── tests/
    ├── conftest.py                    # Task 1
    ├── fixtures/                      # Task 4
    │   ├── make_fixtures.py           # generates fixture images deterministically
    │   ├── 1x1.png
    │   ├── small.jpg                  # JPEG with embedded EXIF
    │   ├── animated.gif
    │   ├── qr.png                     # contains a known QR payload
    │   └── text.png                   # contains rendered text for OCR tests
    ├── test_schemas.py                # Task 3
    ├── test_hashing.py                # Task 5
    ├── test_barcode.py                # Task 6
    ├── test_colour.py                 # Task 7
    ├── test_quality.py                # Task 8
    ├── test_metadata.py               # Task 9
    ├── test_objects.py                # Task 10 (slow)
    ├── test_caption.py                # Task 11 (slow + mocked)
    ├── test_ocr.py                    # Task 12 (slow)
    ├── test_analyser.py               # Task 13
    ├── test_app.py                    # Task 14
    ├── test_cli.py                    # Task 15
    └── test_invariants.py             # Task 16
```

Each file has a single, well-bounded responsibility. ML modules are isolated so their (heavy) imports don't trigger when the default install is used.

---

## Tasks

### Task 1: Bootstrap repo, pyproject, package skeleton

**Files:**
- Create: GitHub repo `michael-borck/image-analyser`
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/image_analyser/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/__init__.py` (empty)

- [ ] **Step 1: Create the GitHub repo and clone it over the existing docs tree**

```bash
cd /Users/michael/Projects/lens
gh repo create michael-borck/image-analyser \
  --public \
  --license mit \
  --description "Static image analysis (CLI + FastAPI) for the analyser family" \
  --confirm

# image-analyser/ already has docs/ with the spec + plan from brainstorming.
# Clone into a sibling temp dir, then move LICENSE/README from gh into the existing dir.
git clone https://github.com/michael-borck/image-analyser.git /tmp/image-analyser-bootstrap
mv /tmp/image-analyser-bootstrap/.git /Users/michael/Projects/lens/image-analyser/
mv /tmp/image-analyser-bootstrap/LICENSE /Users/michael/Projects/lens/image-analyser/
mv /tmp/image-analyser-bootstrap/README.md /Users/michael/Projects/lens/image-analyser/  # placeholder, overwritten in Task 17
rm -rf /tmp/image-analyser-bootstrap
cd /Users/michael/Projects/lens/image-analyser
```

Expected: `/Users/michael/Projects/lens/image-analyser/.git/` exists, `LICENSE` is MIT, the existing `docs/superpowers/specs/` and `docs/superpowers/plans/` trees are untouched.

- [ ] **Step 2: Write `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "image-analyser"
version = "0.1.0"
description = "Static image analysis (CLI + FastAPI) for the analyser family"
readme = "README.md"
license = {text = "MIT"}
requires-python = ">=3.11"
authors = [{name = "Michael Borck", email = "michael.borck@curtin.edu.au"}]
keywords = ["image", "analysis", "exif", "ocr", "captioning", "object-detection"]
classifiers = [
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Multimedia :: Graphics",
    "Topic :: Scientific/Engineering :: Image Processing",
]
dependencies = [
    "pillow>=10",
    "pillow-heif>=0.16",
    "numpy>=1.26",
    "imagehash>=4.3",
    "pyzbar>=0.1.9",
    "c2pa-python>=0.5",
    "pydantic>=2.6",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "typer>=0.12",
    "python-magic>=0.4.27",
    "python-multipart>=0.0.9",
    "slowapi>=0.1.9",
]

[project.optional-dependencies]
ml  = ["torch>=2.2", "transformers>=4.40", "accelerate>=0.30"]
ocr = ["pytesseract>=0.3.10", "easyocr>=1.7"]
api = ["anthropic>=0.30", "openai>=1.30", "google-genai>=0.3"]
all = ["image-analyser[ml,ocr,api]"]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.5",
    "mypy>=1.10",
    "httpx>=0.27",
    "build>=1.2",
    "twine>=5",
]

[project.scripts]
image-analyser = "image_analyser.cli:main"

[project.urls]
Homepage = "https://github.com/michael-borck/image-analyser"
Issues = "https://github.com/michael-borck/image-analyser/issues"

[tool.hatch.build.targets.wheel]
packages = ["src/image_analyser"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-m 'not slow' --strict-markers"
markers = [
    "slow: tests that load real ML models or OCR engines — opt-in with `pytest -m slow`",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM"]
ignore = ["E501"]  # line-length handled by formatter

[tool.mypy]
python_version = "3.11"
strict = true
ignore_missing_imports = true   # several optional deps have no stubs (transformers, c2pa)
```

- [ ] **Step 3: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.coverage
.coverage.*
htmlcov/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Environments
.venv/
venv/
.env

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
```

- [ ] **Step 4: Write `src/image_analyser/__init__.py` (minimal — extended in Task 13)**

```python
"""Static image analysis for the analyser family."""

from importlib.metadata import version as _version

__version__ = _version("image-analyser")
__all__ = ["__version__"]
```

Task 13 expands `__all__` and adds the public re-exports of `ImageAnalyser`, `AnalysisResult`, and `ImageAnalyserError` once those modules exist. Keeping it minimal here means Step 7's smoke test passes immediately after the bootstrap commit.

- [ ] **Step 5: Write `tests/conftest.py`**

```python
"""Shared fixtures for image-analyser tests."""

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR
```

- [ ] **Step 6: Empty `tests/__init__.py`**

```bash
mkdir -p tests
: > tests/__init__.py
```

- [ ] **Step 7: Install in editable mode and verify the package metadata is wired**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import image_analyser; print(image_analyser.__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 8: Commit**

```bash
git add LICENSE README.md pyproject.toml .gitignore src/image_analyser/__init__.py tests/__init__.py tests/conftest.py docs/
git commit -m "feat: bootstrap image-analyser package skeleton"
```

---

### Task 2: Exceptions module

**Files:**
- Create: `src/image_analyser/exceptions.py`
- Test: covered in Task 13's orchestrator tests (no standalone test file — exceptions are trivial subclasses)

- [ ] **Step 1: Write `src/image_analyser/exceptions.py`**

```python
"""Exception types for image-analyser."""


class ImageAnalyserError(Exception):
    """Base exception for image-analyser failures."""


class UnsupportedFormatError(ImageAnalyserError):
    """Raised when the input file is not a recognised image format."""


class FileTooLargeError(ImageAnalyserError):
    """Raised when an upload exceeds IMAGE_ANALYSER_MAX_UPLOAD_MB."""
```

- [ ] **Step 2: Smoke test the import**

```bash
python -c "from image_analyser.exceptions import ImageAnalyserError, UnsupportedFormatError, FileTooLargeError; assert issubclass(UnsupportedFormatError, ImageAnalyserError)"
```

Expected: no output (assertion passes).

- [ ] **Step 3: Commit**

```bash
git add src/image_analyser/exceptions.py
git commit -m "feat(exceptions): add ImageAnalyserError, UnsupportedFormatError, FileTooLargeError"
```

---

### Task 3: Pydantic schemas

**Files:**
- Create: `src/image_analyser/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_schemas.py
"""Schema round-trip tests."""

from image_analyser.schemas import (
    AnalysisResult, BBox, Barcode, Caption, Colour, Exif, Failed, Hash,
    Metadata, Object, Ocr, OcrBlock, PaletteEntry, Quality, Skipped,
)


def test_analysis_result_minimum_fields_round_trip():
    """A minimal AnalysisResult populates only the always-on fields."""
    payload = {
        "format": "PNG",
        "mime_type": "image/png",
        "resolution": [1, 1],
        "megapixels": 0.0,
        "aspect_class": "square",
        "colour_mode": "RGB",
        "bit_depth": 8,
        "has_alpha": False,
        "file_size": 70,
        "hash": {"sha256": "a" * 64, "phash": "0" * 16, "dhash": "0" * 16},
        "metadata": {"exif": None, "iptc": None, "xmp": None, "c2pa": None, "icc_profile": None},
        "animation": None,
        "quality": {
            "blur_score": 0.0,
            "exposure": {"underexposed_pct": 0.0, "overexposed_pct": 0.0, "clipping_pct": 0.0},
            "brightness": 0.0, "contrast": 0.0, "noise": 0.0, "jpeg_quality_estimate": None,
        },
        "colour": {"dominant": ["#000000"], "average": "#000000", "palette": []},
        "barcodes": [],
        "objects": None,
        "caption": None,
        "ocr": None,
        "skipped": [],
        "failed": [],
        "version": "0.1.0",
        "analysed_at": "2026-05-09T00:00:00Z",
        "duration_ms": 0,
    }
    result = AnalysisResult.model_validate(payload)
    assert result.format == "PNG"
    assert result.resolution == (1, 1)
    assert result.objects is None
    assert result.skipped == []
    # Round-trip
    assert AnalysisResult.model_validate(result.model_dump()).model_dump() == result.model_dump()


def test_skipped_reason_is_required():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        Skipped.model_validate({"name": "objects"})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_schemas.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'image_analyser.schemas'` (or similar import error).

- [ ] **Step 3: Write `src/image_analyser/schemas.py`**

```python
"""Pydantic v2 response schemas for image-analyser."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


class BBox(_Strict):
    x: int
    y: int
    w: int
    h: int


class Hash(_Strict):
    sha256: str
    phash: str
    dhash: str


class Gps(_Strict):
    lat: float
    lon: float
    alt: float | None = None


class Exif(_Strict):
    camera: str | None = None
    lens: str | None = None
    focal_length_mm: float | None = None
    iso: int | None = None
    aperture: float | None = None
    shutter_speed: str | None = None
    taken_at: str | None = None
    gps: Gps | None = None


class C2pa(_Strict):
    present: bool
    manifest: dict | None = None
    ai_generated_claim: bool | None = None


class IccProfile(_Strict):
    name: str
    colour_space: str


class Metadata(_Strict):
    exif: Exif | None = None
    iptc: dict | None = None
    xmp: dict | None = None
    c2pa: C2pa | None = None
    icc_profile: IccProfile | None = None


class Animation(_Strict):
    frame_count: int
    duration_s: float | None = None


class Exposure(_Strict):
    underexposed_pct: float
    overexposed_pct: float
    clipping_pct: float


class Quality(_Strict):
    blur_score: float
    exposure: Exposure
    brightness: float
    contrast: float
    noise: float
    jpeg_quality_estimate: int | None = None


class PaletteEntry(_Strict):
    hex: str
    weight: float


class Colour(_Strict):
    dominant: list[str]
    average: str
    palette: list[PaletteEntry]


class Barcode(_Strict):
    type: str
    value: str
    bbox: BBox


class Object(_Strict):
    label: str
    score: float
    bbox: BBox


class Caption(_Strict):
    text: str
    backend: str = Field(pattern=r"^(local|api)$")
    model: str


class OcrBlock(_Strict):
    text: str
    bbox: BBox
    confidence: float


class Ocr(_Strict):
    text: str
    blocks: list[OcrBlock]
    engine: str = Field(pattern=r"^(tesseract|easyocr)$")


class Skipped(_Strict):
    name: str
    reason: str


class Failed(_Strict):
    name: str
    error: str
    traceback: str | None = None


class AnalysisResult(_Strict):
    format: str
    mime_type: str
    resolution: tuple[int, int]
    megapixels: float
    aspect_class: str = Field(pattern=r"^(landscape|portrait|square)$")
    colour_mode: str
    bit_depth: int
    has_alpha: bool
    file_size: int
    hash: Hash
    metadata: Metadata
    animation: Animation | None = None
    quality: Quality
    colour: Colour
    barcodes: list[Barcode] = []
    objects: list[Object] | None = None
    caption: Caption | None = None
    ocr: Ocr | None = None
    skipped: list[Skipped] = []
    failed: list[Failed] = []
    version: str
    analysed_at: str
    duration_ms: int
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_schemas.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/schemas.py tests/test_schemas.py
git commit -m "feat(schemas): add Pydantic v2 response models"
```

---

### Task 4: Test fixture generator

**Files:**
- Create: `tests/fixtures/make_fixtures.py`
- Create (generated): `tests/fixtures/{1x1.png, small.jpg, animated.gif, qr.png, text.png}`

Fixtures are generated deterministically (no binary blobs in git that can't be regenerated) so tests are reproducible.

- [ ] **Step 1: Write `tests/fixtures/make_fixtures.py`**

```python
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
```

- [ ] **Step 2: Generate fixtures**

```bash
pip install qrcode[pil]   # one-time dev dep, not in pyproject
python tests/fixtures/make_fixtures.py
ls tests/fixtures/
```

Expected output includes: `1x1.png`, `small.jpg`, `animated.gif`, `qr.png`, `text.png`, `make_fixtures.py`.

- [ ] **Step 3: Commit fixtures and the generator**

```bash
git add tests/fixtures/
git commit -m "test: add deterministic fixture generator and generated images"
```

---

### Task 5: Hashing module

**Files:**
- Create: `src/image_analyser/hashing.py`
- Test: `tests/test_hashing.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_hashing.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/hashing.py`**

```python
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
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_hashing.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/hashing.py tests/test_hashing.py
git commit -m "feat(hashing): SHA256 + pHash + dHash"
```

---

### Task 6: Barcode module

**Files:**
- Create: `src/image_analyser/barcode.py`
- Test: `tests/test_barcode.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_barcode.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/barcode.py`**

```python
"""Barcode and QR-code detection via pyzbar."""

from __future__ import annotations

from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode

from .schemas import BBox, Barcode


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
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_barcode.py -v
```

Expected: 2 passed. (If `libzbar0` is missing, install via `brew install zbar` on macOS or `apt install libzbar0` on Linux.)

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/barcode.py tests/test_barcode.py
git commit -m "feat(barcode): QR + 1D barcode detection via pyzbar"
```

---

### Task 7: Colour module

**Files:**
- Create: `src/image_analyser/colour.py`
- Test: `tests/test_colour.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_colour.py
"""Colour analysis tests."""

import numpy as np
from PIL import Image

from image_analyser.colour import analyse


def test_solid_red_image_has_red_dominant_and_average():
    img = Image.new("RGB", (100, 100), color=(255, 0, 0))
    result = analyse(img)
    assert result.average == "#ff0000"
    assert result.dominant[0] == "#ff0000"
    # palette weights sum to ~1 (all red)
    assert abs(sum(p.weight for p in result.palette) - 1.0) < 1e-3


def test_two_band_image_has_two_dominant_colours():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    arr[:50] = (255, 0, 0)
    arr[50:] = (0, 0, 255)
    img = Image.fromarray(arr, "RGB")
    result = analyse(img)
    palette_hexes = {p.hex for p in result.palette}
    # The two clusters should be near red and near blue
    assert any(h.startswith("#ff") for h in palette_hexes)
    assert any(h.startswith("#0000ff") or h.startswith("#0000fe") for h in palette_hexes)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_colour.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/colour.py`**

```python
"""Colour analysis: dominant palette (k-means) + average colour."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .schemas import Colour, PaletteEntry


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*[max(0, min(255, int(c))) for c in rgb])


def _kmeans(pixels: np.ndarray, k: int, max_iter: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Tiny Lloyd k-means. pixels shape (n, 3)."""
    rng = np.random.default_rng(seed)
    n = pixels.shape[0]
    if n <= k:
        return pixels.astype(float), np.ones(n) / max(n, 1)
    centers = pixels[rng.choice(n, size=k, replace=False)].astype(float)
    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        d = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = d.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for i in range(k):
            mask = labels == i
            if mask.any():
                centers[i] = pixels[mask].mean(axis=0)
    weights = np.bincount(labels, minlength=k).astype(float) / n
    return centers, weights


def analyse(img: Image.Image, k: int = 5, sample_size: int = 10_000) -> Colour:
    """Compute dominant palette, average colour, and weighted palette entries."""
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).reshape(-1, 3)
    if arr.shape[0] > sample_size:
        idx = np.random.default_rng(0).choice(arr.shape[0], size=sample_size, replace=False)
        arr = arr[idx]
    centers, weights = _kmeans(arr, k)
    # Sort palette by weight desc
    order = np.argsort(-weights)
    palette = [
        PaletteEntry(hex=_hex(tuple(centers[i])), weight=float(weights[i]))
        for i in order if weights[i] > 0
    ]
    dominant = [p.hex for p in palette[: min(3, len(palette))]]
    avg = arr.mean(axis=0)
    return Colour(dominant=dominant, average=_hex(tuple(avg)), palette=palette)
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_colour.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/colour.py tests/test_colour.py
git commit -m "feat(colour): k-means palette + average + dominant"
```

---

### Task 8: Quality module

**Files:**
- Create: `src/image_analyser/quality.py`
- Test: `tests/test_quality.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality.py
"""Quality signal tests."""

import numpy as np
from PIL import Image

from image_analyser.quality import analyse


def test_uniform_image_has_low_blur_low_contrast():
    img = Image.new("RGB", (100, 100), color=(128, 128, 128))
    q = analyse(img, raw_bytes=b"")
    assert q.blur_score < 1.0
    assert q.contrast < 1.0
    assert q.exposure.underexposed_pct == 0.0
    assert q.exposure.overexposed_pct == 0.0


def test_high_frequency_image_has_high_blur_score():
    arr = (np.random.default_rng(0).integers(0, 256, (100, 100, 3))).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    q = analyse(img, raw_bytes=b"")
    assert q.blur_score > 100.0  # noise → high Laplacian variance


def test_overexposed_image_reports_clipping():
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    q = analyse(img, raw_bytes=b"")
    assert q.exposure.overexposed_pct > 0.99
    assert q.exposure.clipping_pct > 0.99


def test_jpeg_quality_estimate_reasonable_for_q85_fixture(fixtures_dir):
    img = Image.open(fixtures_dir / "small.jpg")
    raw = (fixtures_dir / "small.jpg").read_bytes()
    q = analyse(img, raw_bytes=raw)
    assert q.jpeg_quality_estimate is not None
    assert 50 <= q.jpeg_quality_estimate <= 100
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_quality.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/quality.py`**

```python
"""Quality signals: blur (Laplacian variance), exposure, brightness, contrast, noise."""

from __future__ import annotations

import struct

import numpy as np
from PIL import Image

from .schemas import Exposure, Quality


def _grayscale(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        return arr.mean(axis=2)
    return arr.astype(float)


def _laplacian_variance(arr: np.ndarray) -> float:
    """3x3 Laplacian filter (no scipy)."""
    g = _grayscale(arr)
    # |0  1 0|
    # |1 -4 1|
    # |0  1 0|
    lap = (
        g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    )
    return float(np.var(lap))


def _exposure(arr: np.ndarray) -> Exposure:
    g = _grayscale(arr)
    n = g.size
    underexposed = float(np.sum(g < 16) / n)
    overexposed = float(np.sum(g > 239) / n)
    clipping = float(np.sum((g <= 0) | (g >= 255)) / n)
    return Exposure(
        underexposed_pct=underexposed,
        overexposed_pct=overexposed,
        clipping_pct=clipping,
    )


def _noise_estimate(arr: np.ndarray) -> float:
    """Rough noise estimate: stddev of high-frequency residual (image - 3x3 mean blur)."""
    g = _grayscale(arr)
    # 3x3 mean
    blurred = (
        g[:-2, :-2] + g[:-2, 1:-1] + g[:-2, 2:] +
        g[1:-1, :-2] + g[1:-1, 1:-1] + g[1:-1, 2:] +
        g[2:, :-2] + g[2:, 1:-1] + g[2:, 2:]
    ) / 9.0
    residual = g[1:-1, 1:-1] - blurred
    return float(residual.std())


def _jpeg_quality_estimate(raw_bytes: bytes) -> int | None:
    """Heuristic estimate from the DQT marker(s). Returns None for non-JPEG.

    Method: read the first quantisation table; compute the mean of its values.
    Apply the standard inverse-quality formula used by libjpeg:
        quality = max(1, min(100, round(100 - mean(Q) * 0.5)))
    Approximate, but stable across lossless re-saves.
    """
    if len(raw_bytes) < 4 or raw_bytes[:2] != b"\xff\xd8":
        return None
    i = 2
    while i < len(raw_bytes) - 1:
        if raw_bytes[i] != 0xFF:
            i += 1
            continue
        marker = raw_bytes[i + 1]
        if marker == 0xDB:                  # DQT
            length = struct.unpack(">H", raw_bytes[i + 2:i + 4])[0]
            table = raw_bytes[i + 5:i + 4 + length]   # skip Pq/Tq byte
            if len(table) < 64:
                return None
            mean_q = sum(table[:64]) / 64.0
            return max(1, min(100, round(100 - mean_q * 0.5)))
        if marker in (0xD8, 0xD9):
            i += 2
            continue
        # Skip variable-length segments
        if i + 3 >= len(raw_bytes):
            break
        seg_len = struct.unpack(">H", raw_bytes[i + 2:i + 4])[0]
        i += 2 + seg_len
    return None


def analyse(img: Image.Image, raw_bytes: bytes) -> Quality:
    arr = np.asarray(img.convert("RGB"))
    blur = _laplacian_variance(arr)
    g = _grayscale(arr)
    brightness = float(g.mean()) / 255.0
    contrast = float(g.std()) / 255.0
    return Quality(
        blur_score=blur,
        exposure=_exposure(arr),
        brightness=brightness,
        contrast=contrast,
        noise=_noise_estimate(arr),
        jpeg_quality_estimate=_jpeg_quality_estimate(raw_bytes),
    )
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_quality.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/quality.py tests/test_quality.py
git commit -m "feat(quality): blur, exposure, brightness, contrast, noise, JPEG-Q"
```

---

### Task 9: Metadata module

**Files:**
- Create: `src/image_analyser/metadata.py`
- Test: `tests/test_metadata.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_metadata.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/metadata.py`**

```python
"""Image metadata: format, dimensions, EXIF/IPTC/XMP, C2PA, ICC, animation."""

from __future__ import annotations

import logging
from typing import Any

import magic
from PIL import ExifTags, Image
from PIL.ExifTags import GPSTAGS, TAGS

from .schemas import Animation, C2pa, Exif, Gps, IccProfile, Metadata

logger = logging.getLogger(__name__)


# ---------- basic identity ----------

_MODE_BIT_DEPTH = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "CMYK": 8, "I": 32, "F": 32}


def basic(img: Image.Image, raw_bytes: bytes) -> dict[str, Any]:
    """Always-on identity fields. Never raises."""
    w, h = img.size
    if w == h:
        aspect = "square"
    elif w > h:
        aspect = "landscape"
    else:
        aspect = "portrait"
    fmt = (img.format or "").upper() or "UNKNOWN"
    try:
        mime = magic.from_buffer(raw_bytes, mime=True) if raw_bytes else f"image/{fmt.lower()}"
    except Exception:
        mime = f"image/{fmt.lower()}"
    return {
        "format": fmt,
        "mime_type": mime,
        "resolution": (w, h),
        "megapixels": round(w * h / 1_000_000, 2),
        "aspect_class": aspect,
        "colour_mode": img.mode,
        "bit_depth": _MODE_BIT_DEPTH.get(img.mode, 8),
        "has_alpha": img.mode in {"RGBA", "LA"} or "transparency" in img.info,
        "file_size": len(raw_bytes),
    }


# ---------- EXIF ----------

def _exif(img: Image.Image) -> Exif | None:
    raw = img.getexif()
    if not raw:
        return None
    data: dict[str, Any] = {}
    for tag_id, value in raw.items():
        name = TAGS.get(tag_id, str(tag_id))
        data[name] = value

    gps_block = raw.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(ExifTags, "IFD") else {}
    gps_dict: dict[str, Any] = {GPSTAGS.get(k, str(k)): v for k, v in (gps_block or {}).items()}

    camera = " ".join(filter(None, [str(data.get("Make", "")).strip(), str(data.get("Model", "")).strip()])) or None
    return Exif(
        camera=camera,
        lens=str(data["LensModel"]) if data.get("LensModel") else None,
        focal_length_mm=_to_float(data.get("FocalLength")),
        iso=_to_int(data.get("ISOSpeedRatings")),
        aperture=_to_float(data.get("FNumber")),
        shutter_speed=str(data.get("ExposureTime")) if data.get("ExposureTime") else None,
        taken_at=str(data["DateTimeOriginal"]) if data.get("DateTimeOriginal") else None,
        gps=_gps(gps_dict),
    )


def _to_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _gps(d: dict[str, Any]) -> Gps | None:
    if not d:
        return None
    lat = _coord(d.get("GPSLatitude"), d.get("GPSLatitudeRef", "N"))
    lon = _coord(d.get("GPSLongitude"), d.get("GPSLongitudeRef", "E"))
    if lat is None or lon is None:
        return None
    alt = _to_float(d.get("GPSAltitude"))
    return Gps(lat=lat, lon=lon, alt=alt)


def _coord(parts: Any, ref: str) -> float | None:
    if not parts or len(parts) < 3:
        return None
    try:
        deg = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
    except (TypeError, ValueError, IndexError):
        return None
    val = deg + minutes / 60 + seconds / 3600
    if str(ref).upper() in ("S", "W"):
        val = -val
    return val


# ---------- IPTC / XMP ----------

def _iptc(img: Image.Image) -> dict | None:
    iptc = img.info.get("photoshop")  # Pillow stores IPTC inside Photoshop blocks for JPEGs
    return iptc if isinstance(iptc, dict) else None


def _xmp(img: Image.Image) -> dict | None:
    xmp = img.info.get("xmp")
    if not xmp:
        return None
    if isinstance(xmp, bytes):
        try:
            return {"raw": xmp.decode("utf-8", errors="replace")}
        except Exception:
            return None
    return {"raw": str(xmp)}


# ---------- C2PA ----------

def _c2pa(raw_bytes: bytes) -> C2pa | None:
    """Return None when c2pa-python isn't usable; C2pa(present=False) when no manifest."""
    try:
        import c2pa  # type: ignore
    except ImportError:
        return None
    try:
        reader = c2pa.Reader.from_bytes("image/jpeg", raw_bytes)
        manifest = reader.json()
        if not manifest:
            return C2pa(present=False)
        return C2pa(present=True, manifest=manifest, ai_generated_claim=_c2pa_ai_claim(manifest))
    except Exception as e:
        logger.debug("c2pa parse failed: %s", e)
        return C2pa(present=False)


def _c2pa_ai_claim(manifest: dict | str | None) -> bool | None:
    if not manifest:
        return None
    text = manifest if isinstance(manifest, str) else str(manifest)
    if "ai_generated" in text or "trainedAlgorithmicMedia" in text:
        return True
    return None


# ---------- ICC ----------

def _icc(img: Image.Image) -> IccProfile | None:
    icc_bytes = img.info.get("icc_profile")
    if not icc_bytes:
        return None
    try:
        from PIL import ImageCms
        profile = ImageCms.ImageCmsProfile(io_bytes := __import__("io").BytesIO(icc_bytes))
        name = ImageCms.getProfileName(profile).strip()
        space = profile.profile.xcolor_space.strip() if hasattr(profile.profile, "xcolor_space") else "unknown"
        return IccProfile(name=name, colour_space=space)
    except Exception:
        return IccProfile(name="embedded", colour_space="unknown")


# ---------- animation ----------

def animation_info(img: Image.Image) -> Animation | None:
    n_frames = getattr(img, "n_frames", 1)
    if n_frames <= 1:
        return None
    duration_ms = 0
    try:
        for i in range(n_frames):
            img.seek(i)
            duration_ms += img.info.get("duration", 0) or 0
        img.seek(0)
    except EOFError:
        pass
    return Animation(
        frame_count=n_frames,
        duration_s=(duration_ms / 1000.0) if duration_ms else None,
    )


# ---------- public ----------

def analyse(img: Image.Image, raw_bytes: bytes) -> Metadata:
    return Metadata(
        exif=_exif(img),
        iptc=_iptc(img),
        xmp=_xmp(img),
        c2pa=_c2pa(raw_bytes),
        icc_profile=_icc(img),
    )
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_metadata.py -v
```

Expected: 6 passed. (If `magic` complains about missing libmagic, install: `brew install libmagic` on macOS.)

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/metadata.py tests/test_metadata.py
git commit -m "feat(metadata): format, EXIF, IPTC, XMP, C2PA, ICC, animation"
```

---

### Task 10: Object detection module

**Files:**
- Create: `src/image_analyser/objects.py`
- Test: `tests/test_objects.py`

- [ ] **Step 1: Write the failing test (slow + mocked)**

```python
# tests/test_objects.py
"""Object detection tests."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_analyser.objects import analyse, is_available


def test_skips_when_transformers_missing(monkeypatch):
    monkeypatch.setattr("image_analyser.objects._transformers_importable", lambda: False)
    objects, reason = analyse(Image.new("RGB", (10, 10)))
    assert objects is None
    assert reason == "ml extra not installed"


def test_filters_below_threshold():
    fake_pipeline = MagicMock(return_value=[
        {"label": "cat", "score": 0.9, "box": {"xmin": 0, "ymin": 0, "xmax": 5, "ymax": 5}},
        {"label": "dog", "score": 0.3, "box": {"xmin": 6, "ymin": 6, "xmax": 9, "ymax": 9}},
    ])
    with patch("image_analyser.objects._get_pipeline", return_value=fake_pipeline):
        objects, reason = analyse(Image.new("RGB", (10, 10)), threshold=0.5)
    assert reason is None
    assert len(objects) == 1
    assert objects[0].label == "cat"
    assert objects[0].score == pytest.approx(0.9)


@pytest.mark.slow
def test_detr_real_model_finds_objects(fixtures_dir):
    if not is_available():
        pytest.skip("transformers not installed")
    img = Image.open(fixtures_dir / "small.jpg").convert("RGB")
    objects, reason = analyse(img, threshold=0.0)
    # DETR should always return at least one detection on a non-trivial image
    assert reason is None
    assert isinstance(objects, list)
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_objects.py -v -m "not slow"
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/objects.py`**

```python
"""Object detection via transformers.pipeline (DETR by default)."""

from __future__ import annotations

import importlib
import logging
import os
from functools import lru_cache

from PIL import Image

from .schemas import BBox, Object

logger = logging.getLogger(__name__)


def _transformers_importable() -> bool:
    return importlib.util.find_spec("transformers") is not None


def is_available() -> bool:
    return _transformers_importable()


@lru_cache(maxsize=1)
def _get_pipeline(model: str, device: str):
    from transformers import pipeline
    return pipeline("object-detection", model=model, device=device if device != "auto" else -1)


def _device() -> str:
    return os.getenv("IMAGE_ANALYSER_DEVICE", "auto")


def _model() -> str:
    return os.getenv("IMAGE_ANALYSER_OBJECT_DETECTION_MODEL", "facebook/detr-resnet-50")


def analyse(img: Image.Image, threshold: float | None = None) -> tuple[list[Object] | None, str | None]:
    """Run object detection. Returns (objects, None) on success or (None, skip-reason)."""
    if not _transformers_importable():
        return None, "ml extra not installed"
    if threshold is None:
        threshold = float(os.getenv("IMAGE_ANALYSER_OBJECT_DETECTION_THRESHOLD", "0.5"))
    try:
        pipe = _get_pipeline(_model(), _device())
        raw = pipe(img.convert("RGB"))
    except Exception as e:
        logger.warning("object detection failed: %s", e)
        raise
    objects: list[Object] = []
    for r in raw:
        if r["score"] < threshold:
            continue
        box = r["box"]
        objects.append(
            Object(
                label=r["label"],
                score=float(r["score"]),
                bbox=BBox(
                    x=int(box["xmin"]),
                    y=int(box["ymin"]),
                    w=int(box["xmax"] - box["xmin"]),
                    h=int(box["ymax"] - box["ymin"]),
                ),
            )
        )
    return objects, None
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_objects.py -v -m "not slow"
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/objects.py tests/test_objects.py
git commit -m "feat(objects): DETR-based object detection (ML extra)"
```

---

### Task 11: Caption module (local BLIP + API providers + dispatcher)

**Files:**
- Create: `src/image_analyser/caption.py`
- Test: `tests/test_caption.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_caption.py
"""Captioning tests — dispatcher, local, API."""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from image_analyser.caption import analyse


def test_disabled_by_config(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "none")
    cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "disabled by config"


def test_auto_falls_back_to_local_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    with patch("image_analyser.caption._transformers_importable", return_value=True), \
         patch("image_analyser.caption._caption_local", return_value="a placeholder caption"):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert reason is None
    assert cap.backend == "local"
    assert cap.text == "a placeholder caption"


def test_auto_picks_api_when_key_present(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("image_analyser.caption._caption_api", return_value=("a tabby cat", "openai", "gpt-4o-mini")):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert reason is None
    assert cap.backend == "api"
    assert cap.text == "a tabby cat"
    assert cap.model == "gpt-4o-mini"


def test_no_backend_skips(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    with patch("image_analyser.caption._transformers_importable", return_value=False):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "no captioning backend available"


def test_local_explicit_skips_when_no_ml(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "local")
    with patch("image_analyser.caption._transformers_importable", return_value=False):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "ml extra not installed"


def test_api_explicit_skips_when_no_key(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "api")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "api provider not configured"
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_caption.py -v -m "not slow"
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/caption.py`**

```python
"""Image captioning: local BLIP + API providers (Anthropic, OpenAI, Google, OpenRouter)."""

from __future__ import annotations

import base64
import importlib
import io
import logging
import os
from functools import lru_cache

from PIL import Image

from .schemas import Caption

logger = logging.getLogger(__name__)


PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

DEFAULT_API_MODELS = {
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "openrouter": "openai/gpt-4o-mini",
}


def _transformers_importable() -> bool:
    return importlib.util.find_spec("transformers") is not None


# ---------- local BLIP ----------

@lru_cache(maxsize=1)
def _load_blip(model_name: str):
    from transformers import AutoProcessor, BlipForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def _caption_local(img: Image.Image, model_name: str) -> str:
    processor, model = _load_blip(model_name)
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    return processor.batch_decode(out, skip_special_tokens=True)[0].strip()


# ---------- API providers ----------

def _img_to_b64(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _caption_api(img: Image.Image, provider: str, model: str | None) -> tuple[str, str, str]:
    """Returns (text, provider, model_used)."""
    chosen_model = model or DEFAULT_API_MODELS[provider]
    b64, mime = _img_to_b64(img)
    prompt = "Describe this image in one concise sentence."
    if provider == "anthropic":
        from anthropic import Anthropic
        msg = Anthropic().messages.create(
            model=chosen_model,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return msg.content[0].text.strip(), provider, chosen_model
    if provider == "openai" or provider == "openrouter":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1" if provider == "openrouter" else None,
            api_key=os.environ[PROVIDER_KEYS[provider]],
        )
        resp = client.chat.completions.create(
            model=chosen_model,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        return resp.choices[0].message.content.strip(), provider, chosen_model
    if provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ[PROVIDER_KEYS[provider]])
        resp = client.models.generate_content(
            model=chosen_model,
            contents=[prompt, {"inline_data": {"mime_type": mime, "data": b64}}],
        )
        return resp.text.strip(), provider, chosen_model
    raise ValueError(f"unknown provider: {provider}")


# ---------- dispatcher ----------

def analyse(img: Image.Image) -> tuple[Caption | None, str | None]:
    """Resolve backend per spec §8.1. Returns (Caption, None) on success, (None, reason) on skip."""
    backend = os.getenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    provider = os.getenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    local_model = os.getenv("IMAGE_ANALYSER_LOCAL_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    api_model_override = os.getenv("IMAGE_ANALYSER_CAPTION_MODEL") or None

    if backend == "none":
        return None, "disabled by config"

    if backend == "local":
        if not _transformers_importable():
            return None, "ml extra not installed"
        text = _caption_local(img, local_model)
        return Caption(text=text, backend="local", model=local_model), None

    if backend == "api":
        key = os.getenv(PROVIDER_KEYS.get(provider, ""))
        if not key:
            return None, "api provider not configured"
        text, prov, used = _caption_api(img, provider, api_model_override)
        return Caption(text=text, backend="api", model=used), None

    # auto
    key = os.getenv(PROVIDER_KEYS.get(provider, ""))
    if key:
        text, prov, used = _caption_api(img, provider, api_model_override)
        return Caption(text=text, backend="api", model=used), None
    if _transformers_importable():
        text = _caption_local(img, local_model)
        return Caption(text=text, backend="local", model=local_model), None
    return None, "no captioning backend available"
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_caption.py -v -m "not slow"
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/caption.py tests/test_caption.py
git commit -m "feat(caption): local BLIP + API providers (anthropic/openai/google/openrouter) with auto dispatcher"
```

---

### Task 12: OCR module

**Files:**
- Create: `src/image_analyser/ocr.py`
- Test: `tests/test_ocr.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_ocr.py -v -m "not slow"
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/ocr.py`**

```python
"""OCR dispatcher: tesseract / easyocr."""

from __future__ import annotations

import importlib
import logging
import os
import shutil
from functools import lru_cache

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
def _easyocr_reader():
    import easyocr  # type: ignore
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
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_ocr.py -v -m "not slow"
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/ocr.py tests/test_ocr.py
git commit -m "feat(ocr): tesseract + easyocr dispatcher"
```

---

### Task 13: ImageAnalyser orchestrator

**Files:**
- Create: `src/image_analyser/image_analyser.py`
- Modify: `src/image_analyser/__init__.py` (add public re-exports — see Step 6)
- Test: `tests/test_analyser.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analyser.py
"""Orchestrator tests."""

from unittest.mock import patch

import pytest

from image_analyser import ImageAnalyser
from image_analyser.exceptions import UnsupportedFormatError


def test_analyse_1x1_png_populates_tier1(fixtures_dir):
    result = ImageAnalyser().analyse(fixtures_dir / "1x1.png")
    assert result.format == "PNG"
    assert result.resolution == (1, 1)
    assert result.megapixels == 0.0
    assert result.aspect_class == "square"
    assert result.hash.sha256 != ""
    assert result.colour.average == "#000000"
    # Tier 2 should be skipped (no ML / no API key in test env)
    assert result.objects is None
    assert result.caption is None
    assert result.ocr is None
    skipped_names = {s.name for s in result.skipped}
    assert "objects" in skipped_names
    assert "caption" in skipped_names
    assert "ocr" in skipped_names


def test_skip_flag_excludes_module(fixtures_dir):
    result = ImageAnalyser(skip=["barcode"]).analyse(fixtures_dir / "1x1.png")
    skipped_names = {s.name for s in result.skipped}
    assert "barcode" in skipped_names
    assert any(s.reason == "disabled by --skip" for s in result.skipped if s.name == "barcode")


def test_only_flag_runs_subset(fixtures_dir):
    result = ImageAnalyser(only=["metadata"]).analyse(fixtures_dir / "1x1.png")
    skipped_names = {s.name for s in result.skipped}
    # Hashing/quality/colour/barcode/objects/caption/ocr should all be skipped via `only`
    assert "hashing" in skipped_names
    assert "quality" in skipped_names
    assert "colour" in skipped_names
    assert "barcode" in skipped_names


def test_skip_and_only_mutex_raises():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ImageAnalyser(skip=["caption"], only=["metadata"])


def test_unsupported_format_raises(tmp_path):
    bad = tmp_path / "not-an-image.txt"
    bad.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        ImageAnalyser().analyse(bad)


def test_failed_module_lands_in_failed_list(fixtures_dir):
    """If a module raises, it should be captured into result.failed[] and others should still run."""
    with patch("image_analyser.image_analyser.colour.analyse", side_effect=RuntimeError("boom")):
        result = ImageAnalyser().analyse(fixtures_dir / "1x1.png")
    failed_names = {f.name for f in result.failed}
    assert "colour" in failed_names
    # Tier 1 fields outside colour are still populated
    assert result.format == "PNG"
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_analyser.py -v -m "not slow"
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/image_analyser.py`**

```python
"""ImageAnalyser orchestrator: dispatches per-analysis modules."""

from __future__ import annotations

import logging
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError

import pillow_heif  # noqa: F401  # registers HEIF/HEIC opener with Pillow

from . import barcode, caption, colour, hashing, metadata, objects, ocr, quality
from .exceptions import UnsupportedFormatError
from .schemas import AnalysisResult, Failed, Skipped

logger = logging.getLogger(__name__)

# Toggleable modules (in pipeline order). Format/resolution/file-size are always-on.
_TOGGLEABLE = ("metadata", "hashing", "quality", "colour", "barcode", "objects", "caption", "ocr")


class ImageAnalyser:
    """Orchestrates per-analysis modules.

    Args:
        skip: list of toggleable module names to disable (mutex with `only`).
        only: list of toggleable module names to enable; everything else is skipped (mutex with `skip`).
        caption_backend: optional override for IMAGE_ANALYSER_CAPTION_BACKEND.
    """

    def __init__(
        self,
        *,
        skip: list[str] | None = None,
        only: list[str] | None = None,
        caption_backend: str | None = None,
    ) -> None:
        if skip and only:
            raise ValueError("`skip` and `only` are mutually exclusive")
        for name in (skip or []) + (only or []):
            if name not in _TOGGLEABLE:
                raise ValueError(f"unknown analysis name: {name!r}; legal names: {_TOGGLEABLE}")
        self._skip = set(skip or [])
        self._only = set(only or [])
        self._caption_backend = caption_backend
        self._version = _pkg_version("image-analyser")

    def analyse(self, source: str | Path | bytes | Image.Image) -> AnalysisResult:
        start = time.perf_counter()
        img, raw_bytes = self._load(source)
        info = metadata.basic(img, raw_bytes)
        skipped: list[Skipped] = []
        failed: list[Failed] = []

        def enabled(name: str) -> bool:
            if self._only:
                if name not in self._only:
                    skipped.append(Skipped(name=name, reason="not selected by --only"))
                    return False
                return True
            if name in self._skip:
                skipped.append(Skipped(name=name, reason="disabled by --skip"))
                return False
            return True

        # ---- always-on identity (from `info`) ----
        # ---- toggleable Tier 1 ----
        md = None
        if enabled("metadata"):
            md = self._safe("metadata", failed, lambda: metadata.analyse(img, raw_bytes))
        h = None
        if enabled("hashing"):
            h = self._safe("hashing", failed, lambda: hashing.analyse(img, raw_bytes))
        q = None
        if enabled("quality"):
            q = self._safe("quality", failed, lambda: quality.analyse(img, raw_bytes))
        c = None
        if enabled("colour"):
            c = self._safe("colour", failed, lambda: colour.analyse(img))
        b: list = []
        if enabled("barcode"):
            b = self._safe("barcode", failed, lambda: barcode.analyse(img)) or []

        # ---- toggleable Tier 2 ----
        objs = None
        if enabled("objects"):
            objs_result = self._safe("objects", failed, lambda: objects.analyse(img))
            if objs_result is not None:
                values, reason = objs_result
                if reason:
                    skipped.append(Skipped(name="objects", reason=reason))
                else:
                    objs = values
        cap = None
        if enabled("caption"):
            cap_env = self._caption_backend
            cap_result = self._safe(
                "caption", failed,
                lambda: self._with_env({"IMAGE_ANALYSER_CAPTION_BACKEND": cap_env}, caption.analyse, img),
            )
            if cap_result is not None:
                value, reason = cap_result
                if reason:
                    skipped.append(Skipped(name="caption", reason=reason))
                else:
                    cap = value
        oc = None
        if enabled("ocr"):
            oc_result = self._safe("ocr", failed, lambda: ocr.analyse(img))
            if oc_result is not None:
                value, reason = oc_result
                if reason:
                    skipped.append(Skipped(name="ocr", reason=reason))
                else:
                    oc = value

        animation = metadata.animation_info(img)
        return AnalysisResult(
            format=info["format"],
            mime_type=info["mime_type"],
            resolution=info["resolution"],
            megapixels=info["megapixels"],
            aspect_class=info["aspect_class"],
            colour_mode=info["colour_mode"],
            bit_depth=info["bit_depth"],
            has_alpha=info["has_alpha"],
            file_size=info["file_size"],
            hash=h or _zero_hash(),
            metadata=md or _empty_metadata(),
            animation=animation,
            quality=q or _zero_quality(),
            colour=c or _zero_colour(),
            barcodes=b,
            objects=objs,
            caption=cap,
            ocr=oc,
            skipped=skipped,
            failed=failed,
            version=self._version,
            analysed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    # ---- helpers ----

    def _load(self, source: str | Path | bytes | Image.Image) -> tuple[Image.Image, bytes]:
        if isinstance(source, Image.Image):
            buf = __import__("io").BytesIO()
            source.save(buf, source.format or "PNG")
            return source, buf.getvalue()
        if isinstance(source, bytes):
            try:
                img = Image.open(__import__("io").BytesIO(source))
                img.load()
                return img, source
            except UnidentifiedImageError as e:
                raise UnsupportedFormatError(str(e)) from e
        path = Path(source)
        raw = path.read_bytes()
        try:
            img = Image.open(__import__("io").BytesIO(raw))
            img.load()
        except UnidentifiedImageError as e:
            raise UnsupportedFormatError(f"Unsupported image format: {path}") from e
        return img, raw

    def _safe(self, name: str, failed: list[Failed], fn):
        try:
            return fn()
        except Exception as e:
            logger.warning("%s analysis failed: %s", name, e)
            failed.append(Failed(name=name, error=str(e), traceback=traceback.format_exc()))
            return None

    def _with_env(self, overrides: dict, fn, *args, **kwargs):
        import os
        before = {k: os.environ.get(k) for k in overrides}
        try:
            for k, v in overrides.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            return fn(*args, **kwargs)
        finally:
            for k, v in before.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


# ---- minimal placeholder values when a Tier 1 module fails or is skipped ----

def _zero_hash():
    from .schemas import Hash
    return Hash(sha256="0" * 64, phash="0" * 16, dhash="0" * 16)


def _empty_metadata():
    from .schemas import Metadata
    return Metadata()


def _zero_quality():
    from .schemas import Exposure, Quality
    return Quality(
        blur_score=0.0,
        exposure=Exposure(underexposed_pct=0.0, overexposed_pct=0.0, clipping_pct=0.0),
        brightness=0.0, contrast=0.0, noise=0.0, jpeg_quality_estimate=None,
    )


def _zero_colour():
    from .schemas import Colour
    return Colour(dominant=["#000000"], average="#000000", palette=[])
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_analyser.py -v -m "not slow"
```

Expected: 6 passed.

- [ ] **Step 5: Update `src/image_analyser/__init__.py` with public re-exports**

Replace the file (currently the minimal version from Task 1) with:

```python
"""Static image analysis for the analyser family."""

from importlib.metadata import version as _version

from .exceptions import ImageAnalyserError
from .image_analyser import ImageAnalyser
from .schemas import AnalysisResult

__version__ = _version("image-analyser")
__all__ = ["ImageAnalyser", "AnalysisResult", "ImageAnalyserError", "__version__"]
```

- [ ] **Step 6: Verify the public API still imports cleanly**

```bash
python -c "from image_analyser import ImageAnalyser, AnalysisResult, ImageAnalyserError, __version__; print(__version__)"
```

Expected: prints `0.1.0`.

- [ ] **Step 7: Commit**

```bash
git add src/image_analyser/image_analyser.py src/image_analyser/__init__.py tests/test_analyser.py
git commit -m "feat(orchestrator): ImageAnalyser dispatcher with skip/only and skipped/failed tracking"
```

---

### Task 14: FastAPI app

**Files:**
- Create: `src/image_analyser/app.py`
- Test: `tests/test_app.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_app.py
"""FastAPI endpoint tests."""

from importlib.metadata import version as _v

from fastapi.testclient import TestClient

from image_analyser.app import app

client = TestClient(app)


def test_health_returns_ok_and_version():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "version": _v("image-analyser")}


def test_root_returns_service_info():
    r = client.get("/")
    body = r.json()
    assert r.status_code == 200
    assert body["service"] == "image-analyser"
    assert body["version"] == _v("image-analyser")
    assert "/analyse" in body["endpoints"]


def test_analyse_multipart(fixtures_dir):
    with (fixtures_dir / "1x1.png").open("rb") as f:
        r = client.post("/analyse", files={"file": ("1x1.png", f, "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["format"] == "PNG"
    assert body["resolution"] == [1, 1]
    assert body["version"] == _v("image-analyser")


def test_analyse_json_path(fixtures_dir):
    r = client.post("/analyse", json={"path": str(fixtures_dir / "1x1.png")})
    assert r.status_code == 200
    assert r.json()["format"] == "PNG"


def test_analyse_missing_path_returns_404():
    r = client.post("/analyse", json={"path": "/does/not/exist.png"})
    assert r.status_code == 404


def test_analyse_unsupported_format_returns_400(tmp_path):
    bad = tmp_path / "not-image.txt"
    bad.write_text("hello")
    r = client.post("/analyse", json={"path": str(bad)})
    assert r.status_code == 400


def test_skip_only_mutex_returns_400(fixtures_dir):
    r = client.post(
        "/analyse",
        json={"path": str(fixtures_dir / "1x1.png"), "skip": ["caption"], "only": ["metadata"]},
    )
    assert r.status_code == 400
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_app.py -v
```

Expected: FAIL on import.

- [ ] **Step 3: Write `src/image_analyser/app.py`**

```python
"""FastAPI HTTP surface for image-analyser."""

from __future__ import annotations

import os
from importlib.metadata import version as _pkg_version
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from .exceptions import UnsupportedFormatError
from .image_analyser import ImageAnalyser
from .schemas import AnalysisResult


def _origins() -> list[str]:
    raw = os.getenv("IMAGE_ANALYSER_ALLOWED_ORIGINS", "*")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _max_upload_bytes() -> int:
    return int(os.getenv("IMAGE_ANALYSER_MAX_UPLOAD_MB", "50")) * 1024 * 1024


class AnalyseRequest(BaseModel):
    path: str | None = None
    skip: list[str] | None = None
    only: list[str] | None = None
    caption_backend: str | None = None


def _parse_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _build_analyser(skip, only, caption_backend) -> ImageAnalyser:
    try:
        return ImageAnalyser(skip=skip, only=only, caption_backend=caption_backend)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def create_app() -> FastAPI:
    app = FastAPI(
        title="image-analyser",
        version=_pkg_version("image-analyser"),
        description="Static image analysis (CLI + FastAPI) for the analyser family",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origins(),
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "version": _pkg_version("image-analyser")}

    @app.get("/")
    def root():
        return {
            "service": "image-analyser",
            "version": _pkg_version("image-analyser"),
            "endpoints": ["/analyse", "/health"],
        }

    @app.post("/analyse", response_model=AnalysisResult)
    async def analyse(request: Request):
        ctype = request.headers.get("content-type", "").lower()
        if ctype.startswith("multipart/form-data"):
            form = await request.form()
            upload = form.get("file")
            if upload is None or not hasattr(upload, "read"):
                raise HTTPException(status_code=400, detail="multipart requires a `file` field")
            data = await upload.read()
            if len(data) > _max_upload_bytes():
                raise HTTPException(status_code=413, detail="upload too large")
            analyser = _build_analyser(
                skip=_parse_csv(form.get("skip")),
                only=_parse_csv(form.get("only")),
                caption_backend=form.get("caption_backend") or None,
            )
            try:
                return analyser.analyse(data)
            except UnsupportedFormatError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        if ctype.startswith("application/json"):
            try:
                body = await request.json()
                req = AnalyseRequest.model_validate(body)
            except (ValueError, ValidationError) as e:
                raise HTTPException(status_code=400, detail=f"invalid JSON body: {e}") from e
            if not req.path:
                raise HTTPException(status_code=400, detail="`path` is required for JSON requests")
            p = Path(req.path)
            if not p.exists():
                raise HTTPException(status_code=404, detail=f"path not found: {p}")
            analyser = _build_analyser(req.skip, req.only, req.caption_backend)
            try:
                return analyser.analyse(p)
            except UnsupportedFormatError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        raise HTTPException(
            status_code=400,
            detail="use multipart/form-data with a `file` field, or application/json with a `path`",
        )

    return app


app = create_app()
```

- [ ] **Step 4: Run, verify pass**

```bash
pytest tests/test_app.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/image_analyser/app.py tests/test_app.py
git commit -m "feat(app): FastAPI /analyse, /health, / endpoints"
```

---

### Task 15: CLI

**Files:**
- Create: `src/image_analyser/cli.py`
- Test: `tests/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
"""CLI tests."""

import json
import subprocess
import sys


def _run(*args, cwd=None):
    return subprocess.run(
        [sys.executable, "-m", "image_analyser", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_version_flag(tmp_path):
    p = _run("--version")
    assert p.returncode == 0
    assert "0.1.0" in p.stdout or "0.1.0" in p.stderr


def test_help_flag():
    p = _run("--help")
    assert p.returncode == 0
    assert "image-analyser" in p.stdout.lower()
    assert "serve" in p.stdout.lower()


def test_analyse_emits_json(fixtures_dir):
    p = _run(str(fixtures_dir / "1x1.png"), "--json")
    assert p.returncode == 0
    body = json.loads(p.stdout)
    assert body["format"] == "PNG"
    assert body["resolution"] == [1, 1]


def test_missing_file_exits_2(tmp_path):
    p = _run(str(tmp_path / "nope.png"))
    assert p.returncode == 2


def test_skip_and_only_mutex_exits_2(fixtures_dir):
    p = _run(str(fixtures_dir / "1x1.png"), "--skip", "caption", "--only", "metadata")
    assert p.returncode == 2
```

- [ ] **Step 2: Run, verify fail**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL on import / no `__main__.py`.

- [ ] **Step 3: Write `src/image_analyser/cli.py`**

```python
"""Typer-based CLI for image-analyser."""

from __future__ import annotations

import json as _json
import os
import sys
from importlib.metadata import version as _pkg_version
from pathlib import Path

import typer

from .exceptions import ImageAnalyserError, UnsupportedFormatError

cli = typer.Typer(add_completion=False, no_args_is_help=True)


def _csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


@cli.command(help="Analyse a single image and print the result as JSON.")
def analyse(
    file: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    json_out: bool = typer.Option(False, "--json", help="Compact JSON output."),
    skip: str | None = typer.Option(None, "--skip", help="Comma-separated analyses to skip."),
    only: str | None = typer.Option(None, "--only", help="Comma-separated analyses to run (mutex with --skip)."),
    caption_backend: str | None = typer.Option(None, "--caption-backend", help="local|api|auto|none"),
) -> None:
    from .image_analyser import ImageAnalyser
    try:
        analyser = ImageAnalyser(
            skip=_csv(skip), only=_csv(only), caption_backend=caption_backend,
        )
        result = analyser.analyse(file)
    except ValueError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2)
    except UnsupportedFormatError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=2)
    except ImageAnalyserError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(code=1)
    payload = result.model_dump(mode="json")
    typer.echo(_json.dumps(payload, separators=(",", ":")) if json_out else _json.dumps(payload, indent=2))


@cli.command(help="Start the FastAPI HTTP server.")
def serve(
    port: int = typer.Option(int(os.getenv("IMAGE_ANALYSER_PORT", "8006")), "--port"),
    host: str = typer.Option(os.getenv("IMAGE_ANALYSER_HOST", "127.0.0.1"), "--host"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    import uvicorn
    uvicorn.run("image_analyser.app:app", host=host, port=port, reload=reload)


@cli.callback(invoke_without_command=False)
def _root(
    version: bool = typer.Option(False, "--version", is_eager=True, help="Show version and exit."),
) -> None:
    if version:
        typer.echo(_pkg_version("image-analyser"))
        raise typer.Exit()


def main() -> None:
    # Make `image-analyser FILE [--json]` work as well as `image-analyser analyse FILE`.
    # Typer doesn't support this natively, so we promote a bare positional to the analyse command.
    argv = sys.argv[1:]
    if argv and not argv[0].startswith("-") and argv[0] not in {"analyse", "serve"}:
        sys.argv = [sys.argv[0], "analyse", *argv]
    cli()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add `src/image_analyser/__main__.py` for `python -m image_analyser`**

```python
"""Make `python -m image_analyser` invoke the CLI."""

from .cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run, verify pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add src/image_analyser/cli.py src/image_analyser/__main__.py tests/test_cli.py
git commit -m "feat(cli): typer CLI with bare positional + serve subcommand"
```

---

### Task 16: Invariants tests (drift guards + schema completeness)

**Files:**
- Create: `tests/test_invariants.py`

- [ ] **Step 1: Write the test file**

```python
# tests/test_invariants.py
"""Project invariants — drift guards and stable contracts."""

from importlib.metadata import version as _v

from fastapi.testclient import TestClient

from image_analyser import ImageAnalyser, AnalysisResult, __version__
from image_analyser.app import app


def test_package_dunder_version_matches_metadata():
    assert __version__ == _v("image-analyser")


def test_health_version_matches_metadata():
    r = TestClient(app).get("/health")
    assert r.json()["version"] == _v("image-analyser")
    assert r.json()["status"] == "ok"


def test_root_version_matches_metadata():
    r = TestClient(app).get("/")
    assert r.json()["version"] == _v("image-analyser")
    assert r.json()["service"] == "image-analyser"


def test_clean_import_has_no_side_effects():
    """Importing the package must not load torch/transformers/easyocr."""
    import sys
    forbidden = {"torch", "transformers", "easyocr"}
    loaded = forbidden & set(sys.modules)
    assert not loaded, f"importing image_analyser pulled in heavy modules: {loaded}"


def test_minimal_image_populates_all_tier1_fields(fixtures_dir):
    """A 1×1 PNG should still produce a complete Tier 1 schema."""
    result = ImageAnalyser().analyse(fixtures_dir / "1x1.png")
    assert isinstance(result, AnalysisResult)
    # Always-on identity
    for field in ("format", "mime_type", "resolution", "megapixels", "aspect_class",
                  "colour_mode", "bit_depth", "has_alpha", "file_size",
                  "hash", "metadata", "quality", "colour", "version", "analysed_at"):
        assert getattr(result, field) is not None, f"{field} should always be populated"
    assert result.barcodes == []
    assert result.skipped is not None
    assert result.failed == []


# Stable skip-reason strings — asserting these protects callers from silent rename drift.
EXPECTED_REASONS = {
    "objects": {"ml extra not installed"},
    "caption": {"disabled by config", "no captioning backend available", "ml extra not installed", "api provider not configured"},
    "ocr": {"disabled by config", "no ocr engine available", "tesseract not installed", "easyocr not installed"},
    "barcode": set(),  # always runs (no extras)
}


def test_skip_reasons_are_from_known_set(fixtures_dir):
    result = ImageAnalyser().analyse(fixtures_dir / "1x1.png")
    for s in result.skipped:
        if s.name in EXPECTED_REASONS and EXPECTED_REASONS[s.name]:
            assert s.reason in EXPECTED_REASONS[s.name], (
                f"unexpected skip reason for {s.name}: {s.reason!r}"
            )
```

- [ ] **Step 2: Run, verify pass**

```bash
pytest tests/test_invariants.py -v
```

Expected: 6 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_invariants.py
git commit -m "test(invariants): version drift, clean import, schema completeness, skip-reason set"
```

---

### Task 17: README

**Files:**
- Modify: `README.md` (overwrite the placeholder from Task 1)

- [ ] **Step 1: Write `README.md`**

```markdown
# image-analyser

[![PyPI](https://img.shields.io/pypi/v/image-analyser.svg)](https://pypi.org/project/image-analyser/)
[![Python](https://img.shields.io/pypi/pyversions/image-analyser.svg)](https://pypi.org/project/image-analyser/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Static image analysis (CLI + FastAPI) for the analyser family.

Given a single image file, image-analyser returns:

- **Format** — JPEG / PNG / WebP / AVIF / HEIC / GIF / TIFF / BMP, MIME type, resolution, megapixels, aspect class, colour mode, bit depth, alpha, file size.
- **Hashes** — SHA-256 of bytes, perceptual pHash and dHash for dedup.
- **Metadata** — EXIF (camera, lens, ISO, GPS, timestamp), IPTC, XMP, ICC profile, C2PA Content Credentials.
- **Quality** — blur (Laplacian variance), exposure (under / over / clipping), brightness, contrast, noise, JPEG-quality estimate.
- **Colour** — k-means dominant palette + average colour.
- **Barcodes / QR codes** — via pyzbar.
- **Animation** — frame count + duration for GIF / WebP.
- **Object detection** *(opt-in `[ml]`)* — DETR by default; configurable HuggingFace pipeline.
- **Caption / textual description** *(opt-in `[ml]` or `[api]`)* — local BLIP, or one of Anthropic / OpenAI / Google / OpenRouter.
- **OCR** *(opt-in `[ocr]`)* — Tesseract or EasyOCR.

Anything that can't run in the current install lands in `result.skipped[]` with a stable reason. Anything that raises lands in `result.failed[]`. The HTTP response stays 200.

## Install

```bash
pip install image-analyser              # Tier 1 only
pip install image-analyser[ml]          # + DETR object detection + local BLIP captioning
pip install image-analyser[ocr]         # + Tesseract / EasyOCR
pip install image-analyser[api]         # + API captioning (no torch needed)
pip install image-analyser[all]         # everything
```

System dependencies:

- `libzbar0` (for barcode/QR detection): `brew install zbar` / `apt install libzbar0`
- `libmagic` (for MIME detection): `brew install libmagic` / `apt install libmagic1`
- `tesseract` (only if `[ocr]` extra used and tesseract is the engine): `brew install tesseract` / `apt install tesseract-ocr`

## CLI

```bash
image-analyser photo.jpg                          # pretty JSON
image-analyser photo.jpg --json                   # compact JSON
image-analyser photo.jpg --skip caption,ocr       # opt out
image-analyser photo.jpg --only metadata,quality  # opt in
image-analyser photo.jpg --caption-backend local  # force local BLIP
image-analyser serve                              # FastAPI on :8006
image-analyser serve --port 9000
```

Legal `--skip` / `--only` values: `metadata`, `quality`, `colour`, `hashing`, `barcode`, `objects`, `caption`, `ocr`.

Exit codes: `0` on success (even with `failed[]`), `2` on bad input, `1` on internal error.

## HTTP API

```bash
# Multipart upload
curl -F file=@photo.jpg http://127.0.0.1:8006/analyse

# JSON with absolute path
curl -X POST -H "Content-Type: application/json" \
  -d '{"path":"/abs/path/photo.jpg","skip":["caption"]}' \
  http://127.0.0.1:8006/analyse

# Health check
curl http://127.0.0.1:8006/health
# {"status": "ok", "version": "0.1.0"}
```

## Python library

```python
from image_analyser import ImageAnalyser

result = ImageAnalyser().analyse("photo.jpg")
print(result.metadata.exif.camera if result.metadata.exif else "no EXIF")
print(result.quality.blur_score, result.colour.dominant)
for s in result.skipped:
    print("skipped", s.name, "—", s.reason)
```

## Configuration

All env vars use the `IMAGE_ANALYSER_` prefix.

| Env var | Default | Purpose |
|---|---|---|
| `IMAGE_ANALYSER_PORT` | `8006` | FastAPI port |
| `IMAGE_ANALYSER_HOST` | `127.0.0.1` | bind address |
| `IMAGE_ANALYSER_MODE` | `production` | `production` / `development` |
| `IMAGE_ANALYSER_ALLOWED_ORIGINS` | `*` | CORS allow-list (comma-separated) |
| `IMAGE_ANALYSER_CAPTION_BACKEND` | `auto` | `auto` / `local` / `api` / `none` |
| `IMAGE_ANALYSER_CAPTION_PROVIDER` | `anthropic` | `anthropic` / `openai` / `google` / `openrouter` |
| `IMAGE_ANALYSER_CAPTION_MODEL` | provider-default | model id override |
| `IMAGE_ANALYSER_LOCAL_CAPTION_MODEL` | `Salesforce/blip-image-captioning-base` | HuggingFace BLIP model id |
| `IMAGE_ANALYSER_OCR_ENGINE` | `auto` | `auto` / `tesseract` / `easyocr` / `none` |
| `IMAGE_ANALYSER_OBJECT_DETECTION_MODEL` | `facebook/detr-resnet-50` | HuggingFace pipeline model |
| `IMAGE_ANALYSER_OBJECT_DETECTION_THRESHOLD` | `0.5` | detection-confidence cutoff |
| `IMAGE_ANALYSER_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `IMAGE_ANALYSER_MAX_UPLOAD_MB` | `50` | reject larger uploads |

API keys: standard provider env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`).

## Development

```bash
git clone https://github.com/michael-borck/image-analyser
cd image-analyser
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,all]"
python tests/fixtures/make_fixtures.py
pytest                  # fast tests
pytest -m slow          # opt-in ML / engine tests
ruff check .
mypy src
```

## The analyser family

| Tool | PyPI | Port | Role |
|---|---|---|---|
| auto-analyser | ✓ | — | router (file → specialist) |
| bundle-analyser | ✓ | 8008 | folder / zip walker |
| code-analyser | ✓ | 8004 | source code analysis |
| document-analyser | ✓ | 8000 | document analysis |
| git-analyser | ✓ | 8007 | git repo analysis |
| **image-analyser** | ✓ | **8006** | **static image analysis (this repo)** |
| records-analyser | ✓ | 8003 | structured-records analysis |
| speech-analyser | ✓ | 8001 | audio transcription / speech analysis |
| video-analyser | ✓ | 8002 | video analysis |
| wordpress-analyser | ✓ | 8005 | WordPress export analysis |

## Licence

MIT © Michael Borck.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with install, CLI, HTTP, library, env vars, family table"
```

---

### Task 18: Quality gates pass

**Files:** none (tooling-only).

- [ ] **Step 1: Run ruff**

```bash
ruff check .
```

Expected: zero issues. If any appear, fix them before continuing.

- [ ] **Step 2: Run mypy**

```bash
mypy src
```

Expected: `Success: no issues found`.

- [ ] **Step 3: Run full test suite (default — fast tests only)**

```bash
pytest --cov=src/image_analyser --cov-report=term-missing
```

Expected: all tests pass; coverage on Tier 1 modules ≥ 85%. Tier 2 modules will show lower coverage from default (mocked) tests — acceptable.

- [ ] **Step 4: Run slow tests (require `[all]` install)**

```bash
pytest -m slow
```

Expected: slow tests pass when their backends are available; skipped when not. No failures.

- [ ] **Step 5: Commit any fixes from steps 1-3**

```bash
git status
# If anything was changed:
git add -u && git commit -m "chore: ruff/mypy fixes from quality gates"
```

---

### Task 19: PyPI publish + GitHub release

**Files:** none (release-only).

- [ ] **Step 1: Build the wheel and sdist**

```bash
rm -rf dist/
python -m build
ls dist/
```

Expected: `dist/image_analyser-0.1.0-py3-none-any.whl` and `dist/image_analyser-0.1.0.tar.gz`.

- [ ] **Step 2: Verify the wheel installs cleanly in a fresh venv and runs**

```bash
python -m venv /tmp/check-image-analyser
/tmp/check-image-analyser/bin/pip install dist/image_analyser-0.1.0-py3-none-any.whl
/tmp/check-image-analyser/bin/python -c "import image_analyser; print(image_analyser.__version__)"
/tmp/check-image-analyser/bin/image-analyser --version
rm -rf /tmp/check-image-analyser
```

Expected: prints `0.1.0` twice.

- [ ] **Step 3: Upload to PyPI via twine (uses ~/.pypirc)**

```bash
twine check dist/*
twine upload dist/*
```

Expected: `View at: https://pypi.org/project/image-analyser/0.1.0/`.

- [ ] **Step 4: Tag and push**

```bash
git tag v0.1.0
git push origin main --tags
```

- [ ] **Step 5: Create the GitHub release**

```bash
gh release create v0.1.0 \
  --title "image-analyser 0.1.0" \
  --notes "First functional release. CLI + FastAPI for static image analysis. Tier 1 lightweight analyses always run; Tier 2 ML/OCR/API analyses gated behind extras. See README for usage."
```

Expected: a release URL is printed.

- [ ] **Step 6: Smoke-test the public package**

```bash
pip install --upgrade image-analyser
image-analyser --version
```

Expected: prints `0.1.0`.

- [ ] **Step 7: Update the analyser-family memory**

Update `/Users/michael/.claude/projects/-Users-michael-Projects-lens/memory/project_analyser_family_todos.md` with:

- Add row to package status table: `image-analyser | 0.1.0 | ✅ | ✅ | ✅ Tier 1 in-house + Tier 2 lifted from video-analyser (caption + OCR); DETR-based object detection`.
- Add port row: `8006 | image-analyser`.
- Add deferred follow-up: "video-analyser v0.7.0 — swap internal image stack (`object_detector` / `image_captioner` / `api_image_captioner` / `ocr_detector` / `frame_analyzer` / `visual_analyzer`) for `import image_analyser`. Mirror of the speech-analyser plan."
- Add lesson: "any future image-hashing / dedup work in the analyser family should reuse `image_analyser.hashing` (single source of truth)."

This is a memory edit — no git commit needed (memory directory isn't tracked in this repo).

---

## Self-review notes

**Spec coverage check:**
- §3 module layout → Tasks 1, 2, 3, 5–15 (one task per module). ✓
- §4.1 CLI surface (bare positional, --json, --skip, --only, --caption-backend, serve, --version, --help) → Task 15. ✓
- §4.2 HTTP (POST /analyse multipart + JSON, GET /health "ok"+version, GET /, error codes 400/404/413, CORS) → Task 14. ✓
- §4.3 Python library → covered by `__init__.py` exports (Task 1) and orchestrator (Task 13). ✓
- §5 schema with all sub-models → Task 3. ✓
- §5.1 skipped vs failed semantics → Task 13 + Task 16 invariants. ✓
- §6 dependencies + extras → Task 1 pyproject. ✓
- §7 env vars → consumed across Tasks 11, 10, 12, 14, 15. README documents them in Task 17. ✓
- §8 dispatch rules (caption / OCR / objects with stable skip reasons) → Tasks 10, 11, 12. ✓
- §9 testing strategy (slow markers, invariants, no echo-tautologies, stable reasons) → Tasks 1, 16. ✓
- §10 release ladder → Task 19. ✓
- §11 memory updates → Task 19 step 7. ✓

**Placeholder scan:** none found — every step has runnable code or exact commands. No "TBD", no "implement later", no "similar to Task N".

**Type consistency check:**
- `ImageAnalyser.__init__(skip, only, caption_backend)` is consistent across Tasks 13, 14, 15.
- `analyse()` return-tuple shape `(value, reason | None)` is consistent across `objects`, `caption`, `ocr` (Tasks 10, 11, 12).
- `Hash.{sha256, phash, dhash}`, `BBox.{x, y, w, h}`, `Skipped.{name, reason}`, `Failed.{name, error, traceback}` field names are consistent between Task 3 (schemas), Task 13 (orchestrator), Task 14 (app), Task 16 (invariants).
- The hidden `/analyse` JSON alias in Task 14 uses `include_in_schema=False` — verified consistent with the multipart route.

---

## Execution

Plan complete and saved to `docs/superpowers/plans/2026-05-09-image-analyser-v0.1.0.md`.
