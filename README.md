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
