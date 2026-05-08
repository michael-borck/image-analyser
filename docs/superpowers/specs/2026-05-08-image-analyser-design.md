# image-analyser v0.1.0 — design spec

**Date:** 2026-05-08
**Status:** Brainstorming complete; ready for implementation plan
**Author:** michael (with Claude)

## 1. Context

`image-analyser` is the next member of the analyser family (under `/Users/michael/Projects/lens/`). Family pattern: a single-file-in, signals-out tool exposing both a CLI and a FastAPI `/analyse` endpoint, plus a clean Python class for library use. `image-analyser` analyses static images, returning a structured set of metadata, quality, colour, hash, and ML-derived signals.

Initial code base for the ML modules is lifted from `video-analyser/src/video_analyser/analysis/` — specifically `image_captioner.py` (local BLIP + API dispatcher), `api_image_captioner.py` (Anthropic / OpenAI / Google / OpenRouter providers), and `ocr_detector.py` (Tesseract / EasyOCR). Tier 1 lightweight analyses are written from scratch. **Object detection is a fresh implementation** (not a lift): `video-analyser/object_detector.py` is heuristic-only with a "can be replaced with YOLOv5 later" placeholder; image-analyser ships a real `transformers.pipeline("object-detection", ...)` backend.

`video-analyser` itself stays unchanged in this work; a follow-up release (v0.7.0 of `video-analyser`) will swap its internal image stack for `import image_analyser` — mirror of the planned `speech-analyser` library swap.

## 2. Goals & non-goals

### Goals (v0.1.0)
- Functional first release. Published to PyPI as `image-analyser` 0.1.0 (not a stub).
- Mirror `speech-analyser`'s layout, env-var prefix, `/health` endpoint shape, and `importlib.metadata` version sourcing.
- Provide three usable surfaces: CLI, FastAPI HTTP, and Python library class.
- "Do what you can" default: run every analysis whose dependencies are available; report what was skipped or failed in the response.
- Heavy ML deps gated behind extras so the default install stays small.

### Non-goals (v0.1.0)
- Batch / directory walk. Single file in, single result out — `bundle-analyser` is the canonical batch home, routing per-file via `auto-analyser`.
- Tier 3 analyses: scene/place classification, NSFW/safety, aesthetic scoring, face count, heuristic AI-generation detection (C2PA *parsing* is in scope; heuristic detection is not), logo/brand detection, CLIP embeddings.
- Modifying `video-analyser`. Its image stack stays in place; the swap is a separate, future release.

## 3. Architecture

Single Python package `image_analyser` with one orchestrator class `ImageAnalyser` that calls per-analysis modules. Each module is independently importable, has a single `analyse(...)` function returning a typed Pydantic sub-model (or skips with a stable reason string), and has its own test file.

Names: repo + folder + PyPI = `image-analyser` (hyphenated); Python import = `image_analyser` (underscored); CLI command = `image-analyser`. Australian-British spelling throughout (analyse, colour, organise).

```
image-analyser/
├── LICENSE                           # MIT
├── README.md
├── pyproject.toml                    # name = "image-analyser"
├── src/image_analyser/
│   ├── __init__.py                   # __version__ = importlib.metadata.version("image-analyser")
│   ├── app.py                        # FastAPI: /analyse, /health, /
│   ├── cli.py                        # typer-based; bare positional + serve subcommand
│   ├── exceptions.py                 # ImageAnalyserError
│   ├── schemas.py                    # AnalysisResult + sub-models (Pydantic v2)
│   ├── image_analyser.py             # ImageAnalyser class — orchestrator + dispatch
│   ├── metadata.py                   # format / resolution / mode / EXIF / IPTC / XMP / C2PA / ICC
│   ├── quality.py                    # blur (Laplacian variance) / exposure / brightness / contrast / noise / JPEG-Q
│   ├── colour.py                     # palette (k-means) + average + dominant
│   ├── hashing.py                    # SHA256 + pHash + dHash (imagehash)
│   ├── barcode.py                    # pyzbar (QR + 1D barcodes)
│   ├── objects.py                    # NEW: transformers.pipeline("object-detection", DETR by default) (ML)
│   ├── caption.py                    # lifted dispatcher: local BLIP + API providers (ML / API)
│   └── ocr.py                        # lifted from video-analyser/ocr_detector.py (Tesseract / EasyOCR)
└── tests/
    ├── conftest.py
    ├── fixtures/                     # sample JPEG / PNG / HEIC / AVIF / WebP / GIF (small, MIT-licenced fixtures)
    ├── test_analyser.py              # orchestrator behaviour: what runs, what skips
    ├── test_app.py                   # FastAPI endpoints
    ├── test_cli.py                   # CLI smoke + --json + serve
    ├── test_metadata.py
    ├── test_quality.py
    ├── test_colour.py
    ├── test_hashing.py
    ├── test_barcode.py
    ├── test_objects.py               # @pytest.mark.slow (loads ML model)
    ├── test_caption.py               # @pytest.mark.slow + API mocked
    ├── test_ocr.py                   # @pytest.mark.slow + engine-conditional
    └── test_invariants.py            # importability smoke, version drift guard, schema completeness
```

## 4. Public surfaces

### 4.1 CLI

Entry point: `image-analyser` (hyphenated, matches PyPI name). Powered by Typer.

```
image-analyser FILE                                # pretty-printed JSON to stdout, runs everything available
image-analyser FILE --json                         # compact single-line JSON
image-analyser FILE --skip caption,ocr             # opt out of named analyses (mutex with --only)
image-analyser FILE --only metadata,quality        # opt in to a subset (mutex with --skip)
image-analyser FILE --caption-backend local|api|auto|none
image-analyser serve                               # FastAPI on $IMAGE_ANALYSER_PORT (default 8006)
image-analyser serve --port 9000
image-analyser --version
image-analyser --help
```

Legal `--skip` / `--only` values: `metadata`, `quality`, `colour`, `hashing`, `barcode`, `objects`, `caption`, `ocr`. (Format / resolution / file-size are always-on identity fields, not toggleable.)

Exit codes:
- `0` — success (even if individual analyses landed in `failed[]` or `skipped[]`).
- `2` — bad CLI input (file missing, unsupported format, mutex-flag violation).
- `1` — orchestrator-level internal error (a bug in image-analyser itself).

### 4.2 HTTP API

`POST /analyse`
- `multipart/form-data`: `file=@foo.jpg` (preferred for arbitrary clients).
- `application/json`: `{"path": "/abs/path/foo.jpg", "skip": [...], "only": [...], "caption_backend": "auto"}` (preferred for trusted local callers).
- Response: `200 OK` with `AnalysisResult` JSON.
- `400` — unsupported format, mutex-flag violation, malformed body.
- `404` — `path` does not exist (json-mode only).
- `413` — upload exceeds `IMAGE_ANALYSER_MAX_UPLOAD_MB`.
- `500` — orchestrator-level bug *only*. Individual analysis failures land in `failed[]` with a 200 response.

`GET /health` → `{"status": "ok", "version": "<importlib.metadata.version>"}` — matches family standard.

`GET /` → `{"service": "image-analyser", "version": "...", "endpoints": ["/analyse", "/health"]}`.

CORS allow-list from `IMAGE_ANALYSER_ALLOWED_ORIGINS` (comma-separated; default `*`).

### 4.3 Python library

```python
from image_analyser import ImageAnalyser, AnalysisResult

analyser = ImageAnalyser(skip=None, only=None, caption_backend="auto")
result: AnalysisResult = analyser.analyse("foo.jpg")        # accepts path | bytes | PIL.Image
print(result.metadata.exif.camera, result.objects[0].label if result.objects else None)
```

This is the surface `video-analyser` will eventually depend on (mirror of the `speech-analyser` library plan).

## 5. Response schema (Pydantic v2)

Top-level `AnalysisResult`:

| Field | Type | When | Notes |
|---|---|---|---|
| `format` | `str` | always | `"JPEG"` / `"PNG"` / `"WebP"` / `"AVIF"` / `"HEIC"` / `"GIF"` / `"TIFF"` / `"SVG"` / `"BMP"` |
| `mime_type` | `str` | always | from `python-magic` / Pillow inference |
| `resolution` | `tuple[int, int]` | always | (width, height) in pixels |
| `megapixels` | `float` | always | rounded to 2dp |
| `aspect_class` | `str` | always | `"landscape"` / `"portrait"` / `"square"` |
| `colour_mode` | `str` | always | `"RGB"` / `"RGBA"` / `"L"` / `"P"` / `"CMYK"` |
| `bit_depth` | `int` | always | 8 / 16 / 32 |
| `has_alpha` | `bool` | always | |
| `file_size` | `int` | always | bytes |
| `hash` | `Hash` | always | `sha256`, `phash`, `dhash` |
| `metadata` | `Metadata` | always | sub-fields below; missing values are `null`, never raise |
| `animation` | `Animation \| None` | only for animated formats | `frame_count`, `duration_s` |
| `quality` | `Quality` | always | |
| `colour` | `Colour` | always | |
| `barcodes` | `list[Barcode]` | always (empty if none) | |
| `objects` | `list[Object] \| None` | when ML available | |
| `caption` | `Caption \| None` | when caption backend resolves | |
| `ocr` | `Ocr \| None` | when OCR engine available | |
| `skipped` | `list[Skipped]` | always (may be empty) | each: `name`, `reason` |
| `failed` | `list[Failed]` | always (may be empty) | each: `name`, `error`, optional `traceback` |
| `version` | `str` | always | image-analyser version (drift guard) |
| `analysed_at` | `str` (ISO-8601) | always | UTC timestamp |
| `duration_ms` | `int` | always | wall-clock |

Sub-models:

```
Metadata { exif: Exif | None, iptc: dict | None, xmp: dict | None, c2pa: C2pa | None, icc_profile: IccProfile | None }
Exif { camera, lens, focal_length_mm, iso, aperture, shutter_speed, taken_at, gps: Gps | None }
Gps { lat: float, lon: float, alt: float | None }
C2pa { present: bool, manifest: dict | None, ai_generated_claim: bool | None }
IccProfile { name: str, colour_space: str }
Quality { blur_score: float, exposure: Exposure, brightness: float, contrast: float, noise: float, jpeg_quality_estimate: int | None }
Exposure { underexposed_pct: float, overexposed_pct: float, clipping_pct: float }
Colour { dominant: list[str] (hex), average: str (hex), palette: list[PaletteEntry] }
PaletteEntry { hex: str, weight: float }
Barcode { type: str, value: str, bbox: BBox }
Object { label: str, score: float, bbox: BBox }
BBox { x: int, y: int, w: int, h: int }
Caption { text: str, backend: "local" | "api", model: str }
Ocr { text: str, blocks: list[OcrBlock], engine: "tesseract" | "easyocr" }
OcrBlock { text: str, bbox: BBox, confidence: float }
Skipped { name: str, reason: str }
Failed { name: str, error: str, traceback: str | None }
Hash { sha256: str, phash: str, dhash: str }
Animation { frame_count: int, duration_s: float | None }
```

### 5.1 Skipped vs failed semantics
- `skipped`: dependency / config not available (e.g. `[ml]` extra not installed; no API key; OCR engine missing). Expected, not an error.
- `failed`: the analysis raised at runtime (e.g. corrupt EXIF, OOM during model inference). Captured, logged, surfaced in response. HTTP still returns 200.

## 6. Dependencies (extras pattern)

```toml
[project]
dependencies = [
    "pillow>=10",
    "pillow-heif>=0.16",         # HEIC / HEIF support (Apple devices)
    "numpy>=1.26",
    "imagehash>=4.3",            # pHash / dHash
    "pyzbar>=0.1.9",             # barcodes / QR codes (binds to libzbar)
    "c2pa-python>=0.5",          # Content Credentials parsing
    "pydantic>=2.6",
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "typer>=0.12",
    "python-magic>=0.4.27",      # MIME detection
]

[project.optional-dependencies]
ml  = ["torch>=2.2", "transformers>=4.40", "accelerate>=0.30"]
ocr = ["pytesseract>=0.3.10", "easyocr>=1.7"]
api = ["anthropic>=0.30", "openai>=1.30", "google-genai>=0.3"]
all = ["image-analyser[ml,ocr,api]"]
dev = ["pytest>=8", "pytest-cov>=5", "ruff>=0.5", "mypy>=1.10", "httpx>=0.27"]
```

System dependencies (documented in README):
- `libzbar0` (for `pyzbar`) — Linux: `apt install libzbar0`; macOS: `brew install zbar`.
- `tesseract` (only if `[ocr]` extra used and tesseract chosen) — `brew install tesseract` / `apt install tesseract-ocr`.

## 7. Configuration (env vars)

All env vars use the `IMAGE_ANALYSER_` prefix (matches family).

| Env var | Default | Purpose |
|---|---|---|
| `IMAGE_ANALYSER_PORT` | `8006` | FastAPI port |
| `IMAGE_ANALYSER_HOST` | `127.0.0.1` | bind address |
| `IMAGE_ANALYSER_MODE` | `production` | `production` / `development` |
| `IMAGE_ANALYSER_ALLOWED_ORIGINS` | `*` | CORS allow-list (comma-separated) |
| `IMAGE_ANALYSER_CAPTION_BACKEND` | `auto` | `auto` / `local` / `api` / `none` |
| `IMAGE_ANALYSER_CAPTION_PROVIDER` | `anthropic` | `anthropic` / `openai` / `google` / `openrouter` (when backend resolves to api) |
| `IMAGE_ANALYSER_CAPTION_MODEL` | provider-default | model id override (string) |
| `IMAGE_ANALYSER_LOCAL_CAPTION_MODEL` | `Salesforce/blip-image-captioning-base` | HuggingFace id when backend resolves to local |
| `IMAGE_ANALYSER_OCR_ENGINE` | `auto` | `auto` / `tesseract` / `easyocr` / `none` |
| `IMAGE_ANALYSER_OBJECT_DETECTION_MODEL` | `facebook/detr-resnet-50` | HuggingFace object-detection pipeline model id |
| `IMAGE_ANALYSER_OBJECT_DETECTION_THRESHOLD` | `0.5` | confidence threshold (0..1) below which detections are dropped |
| `IMAGE_ANALYSER_DEVICE` | `auto` | `auto` / `cpu` / `cuda` / `mps` |
| `IMAGE_ANALYSER_RATE_LIMIT_ENABLED` | `false` | per-IP request limit on `/analyse` |
| `IMAGE_ANALYSER_MAX_UPLOAD_MB` | `50` | reject uploads bigger than this |

API keys: standard provider env vars (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `OPENROUTER_API_KEY`) — *not* custom-prefixed. Detection of any of these triggers API-first auto behaviour.

## 8. Backend dispatch rules

Reasons listed below are stable strings (asserted by `test_invariants.py`).

### 8.1 Caption backend
1. `IMAGE_ANALYSER_CAPTION_BACKEND=none` → skip with reason `"disabled by config"`.
2. `=local` → require `transformers` importable; else skip with reason `"ml extra not installed"`.
3. `=api` → require provider key; else skip with reason `"api provider not configured"`.
4. `=auto` (default):
   - If `IMAGE_ANALYSER_CAPTION_PROVIDER`'s API key is present in env → use API with that provider.
   - Else if `transformers` importable → use local BLIP (`IMAGE_ANALYSER_LOCAL_CAPTION_MODEL`).
   - Else skip with reason `"no captioning backend available"`.

   Provider-to-key mapping (for `auto` resolution): `anthropic`→`ANTHROPIC_API_KEY`, `openai`→`OPENAI_API_KEY`, `google`→`GOOGLE_API_KEY`, `openrouter`→`OPENROUTER_API_KEY`.

### 8.2 OCR engine
1. `=none` → skip with reason `"disabled by config"`.
2. `=tesseract` → require `pytesseract` *and* tesseract on PATH; else skip with reason `"tesseract not installed"`.
3. `=easyocr` → require `easyocr`; else skip with reason `"easyocr not installed"`.
4. `=auto` (default):
   - If `pytesseract` importable AND tesseract on PATH → use tesseract (faster, smaller).
   - Else if `easyocr` importable → use easyocr.
   - Else skip with reason `"no ocr engine available"`.

### 8.3 Object detection
- Requires `[ml]` extra. If `transformers` not importable → skip with reason `"ml extra not installed"`.
- Backend: `transformers.pipeline("object-detection", model=IMAGE_ANALYSER_OBJECT_DETECTION_MODEL)` — default `facebook/detr-resnet-50`.
- Detections below `IMAGE_ANALYSER_OBJECT_DETECTION_THRESHOLD` (default 0.5) are dropped.
- Model loaded once on first use, cached on the `ImageAnalyser` instance for subsequent `analyse()` calls.

## 9. Testing strategy (apply document-analyser trust-pass lessons from day one)

1. **Default `pytest` runs fast.** ML and engine-conditional tests are `@pytest.mark.slow`, opt-in via `pytest -m slow`.
2. **`tests/test_invariants.py`** from v0.1.0:
   - Clean import smoke (`import image_analyser` without side-effects).
   - `/health` version equals `importlib.metadata.version("image-analyser")` — drift guard.
   - `/` version drift guard.
   - `AnalysisResult` schema completeness on a 1×1 PNG (all always-on Tier 1 fields populated).
   - Skipped reasons are stable strings (assertion against the literals listed in §8).
3. **No echo-tautology assertions** (don't assert response echoes request input).
4. **No vacuous assertions** (`>= 0`, `isinstance(x, list)`) on deterministic fixtures.
5. **No `400 or 500` accepting** — response status codes must be exact.
6. **Single source of truth for version** via `importlib.metadata.version("image-analyser")` in `__init__.py`, `app.py`, `cli.py`. Never hardcoded literals.

Coverage target: ≥ 85% on Tier 1 modules; ML modules tested via mocks + one slow real-model integration test gated on dep availability.

## 10. Repo bootstrap & release

1. `gh repo create michael-borck/image-analyser --public --license mit --description "Static image analysis (CLI + FastAPI) for the analyser family"`
2. Clone to `/Users/michael/Projects/lens/image-analyser/` (overlaying the existing `docs/` tree containing this spec).
3. Copy + adapt from `speech-analyser`: `pyproject.toml`, `README.md` skeleton, `.gitignore`, `Makefile` if present, pre-commit config. (LICENSE is created by `gh repo create --license mit`; verify content matches.)
4. Implement modules per §3.
5. Implement tests per §9.
6. README sections: badges, install (`pip install image-analyser` / `[all]`), system-deps note (libzbar / tesseract), CLI usage, HTTP API, Python library, env vars table, family-table footer, dev setup.
7. Quality gates: `pytest` green, `ruff check` clean, `mypy` clean.
8. `python -m build && twine upload dist/*` via existing `.pypirc`.
9. `git tag v0.1.0 && git push --tags`.
10. `gh release create v0.1.0 --generate-notes`.

## 11. Memory updates (on completion)

1. Add `image-analyser` row to the package-status table in `project_analyser_family_todos.md`. Port 8006 added to the ports table.
2. Add deferred follow-up: "video-analyser v0.7.0 — swap internal image stack (`object_detector` / `image_captioner` / `api_image_captioner` / `ocr_detector` / `frame_analyzer` / `visual_analyzer`) for `import image_analyser`. Mirror of the speech-analyser plan."
3. Add lesson: any future image-hashing or dedup work in the analyser family should reuse `image_analyser.hashing` (single source of truth).

## 12. Decisions log

- **2026-05-08**: scope = lift video-analyser ML modules + Tier 1 lightweight (rejected: stub-only; lift-only without Tier 1).
- **2026-05-08**: default behaviour = run everything available, report skipped/failed (rejected: Tier 1-only by default; per-analysis flags none-by-default).
- **2026-05-08**: caption backend default = API-first auto (rejected: local-first auto; explicit-only).
- **2026-05-08**: batch mode deferred — single file only in v0.1.0 (`bundle-analyser` + `auto-analyser` will handle batch routing once auto-analyser learns about images).
- **2026-05-08**: first PyPI version = 0.1.0 functional (rejected: 0.0.1 stub).
- **2026-05-08**: port = 8006 (free slot between wordpress 8005 and git 8007).

## 13. Open questions

None — all clarifying questions resolved during brainstorming (2026-05-08).
