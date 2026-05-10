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
    manifest: dict[str, object] | None = None
    ai_generated_claim: bool | None = None


class IccProfile(_Strict):
    name: str
    colour_space: str


class Metadata(_Strict):
    exif: Exif | None = None
    iptc: dict[str, object] | None = None
    xmp: dict[str, object] | None = None
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
