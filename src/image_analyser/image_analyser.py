"""ImageAnalyser orchestrator: dispatches per-analysis modules."""

from __future__ import annotations

import io
import logging
import time
import traceback
from collections.abc import Callable
from datetime import UTC, datetime
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import pillow_heif  # noqa: F401  # registers HEIF/HEIC opener with Pillow
from PIL import Image, UnidentifiedImageError

from . import barcode, caption, colour, diagram, hashing, metadata, objects, ocr, quality
from .exceptions import UnsupportedFormatError
from .schemas import AnalysisResult, Failed, Skipped

logger = logging.getLogger(__name__)

# Toggleable modules (in pipeline order). Format/resolution/file-size are always-on.
_TOGGLEABLE = ("metadata", "hashing", "quality", "colour", "barcode", "objects", "caption", "ocr", "diagram")


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
        b: list[Any] = []
        if enabled("barcode"):
            b_result = self._safe("barcode", failed, lambda: barcode.analyse(img))
            if b_result is not None:
                values, reason = b_result
                if reason:
                    skipped.append(Skipped(name="barcode", reason=reason))
                else:
                    b = values or []

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
        dg = None
        if enabled("diagram"):
            dg_result = self._safe("diagram", failed, lambda: diagram.analyse(img))
            if dg_result is not None:
                value, reason = dg_result
                if reason:
                    skipped.append(Skipped(name="diagram", reason=reason))
                else:
                    dg = value

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
            diagram=dg,
            skipped=skipped,
            failed=failed,
            version=self._version,
            analysed_at=datetime.now(UTC).isoformat(timespec="seconds"),
            duration_ms=int((time.perf_counter() - start) * 1000),
        )

    # ---- helpers ----

    def _load(self, source: str | Path | bytes | Image.Image) -> tuple[Image.Image, bytes]:
        if isinstance(source, Image.Image):
            buf = io.BytesIO()
            source.save(buf, source.format or "PNG")
            return source, buf.getvalue()
        if isinstance(source, bytes):
            try:
                img = Image.open(io.BytesIO(source))
                img.load()
                return img, source
            except UnidentifiedImageError as e:
                raise UnsupportedFormatError(str(e)) from e
        path = Path(source)
        raw = path.read_bytes()
        try:
            img = Image.open(io.BytesIO(raw))
            img.load()
        except UnidentifiedImageError as e:
            raise UnsupportedFormatError(f"Unsupported image format: {path}") from e
        return img, raw

    def _safe(self, name: str, failed: list[Failed], fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except Exception as e:
            logger.warning("%s analysis failed: %s", name, e)
            failed.append(Failed(name=name, error=str(e), traceback=traceback.format_exc()))
            return None

    def _with_env(self, overrides: dict[str, str | None], fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
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

def _zero_hash() -> Any:
    from .schemas import Hash
    return Hash(sha256="0" * 64, phash="0" * 16, dhash="0" * 16)


def _empty_metadata() -> Any:
    from .schemas import Metadata
    return Metadata()


def _zero_quality() -> Any:
    from .schemas import Exposure, Quality
    return Quality(
        blur_score=0.0,
        exposure=Exposure(underexposed_pct=0.0, overexposed_pct=0.0, clipping_pct=0.0),
        brightness=0.0, contrast=0.0, noise=0.0, jpeg_quality_estimate=None,
    )


def _zero_colour() -> Any:
    from .schemas import Colour
    return Colour(dominant=["#000000"], average="#000000", palette=[])
