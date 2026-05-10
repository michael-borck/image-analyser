"""Image metadata: format, dimensions, EXIF/IPTC/XMP, C2PA, ICC, animation."""

from __future__ import annotations

import logging
from typing import Any

from PIL import ExifTags, Image
from PIL.ExifTags import GPSTAGS, TAGS

from .schemas import Animation, C2pa, Exif, Gps, IccProfile, Metadata

logger = logging.getLogger(__name__)


# ---------- basic identity ----------

_MODE_BIT_DEPTH = {"1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "CMYK": 8, "I": 32, "F": 32}


def _detect_mime(raw_bytes: bytes, fmt: str) -> str:
    """Detect MIME via libmagic if loadable, else fall back to format-based.

    `python-magic` loads libmagic via ctypes at import time; on Macs without
    `brew install libmagic` this would crash `import image_analyser`. Defer
    the import so the package loads everywhere; degrade to the Pillow format
    if libmagic isn't available.
    """
    if not raw_bytes:
        return f"image/{fmt.lower()}"
    try:
        import magic
        return magic.from_buffer(raw_bytes, mime=True)
    except Exception:  # noqa: BLE001  — ImportError, OSError, magic-internal errors all degrade gracefully
        return f"image/{fmt.lower()}"


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
    mime = _detect_mime(raw_bytes, fmt)
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

def _iptc(img: Image.Image) -> dict[str, Any] | None:
    iptc = img.info.get("photoshop")  # Pillow stores IPTC inside Photoshop blocks for JPEGs
    return iptc if isinstance(iptc, dict) else None


def _xmp(img: Image.Image) -> dict[str, Any] | None:
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
        import c2pa
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


def _c2pa_ai_claim(manifest: dict[str, Any] | str | None) -> bool | None:
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
        import io

        from PIL import ImageCms

        profile = ImageCms.ImageCmsProfile(io.BytesIO(icc_bytes))
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
