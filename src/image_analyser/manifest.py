"""Capability manifest for the lens family (consumed by auto-analyser)."""
from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def _version() -> str:
    try:
        return version("image-analyser")
    except PackageNotFoundError:
        return "0.0.0"


MANIFEST: dict = {
    "name": "image-analyser",
    "version": _version(),
    "role": "analyser",
    "accepts": ["image"],
    "extensions": [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"],
    "auto_routable": True,
    "produces": "AnalysisResult",
}
