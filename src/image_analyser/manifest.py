"""Capability manifest for the lens family (consumed by auto-analyser)."""
from __future__ import annotations

from lens_contract import make_manifest

MANIFEST = make_manifest(
    name="image-analyser",
    accepts=["image"],
    extensions=[".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif", ".webp"],
    auto_routable=True,
    produces="AnalysisResult",
)
