# tests/test_invariants.py
"""Project invariants — drift guards and stable contracts."""

from importlib.metadata import version as _v

from fastapi.testclient import TestClient

from image_analyser import AnalysisResult, ImageAnalyser, __version__
from image_analyser.api import app


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
    "barcode": {"libzbar not loadable"},  # may skip on systems where libzbar isn't installed
}


def test_skip_reasons_are_from_known_set(fixtures_dir):
    result = ImageAnalyser().analyse(fixtures_dir / "1x1.png")
    for s in result.skipped:
        if s.name in EXPECTED_REASONS and EXPECTED_REASONS[s.name]:
            assert s.reason in EXPECTED_REASONS[s.name], (
                f"unexpected skip reason for {s.name}: {s.reason!r}"
            )
