# tests/test_schemas.py
"""Schema round-trip tests."""

import pytest
from pydantic import ValidationError

from image_analyser.schemas import AnalysisResult, Skipped


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
    with pytest.raises(ValidationError):
        Skipped.model_validate({"name": "objects"})
