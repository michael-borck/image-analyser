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
