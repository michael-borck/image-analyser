"""Canonical public-surface tests for the analyser family."""

import image_analyser
from image_analyser import (
    AnalysisResult,
    ImageAnalyser,
    ImageAnalysis,
    analyse,
)


def test_canonical_names_importable() -> None:
    assert ImageAnalyser is not None
    assert ImageAnalysis is AnalysisResult


def test_analyse_is_callable() -> None:
    assert callable(analyse)


def test_manifest_name() -> None:
    assert image_analyser.MANIFEST["name"] == "image-analyser"


def test_version_is_str() -> None:
    assert isinstance(image_analyser.__version__, str)


def test_all_lists_canonical_names() -> None:
    for name in (
        "ImageAnalyser",
        "ImageAnalysis",
        "AnalysisResult",
        "analyse",
        "MANIFEST",
        "__version__",
        "ImageAnalyserError",
        "DiagramHint",
    ):
        assert name in image_analyser.__all__
