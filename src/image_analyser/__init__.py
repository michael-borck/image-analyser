"""Static image analysis for the analyser family."""

from importlib.metadata import version as _version
from pathlib import Path
from typing import TYPE_CHECKING

from .exceptions import ImageAnalyserError
from .image_analyser import ImageAnalyser
from .manifest import MANIFEST
from .schemas import AnalysisResult, DiagramHint

if TYPE_CHECKING:
    from PIL.Image import Image

# Canonical family alias for the result model (keep AnalysisResult for back-compat).
ImageAnalysis = AnalysisResult

__version__ = _version("image-analyser")


def analyse(source: "str | Path | bytes | Image") -> AnalysisResult:
    """Analyse a single image given a path, bytes, or PIL image."""
    return ImageAnalyser().analyse(source)


__all__ = [
    "ImageAnalyser",
    "ImageAnalysis",
    "AnalysisResult",
    "analyse",
    "MANIFEST",
    "__version__",
    "ImageAnalyserError",
    "DiagramHint",
]
