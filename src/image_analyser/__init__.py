"""Static image analysis for the analyser family."""

from importlib.metadata import version as _version

from .exceptions import ImageAnalyserError
from .image_analyser import ImageAnalyser
from .schemas import AnalysisResult

__version__ = _version("image-analyser")
__all__ = ["ImageAnalyser", "AnalysisResult", "ImageAnalyserError", "__version__"]
