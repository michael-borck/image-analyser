"""Exception types for image-analyser."""


class ImageAnalyserError(Exception):
    """Base exception for image-analyser failures."""


class UnsupportedFormatError(ImageAnalyserError):
    """Raised when the input file is not a recognised image format."""


class FileTooLargeError(ImageAnalyserError):
    """Raised when an upload exceeds IMAGE_ANALYSER_MAX_UPLOAD_MB."""
