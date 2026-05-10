"""Object detection via transformers.pipeline (DETR by default)."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
from functools import lru_cache
from typing import Any

from PIL import Image

from .schemas import BBox, Object

logger = logging.getLogger(__name__)


def _transformers_importable() -> bool:
    return importlib.util.find_spec("transformers") is not None


def is_available() -> bool:
    return _transformers_importable()


@lru_cache(maxsize=1)
def _get_pipeline(model: str, device: str) -> Any:
    from transformers import pipeline
    return pipeline("object-detection", model=model, device=device if device != "auto" else -1)


def _device() -> str:
    return os.getenv("IMAGE_ANALYSER_DEVICE", "auto")


def _model() -> str:
    return os.getenv("IMAGE_ANALYSER_OBJECT_DETECTION_MODEL", "facebook/detr-resnet-50")


def analyse(img: Image.Image, threshold: float | None = None) -> tuple[list[Object] | None, str | None]:
    """Run object detection. Returns (objects, None) on success or (None, skip-reason)."""
    if not _transformers_importable():
        return None, "ml extra not installed"
    if threshold is None:
        threshold = float(os.getenv("IMAGE_ANALYSER_OBJECT_DETECTION_THRESHOLD", "0.5"))
    try:
        pipe = _get_pipeline(_model(), _device())
        raw = pipe(img.convert("RGB"))
    except Exception as e:
        logger.warning("object detection failed: %s", e)
        raise
    objects: list[Object] = []
    for r in raw:
        if r["score"] < threshold:
            continue
        box = r["box"]
        objects.append(
            Object(
                label=r["label"],
                score=float(r["score"]),
                bbox=BBox(
                    x=int(box["xmin"]),
                    y=int(box["ymin"]),
                    w=int(box["xmax"] - box["xmin"]),
                    h=int(box["ymax"] - box["ymin"]),
                ),
            )
        )
    return objects, None
