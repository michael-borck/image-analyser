"""Image embedding wiring (lens-embed CLIP) — field presence + graceful degradation."""
from __future__ import annotations
import importlib.util
import io
import pytest
from image_analyser.embedding import embed_image_vector
from image_analyser.schemas import AnalysisResult

_IMG = importlib.util.find_spec("lens_embed") is not None and importlib.util.find_spec("open_clip") is not None

def test_field_default_none():
    assert "embedding" in AnalysisResult.model_fields
    assert AnalysisResult.model_fields["embedding"].default is None

@pytest.mark.skipif(_IMG, reason="image embeddings extra installed")
def test_none_without_backend():
    assert embed_image_vector(b"not an image") is None

@pytest.mark.slow
@pytest.mark.skipif(not _IMG, reason="needs lens-embed[image]")
def test_vector_with_backend():
    from PIL import Image
    buf = io.BytesIO(); Image.new("RGB", (64, 64), (120, 200, 60)).save(buf, format="PNG")
    v = embed_image_vector(buf.getvalue())
    assert isinstance(v, list) and len(v) == 512  # CLIP ViT-B-32
