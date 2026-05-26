"""Tests for the diagram-detection module.

Two backends are exercised:
- heuristic: pure-Python, exercised against synthetic 'diagram-like' and 'photo-like'
  images built in-memory (no real fixtures needed).
- api: tested via the graceful-degradation path (no key set, no SDK available).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
from PIL import Image, ImageDraw

from image_analyser import diagram
from image_analyser.schemas import DiagramHint


# ── synthetic image factories ────────────────────────────────────────────


def _diagram_like() -> Image.Image:
    """A flat-colour 'flowchart' — large solid regions, just a handful of colours."""
    img = Image.new("RGB", (300, 200), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    d.rectangle((20, 40, 120, 100), fill=(200, 220, 240), outline=(0, 0, 0), width=2)
    d.rectangle((180, 40, 280, 100), fill=(200, 220, 240), outline=(0, 0, 0), width=2)
    d.line((120, 70, 180, 70), fill=(0, 0, 0), width=2)
    return img


def _photo_like() -> Image.Image:
    """Noise + gradient — lots of unique colours, very few flat pairs."""
    rng = np.random.default_rng(42)
    base = np.indices((200, 200)).sum(axis=0).astype(int)  # int to avoid uint8 overflow during arithmetic
    arr = np.stack([base, (base + 50) % 256, (base // 2)], axis=-1)
    noise = rng.integers(0, 30, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# ── heuristic ────────────────────────────────────────────────────────────


class TestHeuristic:
    def test_diagram_like_classified_as_diagram(self):
        hint, reason = diagram.analyse(_diagram_like())
        assert reason is None
        assert isinstance(hint, DiagramHint)
        assert hint.backend == "heuristic"
        assert hint.is_diagram is True
        assert hint.confidence > 0.6
        assert "distinct_quantized_colors" in hint.signals
        assert "flat_region_ratio" in hint.signals
        # Sanity: a flowchart with ~3 fills should have low distinct-color count.
        assert hint.signals["distinct_quantized_colors"] < 100
        assert hint.signals["flat_region_ratio"] > 0.85

    def test_photo_like_classified_as_not_diagram(self):
        hint, reason = diagram.analyse(_photo_like())
        assert reason is None
        assert hint.is_diagram is False
        assert hint.confidence < 0.6
        assert hint.signals["distinct_quantized_colors"] > 200

    def test_kind_is_none_for_heuristic(self):
        hint, _ = diagram.analyse(_diagram_like())
        # The heuristic can't tell flowchart from UML — kind stays None.
        assert hint.kind is None


# ── backend env handling ─────────────────────────────────────────────────


class TestBackendEnv:
    def test_disabled(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("IMAGE_ANALYSER_DIAGRAM_BACKEND", "none")
        hint, reason = diagram.analyse(_diagram_like())
        assert hint is None
        assert reason and "disabled" in reason

    def test_unknown_backend(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("IMAGE_ANALYSER_DIAGRAM_BACKEND", "weird")
        hint, reason = diagram.analyse(_diagram_like())
        assert hint is None
        assert reason and "unknown" in reason

    def test_api_with_no_providers_falls_back_to_heuristic(self, monkeypatch: pytest.MonkeyPatch):
        # api requested but no env keys set → fall back to heuristic, signal it via api_error.
        monkeypatch.setenv("IMAGE_ANALYSER_DIAGRAM_BACKEND", "api")
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(key, raising=False)
        hint, reason = diagram.analyse(_diagram_like())
        # The contract: even when the API path fails, we DO return a hint (the heuristic),
        # plus a reason explaining what went wrong with the API.
        assert hint is not None
        assert hint.backend == "heuristic"
        assert reason and "no API provider" in reason
        assert hint.signals.get("api_error") == 1.0


# ── vision JSON parser ───────────────────────────────────────────────────


class TestVisionJsonParse:
    def test_clean_json(self):
        parsed = diagram._parse_vision_json('{"is_diagram": true, "kind": "flowchart", "confidence": 0.9}')
        assert parsed == {"is_diagram": True, "kind": "flowchart", "confidence": 0.9}

    def test_with_fences(self):
        parsed = diagram._parse_vision_json('```json\n{"is_diagram": false}\n```')
        assert parsed == {"is_diagram": False}

    def test_with_prose_around(self):
        parsed = diagram._parse_vision_json(
            'Here is the answer:\n{"is_diagram": true, "kind": "uml"}\nLet me know if you need more.'
        )
        assert parsed == {"is_diagram": True, "kind": "uml"}

    def test_malformed(self):
        assert diagram._parse_vision_json("not json at all") is None
        assert diagram._parse_vision_json("") is None
