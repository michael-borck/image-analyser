# tests/test_caption.py
"""Captioning tests — dispatcher, local, API."""

from unittest.mock import patch

from PIL import Image

from image_analyser.caption import analyse


def test_disabled_by_config(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "none")
    cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "disabled by config"


def test_auto_falls_back_to_local_when_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    with patch("image_analyser.caption._transformers_importable", return_value=True), \
         patch("image_analyser.caption._caption_local", return_value="a placeholder caption"):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert reason is None
    assert cap.backend == "local"
    assert cap.text == "a placeholder caption"


def test_auto_picks_api_when_key_present(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch("image_analyser.caption._caption_api", return_value=("a tabby cat", "openai", "gpt-4o-mini")):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert reason is None
    assert cap.backend == "api"
    assert cap.text == "a tabby cat"
    assert cap.model == "gpt-4o-mini"


def test_no_backend_skips(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    with patch("image_analyser.caption._transformers_importable", return_value=False):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "no captioning backend available"


def test_local_explicit_skips_when_no_ml(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "local")
    with patch("image_analyser.caption._transformers_importable", return_value=False):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "ml extra not installed"


def test_api_explicit_skips_when_no_key(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "api")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert cap is None
    assert reason == "api provider not configured"
