# tests/test_caption.py
"""Captioning tests — dispatcher, local, API."""

from unittest.mock import MagicMock, patch

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
         patch("image_analyser.caption._caption_local", return_value=("a placeholder caption", 7)):
        cap, reason = analyse(Image.new("RGB", (10, 10)))
    assert reason is None
    assert cap.backend == "local"
    assert cap.text == "a placeholder caption"


def test_auto_picks_api_when_key_present(monkeypatch):
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with patch(
        "image_analyser.caption._caption_api",
        return_value=("a tabby cat", "openai", "gpt-4o-mini", 100, 50),
    ):
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


def test_local_caption_reports_tokens_generated(monkeypatch):
    """Local BLIP backend should expose tokens_generated from out.shape[1]."""
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "local")

    fake_processor = MagicMock()
    fake_processor.return_value = {"pixel_values": MagicMock()}
    # batch_decode([...]) returns ["a fake caption"]
    fake_processor.batch_decode.return_value = ["a fake caption"]

    fake_out = MagicMock()
    fake_out.shape = (1, 12)  # batch=1, sequence_length=12
    fake_model = MagicMock()
    fake_model.generate.return_value = fake_out

    with patch("image_analyser.caption._transformers_importable", return_value=True), \
         patch("image_analyser.caption._load_blip", return_value=(fake_processor, fake_model)):
        cap, reason = analyse(Image.new("RGB", (10, 10)))

    assert reason is None
    assert cap is not None
    assert cap.backend == "local"
    assert cap.tokens_generated == 12
    assert cap.cost_estimate_usd is None  # local has no cost


def test_api_caption_reports_tokens_and_cost(monkeypatch):
    """API backend should report combined tokens and a cost estimate when model is known."""
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "api")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_MODEL", "claude-3-5-sonnet-latest")

    with patch(
        "image_analyser.caption._caption_api",
        return_value=("a captioned image", "anthropic", "claude-3-5-sonnet-latest", 100, 50),
    ):
        cap, reason = analyse(Image.new("RGB", (10, 10)))

    assert reason is None
    assert cap is not None
    assert cap.tokens_generated == 150
    # claude-3-5-sonnet-latest: (3.0, 15.0) per million
    # cost = (100*3 + 50*15) / 1_000_000 = (300 + 750)/1_000_000 = 0.00105
    assert cap.cost_estimate_usd is not None
    assert cap.cost_estimate_usd > 0
    assert abs(cap.cost_estimate_usd - 0.00105) < 1e-9


def test_unknown_model_cost_is_none(monkeypatch):
    """Unknown model should produce cost_estimate_usd=None rather than fake zero."""
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_BACKEND", "api")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("IMAGE_ANALYSER_CAPTION_MODEL", "some-unknown-model")

    with patch(
        "image_analyser.caption._caption_api",
        return_value=("a captioned image", "anthropic", "some-unknown-model", 100, 50),
    ):
        cap, reason = analyse(Image.new("RGB", (10, 10)))

    assert reason is None
    assert cap is not None
    assert cap.tokens_generated == 150
    assert cap.cost_estimate_usd is None
