"""Image captioning: local BLIP + API providers (Anthropic, OpenAI, Google, OpenRouter)."""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
from functools import lru_cache
from typing import Any

from PIL import Image

from .schemas import Caption

logger = logging.getLogger(__name__)


PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

DEFAULT_API_MODELS = {
    "anthropic": "claude-3-5-sonnet-latest",
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "openrouter": "openai/gpt-4o-mini",
}


# Per-million-token pricing (USD), as of 2026-05. Update when models change.
# Format: provider -> { model: (input_per_million_usd, output_per_million_usd) }
PRICES_PER_MILLION_TOKENS_USD: dict[str, dict[str, tuple[float, float]]] = {
    "anthropic": {
        "claude-3-5-sonnet-latest": (3.0, 15.0),
        "claude-3-5-haiku-latest": (0.80, 4.0),
        "claude-3-opus-latest": (15.0, 75.0),
    },
    "openai": {
        "gpt-4o": (2.50, 10.0),
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4-turbo": (10.0, 30.0),
    },
    "google": {
        "gemini-1.5-flash": (0.075, 0.30),
        "gemini-1.5-pro": (1.25, 5.0),
        "gemini-2.0-flash-exp": (0.10, 0.40),
    },
    "openrouter": {
        # OpenRouter prices vary; we publish None for unknown.
    },
}


def _estimate_cost_usd(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Return USD cost estimate, or None when we don't know the price.

    Better to return None than fake a number — the consumer can see the
    cost is unknown rather than thinking it's zero.
    """
    if input_tokens is None or output_tokens is None:
        return None
    table = PRICES_PER_MILLION_TOKENS_USD.get(provider, {})
    rates = table.get(model)
    if not rates:
        return None
    in_rate, out_rate = rates
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


def _transformers_importable() -> bool:
    return importlib.util.find_spec("transformers") is not None


# ---------- local BLIP ----------

@lru_cache(maxsize=1)
def _load_blip(model_name: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor, BlipForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def _caption_local(img: Image.Image, model_name: str) -> tuple[str, int | None]:
    """Returns (text, tokens_generated)."""
    processor, model = _load_blip(model_name)
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    text = str(processor.batch_decode(out, skip_special_tokens=True)[0].strip())
    # out shape is [batch, sequence_length]; for BLIP the input is image-only,
    # so the output sequence length IS the generated text length.
    tokens_generated = int(out.shape[1])
    return text, tokens_generated


# ---------- API providers ----------

def _img_to_b64(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _caption_api(
    img: Image.Image, provider: str, model: str | None
) -> tuple[str, str, str, int | None, int | None]:
    """Returns (text, provider, model_used, input_tokens, output_tokens)."""
    chosen_model = model or DEFAULT_API_MODELS[provider]
    b64, mime = _img_to_b64(img)
    prompt = "Describe this image in one concise sentence."
    if provider == "anthropic":
        from anthropic import Anthropic
        msg = Anthropic().messages.create(
            model=chosen_model,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        text = msg.content[0].text.strip()
        input_tokens = getattr(msg.usage, "input_tokens", None) if msg.usage else None
        output_tokens = getattr(msg.usage, "output_tokens", None) if msg.usage else None
        return text, provider, chosen_model, input_tokens, output_tokens
    if provider == "openai" or provider == "openrouter":
        from openai import OpenAI
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1" if provider == "openrouter" else None,
            api_key=os.environ[PROVIDER_KEYS[provider]],
        )
        resp = client.chat.completions.create(
            model=chosen_model,
            max_tokens=120,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                ],
            }],
        )
        text = resp.choices[0].message.content.strip()
        input_tokens = resp.usage.prompt_tokens if resp.usage else None
        output_tokens = resp.usage.completion_tokens if resp.usage else None
        return text, provider, chosen_model, input_tokens, output_tokens
    if provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ[PROVIDER_KEYS[provider]])
        resp = client.models.generate_content(
            model=chosen_model,
            contents=[prompt, {"inline_data": {"mime_type": mime, "data": b64}}],
        )
        text = resp.text.strip()
        # google's response: resp.usage_metadata.prompt_token_count / candidates_token_count
        usage_metadata = getattr(resp, "usage_metadata", None)
        if usage_metadata:
            input_tokens = getattr(usage_metadata, "prompt_token_count", None)
            output_tokens = getattr(usage_metadata, "candidates_token_count", None)
        else:
            input_tokens = None
            output_tokens = None
        return text, provider, chosen_model, input_tokens, output_tokens
    raise ValueError(f"unknown provider: {provider}")


# ---------- dispatcher ----------

def analyse(img: Image.Image) -> tuple[Caption | None, str | None]:
    """Resolve backend per spec §8.1. Returns (Caption, None) on success, (None, reason) on skip."""
    backend = os.getenv("IMAGE_ANALYSER_CAPTION_BACKEND", "auto")
    provider = os.getenv("IMAGE_ANALYSER_CAPTION_PROVIDER", "anthropic")
    local_model = os.getenv("IMAGE_ANALYSER_LOCAL_CAPTION_MODEL", "Salesforce/blip-image-captioning-base")
    api_model_override = os.getenv("IMAGE_ANALYSER_CAPTION_MODEL") or None

    if backend == "none":
        return None, "disabled by config"

    if backend == "local":
        if not _transformers_importable():
            return None, "ml extra not installed"
        text, tokens = _caption_local(img, local_model)
        return Caption(
            text=text,
            backend="local",
            model=local_model,
            tokens_generated=tokens,
            cost_estimate_usd=None,  # local has no cost
        ), None

    if backend == "api":
        key = os.getenv(PROVIDER_KEYS.get(provider, ""))
        if not key:
            return None, "api provider not configured"
        text, _prov, used, in_tok, out_tok = _caption_api(img, provider, api_model_override)
        total_tokens = (in_tok or 0) + (out_tok or 0) if in_tok is not None else None
        return Caption(
            text=text,
            backend="api",
            model=used,
            tokens_generated=total_tokens,
            cost_estimate_usd=_estimate_cost_usd(provider, used, in_tok, out_tok),
        ), None

    # auto
    key = os.getenv(PROVIDER_KEYS.get(provider, ""))
    if key:
        try:
            text, _prov, used, in_tok, out_tok = _caption_api(img, provider, api_model_override)
            total_tokens = (in_tok or 0) + (out_tok or 0) if in_tok is not None else None
            return Caption(
                text=text,
                backend="api",
                model=used,
                tokens_generated=total_tokens,
                cost_estimate_usd=_estimate_cost_usd(provider, used, in_tok, out_tok),
            ), None
        except ImportError:
            logger.debug("caption api provider package not installed for %s; falling back", provider)
        except Exception as e:
            logger.debug("caption api call failed for %s; falling back to local: %s", provider, e)
    if _transformers_importable():
        try:
            text, tokens = _caption_local(img, local_model)
            return Caption(
                text=text,
                backend="local",
                model=local_model,
                tokens_generated=tokens,
                cost_estimate_usd=None,
            ), None
        except ImportError:
            logger.debug("local captioning dependencies not available; skipping")
        except Exception as e:
            logger.debug("local captioning failed; skipping: %s", e)
    return None, "no captioning backend available"
