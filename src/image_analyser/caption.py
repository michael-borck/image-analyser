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


def _transformers_importable() -> bool:
    return importlib.util.find_spec("transformers") is not None


# ---------- local BLIP ----------

@lru_cache(maxsize=1)
def _load_blip(model_name: str) -> tuple[Any, Any]:
    from transformers import AutoProcessor, BlipForConditionalGeneration
    processor = AutoProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    return processor, model


def _caption_local(img: Image.Image, model_name: str) -> str:
    processor, model = _load_blip(model_name)
    inputs = processor(images=img.convert("RGB"), return_tensors="pt")
    out = model.generate(**inputs, max_new_tokens=40)
    return str(processor.batch_decode(out, skip_special_tokens=True)[0].strip())


# ---------- API providers ----------

def _img_to_b64(img: Image.Image) -> tuple[str, str]:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _caption_api(img: Image.Image, provider: str, model: str | None) -> tuple[str, str, str]:
    """Returns (text, provider, model_used)."""
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
        return msg.content[0].text.strip(), provider, chosen_model
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
        return resp.choices[0].message.content.strip(), provider, chosen_model
    if provider == "google":
        from google import genai
        client = genai.Client(api_key=os.environ[PROVIDER_KEYS[provider]])
        resp = client.models.generate_content(
            model=chosen_model,
            contents=[prompt, {"inline_data": {"mime_type": mime, "data": b64}}],
        )
        return resp.text.strip(), provider, chosen_model
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
        text = _caption_local(img, local_model)
        return Caption(text=text, backend="local", model=local_model), None

    if backend == "api":
        key = os.getenv(PROVIDER_KEYS.get(provider, ""))
        if not key:
            return None, "api provider not configured"
        text, _prov, used = _caption_api(img, provider, api_model_override)
        return Caption(text=text, backend="api", model=used), None

    # auto
    key = os.getenv(PROVIDER_KEYS.get(provider, ""))
    if key:
        try:
            text, _prov, used = _caption_api(img, provider, api_model_override)
            return Caption(text=text, backend="api", model=used), None
        except ImportError:
            logger.debug("caption api provider package not installed for %s; falling back", provider)
    if _transformers_importable():
        text = _caption_local(img, local_model)
        return Caption(text=text, backend="local", model=local_model), None
    return None, "no captioning backend available"
