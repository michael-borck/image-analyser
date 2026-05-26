"""Diagram detection — heuristic-first, with optional vision-confirm.

The job of this module is *not* to extract a diagram's structure (that's
diagram-analyser's job). It's to **classify** an image as diagram-or-not so
auto-analyser can cascade-route diagram-like images to diagram-analyser.

Two backends:

- **heuristic** (default, no extra deps, ~50ms typical) — color quantization +
  flat-region ratio. Diagrams have few unique colors and large flat regions;
  photographs have many unique colors and continuous gradients.
- **api** (opt-in via `IMAGE_ANALYSER_DIAGRAM_BACKEND=api`) — sends the image
  to an LLM provider (anthropic / openai / google / openrouter) using the same
  provider plumbing as caption.py, asking for a yes/no + kind. The vision
  result *overrides* the heuristic.

Both surface their raw signals in DiagramHint.signals so callers can audit.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import logging
import os
from typing import Any

import numpy as np
from PIL import Image

from .schemas import DiagramHint

logger = logging.getLogger(__name__)


# Composite-score threshold above which an image is classified as a diagram.
# Tuned against the typical ranges below; surfaced via DIAGRAM_THRESHOLD env if needed.
_DEFAULT_THRESHOLD = 0.6

# Provider plumbing (mirrors caption.py).
_PROVIDER_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

_DEFAULT_API_MODELS = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
    "google": "gemini-2.0-flash",
    "openrouter": "openai/gpt-4o-mini",
}


# ── Heuristic ────────────────────────────────────────────────────────────


def _heuristic_signals(img: Image.Image) -> dict[str, float]:
    """Return raw heuristic signals — no thresholding here.

    Computed:
    - distinct_quantized_colors: number of unique 4-bit-per-channel colors.
      Photos typically 500–3000+; diagrams typically <200.
    - flat_region_ratio: fraction of adjacent pixel pairs (downsampled) with
      *identical* quantized color. Photos ~0.3–0.6; diagrams ~0.85–0.99.
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb)

    # 4-bit/channel quantize — keeps regions of "nearly the same colour" identical.
    quant = (arr & 0xF0).astype(np.uint32)

    # Pack RGB into a single uint32 for a fast unique count.
    packed = (quant[..., 0] << 16) | (quant[..., 1] << 8) | quant[..., 2]
    distinct = int(np.unique(packed).size)

    # Flat-region ratio on a downsampled grid (speed) — adjacent pairs identical.
    small = quant[::4, ::4]
    sh, sw = small.shape[:2]
    if sh > 1 and sw > 1:
        h_same = np.all(small[:, 1:] == small[:, :-1], axis=2)
        v_same = np.all(small[1:, :] == small[:-1, :], axis=2)
        same_pairs = int(h_same.sum() + v_same.sum())
        total_pairs = h_same.size + v_same.size
        flat_ratio = same_pairs / total_pairs if total_pairs else 0.0
    else:
        flat_ratio = 0.0

    return {
        "distinct_quantized_colors": float(distinct),
        "flat_region_ratio": round(float(flat_ratio), 4),
    }


def _composite_from_signals(signals: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Map raw signals → per-signal 0–1 score → composite confidence."""
    distinct = signals.get("distinct_quantized_colors", 0.0)
    flat = signals.get("flat_region_ratio", 0.0)

    # Few distinct colours → diagram-like. Above 300 distinct → not diagram.
    color_score = max(0.0, min(1.0, (300.0 - distinct) / 300.0))
    # High flat-ratio → diagram-like. The 0.7 floor / 0.3 slope was tuned against
    # typical photos vs diagrams in informal testing.
    flat_score = max(0.0, min(1.0, (flat - 0.7) / 0.3))

    confidence = (color_score + flat_score) / 2.0
    return round(confidence, 4), {
        "color_score": round(color_score, 4),
        "flat_score": round(flat_score, 4),
    }


def _heuristic(img: Image.Image) -> DiagramHint:
    signals = _heuristic_signals(img)
    confidence, sub = _composite_from_signals(signals)
    return DiagramHint(
        is_diagram=confidence > _DEFAULT_THRESHOLD,
        confidence=confidence,
        kind=None,  # the heuristic can't reliably tell flowchart from UML
        signals={**signals, **sub},
        backend="heuristic",
        model=None,
    )


# ── Optional vision-confirm ──────────────────────────────────────────────


_VISION_PROMPT = (
    "Classify this image. Respond with STRICT JSON only (no prose, no fences):\n"
    "{\"is_diagram\": true|false, "
    '"kind": "flowchart"|"uml"|"er"|"sequence"|"state"|"architecture"|"mindmap"|"other"|null, '
    "\"confidence\": 0.0-1.0}\n"
    "A diagram is a structured visual with nodes/edges/relationships (flowcharts, UML, ER, "
    "architecture, sequence, state, mindmaps). Photographs, screenshots of prose, GUI screenshots, "
    "and illustrations are NOT diagrams."
)


def _img_to_b64_jpeg(img: Image.Image) -> str:
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=85)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


def _resolve_provider() -> tuple[str, str] | None:
    """Pick the first provider whose key is set AND whose SDK is installed.

    Returns (provider, model) or None if nothing usable is configured.
    """
    pref = (os.getenv("IMAGE_ANALYSER_API_PROVIDER") or "").strip().lower()
    order = [pref] if pref else []
    order.extend(p for p in _PROVIDER_KEYS if p != pref)
    for provider in order:
        if provider not in _PROVIDER_KEYS:
            continue
        if not os.getenv(_PROVIDER_KEYS[provider]):
            continue
        sdk_present = {
            "anthropic": "anthropic",
            "openai": "openai",
            "google": "google.genai",
            "openrouter": "openai",  # openrouter uses the openai-compatible SDK
        }[provider]
        if importlib.util.find_spec(sdk_present) is None:
            continue
        model = os.getenv("IMAGE_ANALYSER_DIAGRAM_MODEL") or _DEFAULT_API_MODELS[provider]
        return provider, model
    return None


def _api(img: Image.Image) -> tuple[DiagramHint | None, str | None]:
    """Run the API-backed detector. Returns (hint_or_none, error_reason_or_none)."""
    resolved = _resolve_provider()
    if resolved is None:
        return None, "no API provider configured (set ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / OPENROUTER_API_KEY and install the matching SDK via the [api] extra)"
    provider, model = resolved

    try:
        raw_text = _call_provider(provider, model, img)
    except Exception as e:
        return None, f"{provider} API call failed: {e}"

    parsed = _parse_vision_json(raw_text)
    if parsed is None:
        return None, f"{provider} returned unstructured response: {raw_text[:200]!r}"

    is_diagram = bool(parsed.get("is_diagram"))
    kind = parsed.get("kind")
    if kind in ("null", ""):
        kind = None
    confidence = float(parsed.get("confidence") or (0.9 if is_diagram else 0.1))

    # Merge: surface heuristic signals alongside the API verdict so the verdict is auditable.
    heuristic_signals = _heuristic_signals(img)
    return DiagramHint(
        is_diagram=is_diagram,
        confidence=round(confidence, 4),
        kind=kind,
        signals=heuristic_signals,
        backend="api",
        model=f"{provider}/{model}",
    ), None


def _call_provider(provider: str, model: str, img: Image.Image) -> str:
    """Dispatch to the right SDK. Returns the raw text response."""
    b64 = _img_to_b64_jpeg(img)
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                    },
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }],
        )
        return _extract_anthropic_text(msg)
    if provider in ("openai", "openrouter"):
        import openai

        base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
        client = openai.OpenAI(base_url=base_url) if base_url else openai.OpenAI()
        resp = client.chat.completions.create(
            model=model,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }],
        )
        return resp.choices[0].message.content or ""
    if provider == "google":
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client()
        resp = client.models.generate_content(
            model=model,
            contents=[
                genai_types.Part.from_bytes(
                    data=base64.b64decode(b64), mime_type="image/jpeg"
                ),
                _VISION_PROMPT,
            ],
        )
        return resp.text or ""
    raise ValueError(f"unknown provider: {provider}")


def _extract_anthropic_text(msg: Any) -> str:
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            return block.text or ""
    return ""


def _parse_vision_json(raw: str) -> dict[str, Any] | None:
    """Greedy {…} slice + json.loads — tolerates accidental fences."""
    if not raw:
        return None
    import json

    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    first, last = text.find("{"), text.rfind("}")
    if first == -1 or last <= first:
        return None
    try:
        return json.loads(text[first : last + 1])
    except json.JSONDecodeError:
        return None


# ── Public entry point ──────────────────────────────────────────────────


def analyse(img: Image.Image) -> tuple[DiagramHint | None, str | None]:
    """Resolve backend, run detection. Returns (DiagramHint, None) or (None, reason).

    Backend env: `IMAGE_ANALYSER_DIAGRAM_BACKEND` ∈ {auto, heuristic, api, none}.
    Default `auto` runs the heuristic only — the API path must be opted into
    because it costs money (or at least a network round-trip).
    """
    backend = (os.getenv("IMAGE_ANALYSER_DIAGRAM_BACKEND") or "auto").strip().lower()

    if backend == "none":
        return None, "disabled by IMAGE_ANALYSER_DIAGRAM_BACKEND=none"

    if backend in ("auto", "heuristic"):
        return _heuristic(img), None

    if backend == "api":
        hint, reason = _api(img)
        if hint is None:
            # API failed — fall back to the heuristic so we still report *something*.
            heuristic = _heuristic(img)
            heuristic.signals["api_error"] = 1.0  # marker for "we tried and failed"
            return heuristic, reason
        return hint, None

    return None, f"unknown IMAGE_ANALYSER_DIAGRAM_BACKEND: {backend!r}"
