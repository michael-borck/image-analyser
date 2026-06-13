"""Image embedding via the family's shared helper (lens-embed, CLIP).

A single pinned CLIP model across the family means this vector is comparable to
other members' image vectors — for visual similarity / cohort distinctiveness.
Opt-in and degradable: install the [embeddings] extra to populate it; without
it (or on any failure) this returns None.
"""

from __future__ import annotations

from typing import Any


def embed_image_vector(source: Any) -> list[float] | None:
    """CLIP image vector (from path/bytes/PIL image), or None if off."""
    try:
        from lens_embed import backend_available, embed_image
    except ImportError:
        return None
    if not backend_available("image"):
        return None
    try:
        return embed_image(source)
    except Exception:
        return None
