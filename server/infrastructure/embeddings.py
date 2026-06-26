# SPDX-FileCopyrightText: 2024-2026 Patryk Ciechański
# SPDX-License-Identifier: Apache-2.0

"""Local text embeddings for the script-memory (RAG) area.

Uses a small local sentence-transformers model so no embedding API key or
network call is required. The model is loaded lazily and cached for the life of
the process (~150 MB resident, CPU-only, a few ms per short query).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, List

# 384-dimensional; must match the vector(N) column in the SQL migration.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384


@lru_cache(maxsize=1)
def _get_model() -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "sentence-transformers is not installed. Run: poetry install"
        ) from exc
    return SentenceTransformer(EMBEDDING_MODEL)


def embed(text: str) -> List[float]:
    """Return a normalized embedding vector for `text` (cosine-ready)."""

    model = _get_model()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()
