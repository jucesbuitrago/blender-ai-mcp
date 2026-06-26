# SPDX-FileCopyrightText: 2024-2026 Patryk Ciechański
# SPDX-License-Identifier: Apache-2.0

"""Lazy Supabase client for the script-memory (RAG) area.

The client is created on first use, not at import time, so the MCP server still
boots when Supabase is not configured. Credentials come from the same `.env`
file used by the rest of the app (``SUPABASE_URL`` / ``SUPABASE_KEY``).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any


@lru_cache(maxsize=1)
def get_supabase_client() -> Any:
    """Return a cached Supabase client, raising a clear error if unconfigured."""

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY in your .env."
        )

    try:
        from supabase import create_client
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "The 'supabase' package is not installed. Run: poetry add supabase"
        ) from exc

    return create_client(url, key)
