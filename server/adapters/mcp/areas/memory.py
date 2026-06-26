# SPDX-FileCopyrightText: 2024-2026 Patryk Ciechański
# SPDX-License-Identifier: Apache-2.0

"""Script-memory (RAG) tools backed by Supabase + pgvector.

These tools let the LLM reuse previously validated Blender Python scripts.
``buscar_referencias`` runs a semantic (embedding) similarity search over the
``blender_scripts`` table before new code is written, and ``guardar_script``
persists an approved script (with its embedding) for future reuse.

Unlike the other MCP areas, these tools do not talk to Blender over RPC — they
embed text locally and query Supabase directly — so they skip the router/RPC
machinery. See scripts/sql/blender_scripts_embeddings.sql for the schema.
"""

from __future__ import annotations

from typing import Any, Dict

from fastmcp import Context

from server.adapters.mcp.areas._registration import register_existing_tools
from server.adapters.mcp.visibility.tags import get_capability_tags
from server.infrastructure.embeddings import embed
from server.infrastructure.supabase_client import get_supabase_client

MEMORY_PUBLIC_TOOL_NAMES = (
    "buscar_referencias",
    "guardar_script",
)

_TABLE = "blender_scripts"
_MATCH_RPC = "match_blender_scripts"
_MATCH_COUNT = 5


def register_memory_tools(target: Any) -> Dict[str, Any]:
    """Register the Supabase-backed script-memory tools on a server/provider."""

    return register_existing_tools(
        globals(),
        target,
        MEMORY_PUBLIC_TOOL_NAMES,
        tags=get_capability_tags("memory"),
    )


def buscar_referencias(ctx: Context, query: str) -> str:
    """
    [MEMORY][SAFE] Semantic search of the script memory for a validated script.

    Call this BEFORE writing any new Python: it embeds `query` and returns prior
    scripts whose meaning is closest (even with different wording), so you can
    reuse or adapt the stored `script_python` instead of starting from scratch.

    Args:
        query: Free-text description of the geometry/task to look up.
    """
    term = query.strip()
    if not term:
        return "No query provided. Nothing to search."

    try:
        embedding = embed(term)
        client = get_supabase_client()
        response = client.rpc(
            _MATCH_RPC,
            {"query_embedding": embedding, "match_count": _MATCH_COUNT},
        ).execute()
    except RuntimeError as exc:
        return str(exc)

    rows = getattr(response, "data", None) or []
    if not rows:
        return f"No validated scripts found for '{query}'. Proceed to write new code."

    blocks = []
    for row in rows:
        prompt = row.get("prompt_original") or "(no prompt)"
        notas = row.get("notas") or ""
        script = row.get("script_python") or ""
        similarity = row.get("similarity")
        header = f"# Reference (id={row.get('id')}"
        if isinstance(similarity, (int, float)):
            header += f", similarity={similarity:.3f}"
        header += f"): {prompt}"
        if notas:
            header += f"\n# Notes: {notas}"
        blocks.append(f"{header}\n{script}")

    return f"Found {len(rows)} reference script(s):\n\n" + "\n\n---\n\n".join(blocks)


def guardar_script(
    ctx: Context,
    prompt_original: str,
    script_python: str,
    notas: str = "",
) -> str:
    """
    [MEMORY] Stores an approved Blender Python script in the script memory.

    Call this AFTER a script has been validated/approved so future requests can
    reuse it via `buscar_referencias`. The prompt and notes are embedded so the
    script becomes findable by semantic similarity.

    Args:
        prompt_original: The original user prompt that produced the script.
        script_python: The validated Python script to store.
        notas: Optional notes (caveats, parameters, tags) for later retrieval.
    """
    if not script_python.strip():
        return "Refusing to store an empty script."

    try:
        embedding = embed(f"{prompt_original}\n{notas}".strip())
        client = get_supabase_client()
        response = (
            client.table(_TABLE)
            .insert(
                {
                    "prompt_original": prompt_original,
                    "script_python": script_python,
                    "notas": notas,
                    "embedding": embedding,
                }
            )
            .execute()
        )
    except RuntimeError as exc:
        return str(exc)

    rows = getattr(response, "data", None) or []
    new_id = rows[0].get("id") if rows else None
    return f"Stored script in memory (id={new_id})." if new_id else "Stored script in memory."
