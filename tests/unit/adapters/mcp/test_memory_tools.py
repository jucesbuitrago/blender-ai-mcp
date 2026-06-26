# SPDX-FileCopyrightText: 2024-2026 Patryk Ciechański
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Supabase + pgvector script-memory tools."""

from __future__ import annotations

from types import SimpleNamespace

from server.adapters.mcp.areas import memory


class _FakeQuery:
    """Minimal PostgREST/RPC builder stub that records calls."""

    def __init__(self, recorder, response):
        self._recorder = recorder
        self._response = response

    def insert(self, payload):
        self._recorder["inserted"] = payload
        return self

    def execute(self):
        return self._response


class _FakeClient:
    def __init__(self, recorder, response):
        self._recorder = recorder
        self._response = response

    def table(self, name):
        self._recorder["table"] = name
        return _FakeQuery(self._recorder, self._response)

    def rpc(self, name, params):
        self._recorder["rpc"] = name
        self._recorder["rpc_params"] = params
        return _FakeQuery(self._recorder, self._response)


def _patch(monkeypatch, recorder, data):
    response = SimpleNamespace(data=data)
    monkeypatch.setattr(memory, "embed", lambda text: [0.1, 0.2, 0.3])
    monkeypatch.setattr(memory, "get_supabase_client", lambda: _FakeClient(recorder, response))


def test_buscar_referencias_uses_vector_rpc(monkeypatch):
    recorder = {}
    _patch(
        monkeypatch,
        recorder,
        data=[
            {
                "id": 7,
                "prompt_original": "a gear",
                "script_python": "import bpy",
                "notas": "20 teeth",
                "similarity": 0.91,
            }
        ],
    )

    result = memory.buscar_referencias(ctx=None, query="cogwheel")

    assert "import bpy" in result
    assert "id=7" in result
    assert "similarity=0.910" in result
    assert recorder["rpc"] == "match_blender_scripts"
    assert recorder["rpc_params"]["query_embedding"] == [0.1, 0.2, 0.3]
    assert recorder["rpc_params"]["match_count"] == memory._MATCH_COUNT


def test_buscar_referencias_no_match(monkeypatch):
    _patch(monkeypatch, {}, data=[])
    result = memory.buscar_referencias(ctx=None, query="nonexistent")
    assert "No validated scripts found" in result


def test_buscar_referencias_empty_query(monkeypatch):
    _patch(monkeypatch, {}, data=[])
    result = memory.buscar_referencias(ctx=None, query="   ")
    assert "No query provided" in result


def test_guardar_script_inserts_row_with_embedding(monkeypatch):
    recorder = {}
    _patch(monkeypatch, recorder, data=[{"id": 42}])

    result = memory.guardar_script(
        ctx=None,
        prompt_original="a cube",
        script_python="bpy.ops.mesh.primitive_cube_add()",
        notas="size 2",
    )

    assert "id=42" in result
    assert recorder["table"] == "blender_scripts"
    assert recorder["inserted"]["prompt_original"] == "a cube"
    assert recorder["inserted"]["embedding"] == [0.1, 0.2, 0.3]


def test_guardar_script_rejects_empty(monkeypatch):
    _patch(monkeypatch, {}, data=[])
    result = memory.guardar_script(ctx=None, prompt_original="x", script_python="   ")
    assert "empty" in result.lower()
