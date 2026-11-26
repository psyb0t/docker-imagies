from __future__ import annotations

import base64
import hashlib
import json
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagies.generation import Generation
from imagies.generation_models import Asset, download_asset
from imagies.server import build_app


def test_openai_routes_and_mcp(tmp_path) -> None:
    png = BytesIO()
    Image.new("RGB", (12, 12), "red").save(png, "PNG")
    calls = []

    def worker(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-sdcpp-cpu-sd-turbo"}]})
        if request.url.path == "/v1/images/generations":
            if json.loads(request.content)["model"] == "busy":
                return httpx.Response(503, json={"error": {"message": "model busy"}})
            return httpx.Response(
                200, json={"data": [{"b64_json": base64.b64encode(png.getvalue()).decode()}]}
            )
        return httpx.Response(200, json={"loaded": True})

    backend = Generation(transport=httpx.MockTransport(worker))
    headers = {"Authorization": "Bearer test-token"}
    with TestClient(
        build_app(tmp_path, "test-token", backend), base_url="http://localhost"
    ) as client:
        assert client.get("/healthz").is_success
        assert client.get("/v1/models").status_code == 401
        assert (
            client.get("/v1/models", headers=headers).json()["data"][0]["id"]
            == "local-sdcpp-cpu-sd-turbo"
        )
        for path in ("status", "load", "unload", "cancel"):
            method = "GET" if path == "status" else "POST"
            params = {"model": "sd-turbo"} if path == "load" else None
            response = client.request(method, f"/sdcpp/v1/{path}", headers=headers, params=params)
            assert response.is_success
            assert calls[-1].url.path == f"/sdcpp/v1/{path}"
            if params:
                assert calls[-1].url.params.get("model") == params["model"]
        args = {"model": "local-sdcpp-cpu-sd-turbo", "prompt": "A simple square", "size": "256x256"}
        result = client.post("/v1/images/generations", json={**args, "seed": 42}, headers=headers)
        assert result.status_code == 200, result.text
        assert json.loads(calls[-1].content)["seed"] == 42
        busy = client.post(
            "/v1/images/generations", json={**args, "model": "busy"}, headers=headers
        )
        assert busy.status_code == 503 and busy.json()["error"]["message"] == "model busy"
        previous = len(calls)
        bad = client.post("/v1/images/generations", json={**args, "n": "1"}, headers=headers)
        assert bad.status_code == 400 and len(calls) == previous
        tool = client.post(
            "/v1/mcp",
            headers={**headers, "Accept": "application/json, text/event-stream"},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "generate",
                    "arguments": args,
                },
            },
        )
        assert tool.status_code == 200, tool.text
        image_id = tool.json()["result"]["structuredContent"]["id"]
        assert client.get("/v1/images/" + image_id, headers=headers).content.startswith(b"\x89PNG")


def test_generation_disabled_without_worker(tmp_path) -> None:
    with TestClient(build_app(tmp_path)) as client:
        assert "/v1/images/generations" not in client.get("/openapi.json").json()["paths"]
        assert client.get("/v1/generation/engines").status_code == 404
        assert client.get("/healthz").status_code == 200


def test_engine_controls_and_legacy_aliases(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MODEL_PREFIX", "local-sdcpp-cuda-")
    calls = []
    model = "local-sdcpp-cuda-flux-schnell"

    def worker(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/status"):
            return httpx.Response(
                200,
                json={
                    "loaded": True,
                    "loading": False,
                    "generating": True,
                    "current_model": "flux-schnell",
                    "current_model_args": ["--model", "/models/flux.gguf"],
                    "process_pid": 42,
                    "models": ["flux-schnell"],
                    "idle_timeout": 300,
                },
            )
        if request.url.params.get("fail_if_busy") == "true":
            return httpx.Response(409, json={"error": "engine is loading or generating"})
        if request.url.params.get("model") == "missing":
            return httpx.Response(400, json={"error": "model is unavailable"})
        return httpx.Response(200, json={"ok": True})

    backend = Generation(transport=httpx.MockTransport(worker))
    root = "/v1/generation/engines"
    engine = root + "/sdcpp"
    with TestClient(build_app(tmp_path, "test-token", backend)) as client:
        assert client.get(root).status_code == 401
        client.headers["Authorization"] = "Bearer test-token"
        assert client.get(root).json() == [{"id": "sdcpp"}]
        assert client.get(root + "/missing").status_code == 404
        assert client.put(engine + "/model", json={"model": " "}).status_code == 422
        assert not calls
        assert client.get(engine).json() == {
            "id": "sdcpp",
            "loaded": True,
            "loading": False,
            "generating": True,
            "resident_model": model,
            "models": [model],
            "idle_timeout_seconds": 300,
        }
        assert client.put(engine + "/model", json={"model": model}).status_code == 204
        assert calls[-1].method == "POST"
        assert calls[-1].url.path == "/sdcpp/v1/load"
        assert calls[-1].url.params["model"] == model
        missing = client.put(engine + "/model", json={"model": "missing"})
        assert missing.status_code == 400
        assert missing.json() == {"detail": "unknown model for this generation engine"}
        assert client.delete(engine + "/model").status_code == 409
        assert client.post("/sdcpp/v1/unload").json() == {"ok": True}
        assert client.delete(engine + "/active").status_code == 204
        assert calls[-1].url.path == "/sdcpp/v1/cancel"
        paths = client.get("/openapi.json").json()["paths"]
        assert "put" in paths[root + "/{engine}/model"]
        assert all(not path.startswith("/sdcpp/") for path in paths)


def test_download_integrity_and_existing_weight(tmp_path) -> None:
    payload = b"synthetic weight file"
    asset = Asset(
        "model.gguf", "example/model", "revision", len(payload), hashlib.sha256(payload).hexdigest()
    )
    calls = []

    def origin(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=payload)

    with httpx.Client(transport=httpx.MockTransport(origin)) as client:
        download_asset(client, tmp_path, asset, "")
        download_asset(client, tmp_path, asset, "")
    path = tmp_path / asset.name
    assert path.is_file() and not path.is_symlink() and path.read_bytes() == payload
    assert len(calls) == 1 and calls[0].url.path == "/example/model/resolve/revision/model.gguf"
    assert calls[0].headers.get("authorization") is None


def test_download_discards_partial_file_and_respects_gating(tmp_path) -> None:
    asset = Asset("gated.gguf", "example/model", "revision", 3, "bad-checksum", gated=True)

    def origin(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("authorization") == "Bearer test-token"
        return httpx.Response(200, content=b"bad")

    with httpx.Client(transport=httpx.MockTransport(origin)) as client:
        with pytest.raises(ValueError, match="HF_TOKEN"):
            download_asset(client, tmp_path, asset, "")
        with pytest.raises(ValueError, match="integrity"):
            download_asset(client, tmp_path, asset, "test-token")
    assert list(tmp_path.iterdir()) == []
