from __future__ import annotations

import os
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from imagies import models
from imagies.model_process import ModelProcess
from imagies.server import build_app


class FakeVisionWorker:
    def __init__(self):
        self.models = set()

    async def execute(self, operation, paths, value):
        assert all(path.is_file() for path in paths)
        if operation == "match":
            self.models.add("clip")
            return [
                {"matches": [{"query": value[0], "score": 0.7}], "pid": os.getpid()} for _ in paths
            ]
        assert value == pytest.approx(0.45)
        self.models.add("yolo")
        return [
            {"detections": [{"label": "test", "confidence": 0.8, "box": [0, 0, 0.5, 0.5]}]}
            for _ in paths
        ]

    def resident(self):
        return sorted(self.models)

    async def close(self):
        return None


def png(color: str = "red") -> bytes:
    with BytesIO() as out:
        Image.new("RGB", (64, 48), color).save(out, "PNG")
        return out.getvalue()


def test_images_roundtrip(tmp_path) -> None:
    with TestClient(build_app(tmp_path)) as client:
        first = client.post("/v1/images", content=png())
        second = client.post("/v1/images", content=png("blue"))
        assert first.status_code == second.status_code == 201
        image_id = first.json()["id"]
        edited = client.post(
            "/v1/images/transform",
            json={
                "image_id": image_id,
                "crop": [0, 0, 32, 16],
                "width": 20,
                "height": 20,
                "fit": "pad",
                "format": "webp",
            },
        )
        assert edited.status_code == 200, edited.text
        response = client.get("/v1/images/" + edited.json()["id"])
        assert response.headers["content-type"] == "image/webp"
        with Image.open(BytesIO(response.content)) as image:
            assert image.size == (20, 20)
        montage = client.post(
            "/v1/images/montage",
            json={
                "image_ids": [image_id, second.json()["id"]],
                "columns": 2,
                "cell_width": 40,
                "cell_height": 30,
            },
        )
        assert montage.status_code == 200, montage.text
        assert (montage.json()["width"], montage.json()["height"]) == (80, 30)
        assert client.delete("/v1/images/" + image_id).status_code == 204
        assert client.get("/v1/images/" + image_id).status_code == 404


def test_auth_and_bad_input(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("imagies.server.MAX_UPLOAD_BYTES", 1024)
    with TestClient(build_app(tmp_path, "test-token")) as client:
        assert client.get("/healthz").is_success
        assert client.get("/v1/engines").status_code == 401
        assert client.post("/v1/mcp").status_code == 401
        assert client.post("/v1/unload").status_code == 401
        auth = {"Authorization": "Bearer test-token"}
        assert client.post("/v1/images", content=b"bad", headers=auth).status_code == 400
        assert client.post("/v1/images", content=b"x" * 1025, headers=auth).status_code == 413
        assert (
            client.post(
                "/v1/images/transform", json={"image_id": "../secret"}, headers=auth
            ).status_code
            == 422
        )
        image_id = client.post("/v1/images", content=png(), headers=auth).json()["id"]
        response = client.post(
            "/v1/images/transform",
            json={"image_id": image_id, "crop": [63, 0, 2, 10]},
            headers=auth,
        )
        assert response.status_code == 400
        assert (
            client.post(
                "/v1/images/match",
                json={"image_ids": [image_id], "queries": ["something"]},
                headers=auth,
            ).status_code
            == 404
        )


def test_mcp_and_analysis_contract(tmp_path) -> None:
    app = build_app(tmp_path, vision_worker=ModelProcess(FakeVisionWorker))
    with TestClient(app, base_url="http://localhost") as client:
        assert client.post("/v1/unload").json() == {"unloaded": []}
        image_id = client.post("/v1/images", content=png()).json()["id"]
        payload = {"image_ids": [image_id], "queries": ["example"]}
        matched = client.post("/v1/images/match", json=payload).json()["images"][0]
        assert matched["image_id"] == image_id
        pid = matched["pid"]
        assert pid != os.getpid()
        result = client.post(
            "/v1/images/detect", json={"image_ids": [image_id], "confidence": 0.45}
        )
        assert result.json()["images"][0]["detections"][0]["box"] == [0, 0, 0.5, 0.5]
        reply = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "match", "arguments": {"spec": payload}},
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert reply.status_code == 200, reply.text
        assert reply.json()["result"]["structuredContent"]["images"][0]["image_id"] == image_id
        assert client.post("/v1/unload").json() == {"unloaded": ["clip", "yolo"]}
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        assert client.post("/v1/unload").json() == {"unloaded": []}
        assert client.post("/v1/images/match", json=payload).json()["images"][0]["pid"] != pid


@pytest.mark.parametrize("mode", ["startup", "on-demand"])
def test_clip_download_timing_and_reuse(tmp_path, monkeypatch, mode) -> None:
    monkeypatch.setenv("IMAGIES_MODEL_DOWNLOAD", mode)
    monkeypatch.setattr(models, "_CLIP_FILES", ("config.json",))
    fetched = []

    def fetch(url, **_options):
        fetched.append(url)
        return BytesIO(b"{}")

    monkeypatch.setattr(models, "urlopen", fetch)
    app = build_app(tmp_path, vision_worker=ModelProcess(FakeVisionWorker))
    with TestClient(app, base_url="http://localhost") as client:
        assert len(fetched) == (1 if mode == "startup" else 0)
        image_id = client.post("/v1/images", content=png()).json()["id"]
        payload = {"image_ids": [image_id], "queries": ["example"]}
        reply = client.post(
            "/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "match", "arguments": {"spec": payload}},
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
        assert reply.json()["result"]["structuredContent"]["images"][0]["image_id"] == image_id
        assert len(fetched) == 1
        assert client.post("/v1/images/match", json=payload).is_success
        assert len(fetched) == 1
