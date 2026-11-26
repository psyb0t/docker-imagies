from __future__ import annotations

import hmac
import os
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.concurrency import run_in_threadpool

from imagies.constants import DATA_DIR
from imagies.errors import ImageMissing, InvalidImage, ModelUnavailable
from imagies.generation import Generation, attach_generation
from imagies.images import (
    MAX_UPLOAD_BYTES,
    Asset,
    Detect,
    ImageId,
    Match,
    Montage,
    Store,
    Transform,
)
from imagies.vision import Vision

_SERVICE_TAGS = ["Service"]
_IMAGE_TAGS = ["Images"]
_ANALYSIS_TAGS = ["Analysis"]


def build_app(
    data_dir: Path, auth_token: str = "", generation: Generation | None = None
) -> FastAPI:
    store = Store(data_dir / "images")
    vision = Vision(data_dir / "models")
    mcp = FastMCP(
        "imagies",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(
            allowed_hosts=[
                "localhost",
                "localhost:*",
                "127.0.0.1",
                "127.0.0.1:*",
                "[::1]",
                "[::1]:*",
                *(
                    host.strip()
                    for host in os.environ.get("IMAGIES_MCP_ALLOWED_HOSTS", "").split(",")
                    if host.strip()
                ),
            ],
            allowed_origins=["http://localhost:*", "http://127.0.0.1:*", "http://[::1]:*"],
        ),
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        async with AsyncExitStack() as stack:
            if generation is not None:
                await stack.enter_async_context(generation.run())
            await stack.enter_async_context(mcp.session_manager.run())
            yield

    app = FastAPI(title="Imagies", version="0.2.1", lifespan=lifespan)

    @app.middleware("http")
    async def authorize(request: Request, call_next: Any) -> Response:
        if auth_token and request.url.path not in ("/healthz", "/health"):
            scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                supplied.encode(), auth_token.encode()
            ):
                return JSONResponse(
                    {"detail": "Bearer token required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
        return await call_next(request)

    @app.exception_handler(InvalidImage)
    async def invalid_image(_request: Request, error: InvalidImage) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(ImageMissing)
    @app.exception_handler(ModelUnavailable)
    async def unavailable(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=404)

    @app.get("/health", include_in_schema=False)
    @app.get(
        "/healthz", tags=_SERVICE_TAGS, description="Process health. No authentication required."
    )
    async def health() -> dict[str, str]:
        if generation is not None and not generation.ready:
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE, "image generation worker is offline"
            )
        return {"status": "ok"}

    @app.get(
        "/v1/engines",
        tags=_SERVICE_TAGS,
        description="List models with installed dependencies and local weights.",
    )
    async def engines() -> dict[str, list[str]]:
        return {"engines": vision.available()}

    @app.post(
        "/v1/images",
        tags=_IMAGE_TAGS,
        status_code=201,
        description="Stage a PNG, JPEG or WebP still image up to 20 MiB as raw request bytes.",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    mime: {"schema": {"type": "string", "format": "binary"}}
                    for mime in (
                        "image/png",
                        "image/jpeg",
                        "image/webp",
                        "application/octet-stream",
                    )
                },
            }
        },
    )
    async def upload(request: Request) -> Asset:
        payload = bytearray()
        async for chunk in request.stream():
            if len(payload) + len(chunk) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, "image exceeds the upload limit")
            payload.extend(chunk)
        return await run_in_threadpool(store.ingest, bytes(payload))

    @app.get(
        "/v1/images/{image_id}", tags=_IMAGE_TAGS, description="Download a staged image by ID."
    )
    async def download(image_id: ImageId) -> FileResponse:
        path = store.path(image_id)
        return FileResponse(path, media_type=f"image/{store.format(image_id)}")

    @app.delete(
        "/v1/images/{image_id}",
        tags=_IMAGE_TAGS,
        status_code=204,
        description="Delete a staged image.",
    )
    async def delete(image_id: ImageId) -> Response:
        store.path(image_id).unlink(missing_ok=True)
        return Response(status_code=204)

    @app.post(
        "/v1/images/transform",
        tags=_IMAGE_TAGS,
        description="Crop, resize and encode a staged image.",
    )
    @mcp.tool()
    async def transform(spec: Transform) -> Asset:
        """Crop, resize, or convert one staged image. Returns the new image ID and dimensions."""
        return await run_in_threadpool(store.transform, spec)

    @app.post(
        "/v1/images/montage",
        tags=_IMAGE_TAGS,
        description="Lay out staged still images in a single grid.",
    )
    @mcp.tool()
    async def montage(spec: Montage) -> Asset:
        """Arrange staged images into a grid in input order. Returns the staged grid image ID."""
        return await run_in_threadpool(store.montage, spec)

    @app.post(
        "/v1/images/match",
        tags=_ANALYSIS_TAGS,
        description="Rank text queries against each staged image with CLIP.",
    )
    @mcp.tool()
    async def match(spec: Match) -> dict[str, list[dict[str, Any]]]:
        """Rank text queries for each staged image by CLIP similarity. Requires local CLIP weights."""
        paths = store.paths(spec.image_ids)
        results = await run_in_threadpool(vision.match, paths, spec.queries)
        return {
            "images": [
                {"image_id": image_id, **result}
                for image_id, result in zip(spec.image_ids, results, strict=True)
            ]
        }

    @app.post(
        "/v1/images/detect",
        tags=_ANALYSIS_TAGS,
        description="Return YOLO object labels, scores and normalized boxes.",
    )
    @mcp.tool()
    async def detect(spec: Detect) -> dict[str, list[dict[str, Any]]]:
        """Find objects in staged images. Returns labels, scores, and normalized boxes."""
        paths = store.paths(spec.image_ids)
        results = await run_in_threadpool(vision.detect, paths, spec.confidence)
        return {
            "images": [
                {"image_id": image_id, **result}
                for image_id, result in zip(spec.image_ids, results, strict=True)
            ]
        }

    if generation is not None:
        attach_generation(app, mcp, store, generation)
    app.mount("/v1/mcp", mcp.streamable_http_app())
    return app


app = build_app(
    DATA_DIR,
    os.environ.get("IMAGIES_AUTH_TOKEN", ""),
    Generation() if os.environ.get("IMAGIES_GENERATION") == "1" else None,
)
