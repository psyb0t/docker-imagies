from __future__ import annotations

import asyncio
import base64
import binascii
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from http import HTTPStatus
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from imagies.images import MAX_UPLOAD_BYTES, Asset, Store

_log = logging.getLogger(__name__)
_WORKER_ROOT = "http://127.0.0.1:17234"
_WORKER_READ_TIMEOUT = 900.0
_WORKER_START_TIMEOUT = 12.0
_WORKER_STOP_TIMEOUT = 25.0
_MAX_REQUEST_BYTES = 10 * 1024 * 1024
_GENERATE = "/v1/images/generations"
_MODEL_CONTROL = "/sdcpp/v1"


class GenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str = Field(min_length=1, max_length=160, description="Model ID from GET /v1/models.")
    prompt: str = Field(min_length=1, max_length=16384)
    n: int = Field(default=1, strict=True, ge=1, le=4)
    size: str | None = Field(default=None, pattern=r"^[1-9]\d{0,3}x[1-9]\d{0,3}$|^auto$")
    response_format: Literal["b64_json", "url"] | None = None


class Generation:
    def __init__(
        self,
        executable: Path = Path("/usr/local/bin/imagies-generation-worker"),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._executable = executable
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._process: asyncio.subprocess.Process | None = None

    @property
    def ready(self) -> bool:
        if self._transport is not None:
            return self._client is not None
        return self._process is not None and self._process.returncode is None

    @asynccontextmanager
    async def run(self) -> AsyncIterator[None]:
        async with httpx.AsyncClient(
            base_url=_WORKER_ROOT,
            timeout=httpx.Timeout(_WORKER_READ_TIMEOUT, connect=3.0),
            trust_env=False,
            transport=self._transport,
        ) as client:
            self._client = client
            try:
                if self._transport is None:
                    self._process = await asyncio.create_subprocess_exec(str(self._executable))
                    await self._wait_ready(client)
                yield
            finally:
                self._client = None
                if self._process is not None and self._process.returncode is None:
                    with suppress(ProcessLookupError):
                        self._process.terminate()
                    try:
                        await asyncio.wait_for(self._process.wait(), timeout=_WORKER_STOP_TIMEOUT)
                    except TimeoutError:
                        with suppress(ProcessLookupError):
                            self._process.kill()
                        await self._process.wait()
                self._process = None

    async def _wait_ready(self, client: httpx.AsyncClient) -> None:
        deadline = asyncio.get_running_loop().time() + _WORKER_START_TIMEOUT
        while self.ready and asyncio.get_running_loop().time() < deadline:
            with suppress(httpx.RequestError):
                if (await client.get("/health", timeout=1)).is_success:
                    return
            await asyncio.sleep(0.1)
        raise RuntimeError("image generation worker failed to start")

    async def request(
        self,
        method: str,
        endpoint: str,
        *,
        options: GenerationRequest | None = None,
        model_key: str | None = None,
    ) -> httpx.Response:
        client = self._client
        if client is None or not self.ready:
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE, "image generation worker is offline"
            )
        try:
            return await client.request(
                method,
                endpoint,
                params={"model": model_key} if model_key is not None else None,
                json=options.model_dump(exclude_none=True, exclude_unset=True) if options else None,
            )
        except httpx.TimeoutException as exc:
            raise HTTPException(
                HTTPStatus.GATEWAY_TIMEOUT, "image generation worker timed out"
            ) from exc
        except httpx.RequestError as exc:
            _log.warning(
                "image generation worker request failed", extra={"reason": type(exc).__name__}
            )
            raise HTTPException(
                HTTPStatus.BAD_GATEWAY, "image generation worker is unreachable"
            ) from exc


def _relay(upstream: httpx.Response) -> Response:
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def attach_generation(app: FastAPI, mcp: FastMCP, store: Store, backend: Generation) -> None:
    @app.get("/v1/models", description="List local image generation model IDs.")
    async def models() -> Response:
        return _relay(await backend.request("GET", "/v1/models"))

    @app.get(f"{_MODEL_CONTROL}/status", description="Read the generation worker and model state.")
    async def status() -> Response:
        return _relay(await backend.request("GET", f"{_MODEL_CONTROL}/status"))

    @app.post(f"{_MODEL_CONTROL}/load", description="Load a model by short key.")
    async def load(model: str | None = None) -> Response:
        return _relay(await backend.request("POST", f"{_MODEL_CONTROL}/load", model_key=model))

    @app.post(f"{_MODEL_CONTROL}/unload", description="Unload the idle model and free memory.")
    async def unload() -> Response:
        return _relay(await backend.request("POST", f"{_MODEL_CONTROL}/unload"))

    @app.post(f"{_MODEL_CONTROL}/cancel", description="Cancel the current generation.")
    async def cancel() -> Response:
        return _relay(await backend.request("POST", f"{_MODEL_CONTROL}/cancel"))

    @app.post(
        _GENERATE,
        description="Generate an image using the OpenAI-compatible image API.",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/json": {"schema": GenerationRequest.model_json_schema()}},
            }
        },
    )
    async def generate_image(request: Request) -> Response:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > _MAX_REQUEST_BYTES:
                raise HTTPException(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "image request exceeds 10 MiB"
                )
            body.extend(chunk)
        try:
            options = GenerationRequest.model_validate_json(body)
        except ValidationError as exc:
            raise HTTPException(HTTPStatus.BAD_REQUEST, "invalid image generation request") from exc
        return _relay(await backend.request("POST", _GENERATE, options=options))

    @mcp.tool()
    async def generate(model: str, prompt: str, size: str | None = None) -> Asset:
        """Generate one image and stage it for download from the image API."""
        options = GenerationRequest(
            model=model, prompt=prompt, size=size, n=1, response_format="b64_json"
        )
        upstream = await backend.request("POST", _GENERATE, options=options)
        if not upstream.is_success:
            raise ValueError(f"image generation failed (HTTP {upstream.status_code})")
        try:
            data = upstream.json()["data"][0]["b64_json"]
            if len(data) > 4 * ((MAX_UPLOAD_BYTES + 2) // 3):
                raise ValueError("image generation output exceeds staging limit")
            content = base64.b64decode(data, validate=True)
        except (KeyError, IndexError, TypeError, ValueError, binascii.Error) as exc:
            raise ValueError("generation worker returned an invalid image") from exc
        if len(content) > MAX_UPLOAD_BYTES:
            raise ValueError("image generation output exceeds staging limit")
        return await run_in_threadpool(store.ingest, content)
