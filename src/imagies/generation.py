from __future__ import annotations

import asyncio
import base64
import binascii
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from http import HTTPStatus
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi import Path as PathParam
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
_WORKER_CONTROL = "/sdcpp/v1"
_ENGINES = "/v1/generation/engines"
_ENGINE = f"{_ENGINES}/{{engine}}"
_SDCPP = "sdcpp"
_GENERATION_TAGS = ["Generation"]
EngineID = Annotated[
    str, PathParam(description="ID from GET /v1/generation/engines.", examples=[_SDCPP])
]


class ModelSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(
        min_length=1,
        max_length=160,
        pattern=r"^\S+$",
        description="Generation model ID from GET /v1/models.",
        examples=["imagies-sd-turbo"],
    )


class GenerationEngine(BaseModel):
    id: str = Field(description="Engine ID used in generation control routes.", examples=[_SDCPP])


class EngineState(GenerationEngine):
    loaded: bool
    loading: bool
    generating: bool
    resident_model: str | None
    models: list[str]
    idle_timeout_seconds: int


class _WorkerState(BaseModel):
    loaded: bool
    loading: bool
    generating: bool
    current_model: str
    models: list[str]
    idle_timeout: int


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
        self._model_prefix = os.environ.get("MODEL_PREFIX", "imagies-")
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
        fail_if_busy: bool = False,
    ) -> httpx.Response:
        client = self._client
        if client is None or not self.ready:
            raise HTTPException(
                HTTPStatus.SERVICE_UNAVAILABLE, "image generation worker is offline"
            )
        try:
            params = {}
            if model_key is not None:
                params["model"] = model_key
            if fail_if_busy:
                params["fail_if_busy"] = "true"
            return await client.request(
                method,
                endpoint,
                params=params,
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

    async def engine_state(self) -> EngineState:
        upstream = await self.request("GET", f"{_WORKER_CONTROL}/status")
        try:
            upstream.raise_for_status()
            state = _WorkerState.model_validate_json(upstream.content)
        except (httpx.HTTPStatusError, ValidationError) as exc:
            raise HTTPException(
                HTTPStatus.BAD_GATEWAY, "image generation worker returned an invalid status"
            ) from exc
        return EngineState(
            id=_SDCPP,
            loaded=state.loaded,
            loading=state.loading,
            generating=state.generating,
            resident_model=self._model_prefix + state.current_model if state.loaded else None,
            models=[self._model_prefix + key for key in state.models],
            idle_timeout_seconds=state.idle_timeout,
        )

    async def load_model(self, model: str) -> httpx.Response:
        return await self.request("POST", f"{_WORKER_CONTROL}/load", model_key=model)


def _relay(upstream: httpx.Response) -> Response:
    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )


def _require_engine(engine: str) -> None:
    if engine != _SDCPP:
        raise HTTPException(HTTPStatus.NOT_FOUND, "unknown generation engine")


def _control_result(upstream: httpx.Response) -> Response:
    if upstream.is_success:
        return Response(status_code=HTTPStatus.NO_CONTENT)
    if upstream.status_code == HTTPStatus.BAD_REQUEST:
        raise HTTPException(HTTPStatus.BAD_REQUEST, "unknown model for this generation engine")
    if upstream.status_code == HTTPStatus.CONFLICT:
        raise HTTPException(HTTPStatus.CONFLICT, "generation engine is busy")
    raise HTTPException(
        HTTPStatus.SERVICE_UNAVAILABLE, "generation engine could not complete the operation"
    )


def attach_generation(app: FastAPI, mcp: FastMCP, store: Store, backend: Generation) -> None:
    @app.get(
        "/v1/models", tags=_GENERATION_TAGS, description="List local image generation model IDs."
    )
    async def models() -> Response:
        return _relay(await backend.request("GET", "/v1/models"))

    @app.get(
        _ENGINES, tags=_GENERATION_TAGS, description="List the configured image generation engines."
    )
    async def engines() -> list[GenerationEngine]:
        return [GenerationEngine(id=_SDCPP)]

    @app.get(
        _ENGINE,
        tags=_GENERATION_TAGS,
        description="Read the engine state and available model IDs.",
        responses={404: {"description": "Unknown generation engine."}},
    )
    async def engine_state(engine: EngineID) -> EngineState:
        _require_engine(engine)
        return await backend.engine_state()

    @app.put(
        f"{_ENGINE}/model",
        tags=_GENERATION_TAGS,
        status_code=HTTPStatus.NO_CONTENT,
        description="Load the selected model and wait until the engine is ready.",
        responses={
            400: {"description": "The engine does not serve the requested model."},
            404: {"description": "Unknown generation engine."},
            503: {"description": "The engine could not load the model."},
        },
    )
    async def select_model(engine: EngineID, selection: ModelSelection) -> Response:
        _require_engine(engine)
        return _control_result(await backend.load_model(selection.model))

    @app.delete(
        f"{_ENGINE}/model",
        tags=_GENERATION_TAGS,
        status_code=HTTPStatus.NO_CONTENT,
        description="Unload the resident model. Returns 409 while the engine is busy.",
        responses={
            404: {"description": "Unknown generation engine."},
            409: {"description": "The engine is loading or generating."},
        },
    )
    async def unload_model(engine: EngineID) -> Response:
        _require_engine(engine)
        upstream = await backend.request("POST", f"{_WORKER_CONTROL}/unload", fail_if_busy=True)
        return _control_result(upstream)

    @app.delete(
        f"{_ENGINE}/active",
        tags=_GENERATION_TAGS,
        status_code=HTTPStatus.NO_CONTENT,
        description="Cancel the engine's active generation. Succeeds if it is already idle.",
        responses={404: {"description": "Unknown generation engine."}},
    )
    async def cancel_generation(engine: EngineID) -> Response:
        _require_engine(engine)
        return _control_result(await backend.request("POST", f"{_WORKER_CONTROL}/cancel"))

    @app.get(f"{_WORKER_CONTROL}/status", include_in_schema=False, deprecated=True)
    async def status() -> Response:
        return _relay(await backend.request("GET", f"{_WORKER_CONTROL}/status"))

    @app.post(f"{_WORKER_CONTROL}/load", include_in_schema=False, deprecated=True)
    async def load(model: str | None = None) -> Response:
        return _relay(await backend.request("POST", f"{_WORKER_CONTROL}/load", model_key=model))

    @app.post(f"{_WORKER_CONTROL}/unload", include_in_schema=False, deprecated=True)
    async def unload() -> Response:
        return _relay(await backend.request("POST", f"{_WORKER_CONTROL}/unload"))

    @app.post(f"{_WORKER_CONTROL}/cancel", include_in_schema=False, deprecated=True)
    async def cancel() -> Response:
        return _relay(await backend.request("POST", f"{_WORKER_CONTROL}/cancel"))

    @app.post(
        _GENERATE,
        tags=_GENERATION_TAGS,
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
