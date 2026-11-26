from __future__ import annotations

import fcntl
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx

_HUB = "https://huggingface.co"
_CHUNK_BYTES = 1024 * 1024
_ASSET_ROOT = "/models/"
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Asset:
    name: str
    repo: str
    revision: str
    size: int
    sha256: str
    remote_name: str | None = None
    gated: bool = False

    @property
    def url(self) -> str:
        return f"{_HUB}/{self.repo}/resolve/{self.revision}/{self.remote_name or self.name}"


CPU_ASSETS = (
    Asset(
        "sd_turbo-f16-q8_0.gguf",
        "Green-Sky/SD-Turbo-GGUF",
        "19a31586d02d64a73b4419bc193b3ecfaf38e1f0",
        2023745376,
        "d50be7655f0a554cf8041c145d88b210bd5f3c545423119dee62ae08cae51580",
    ),
    Asset(
        "sd_xl_turbo_1.0.q8_0.gguf",
        "OlegSkutte/sdxl-turbo-GGUF",
        "62a28b8b2c3cb4bf76d9aad45a5b798047dc786c",
        4098988672,
        "f5659e6cdad699b1432870cc9fe811359e45c8a331687be505d0ba05c8364720",
    ),
)
CUDA_ASSETS = (
    *CPU_ASSETS,
    Asset(
        "sdxl_lightning_4step.q8_0.gguf",
        "mzwing/SDXL-Lightning-GGUF",
        "4fe21f71b298e088be341bb6545f6a2df950b5ff",
        4098988672,
        "3a8b2ea93d0b0c5e72c82ed518aea715203173e2afbf8b7e47335d7f8799ef78",
    ),
    Asset(
        "flux1-schnell-Q4_0.gguf",
        "leejet/FLUX.1-schnell-gguf",
        "6aed6e76e52ba97c0050cbb5b595bfb779a19545",
        6884606880,
        "4f30741d2bfc786c92934ce925fcb0a43df3441e76504b797c3d5d5f0878fa6f",
        remote_name="flux1-schnell-q4_0.gguf",
    ),
    Asset(
        "ae.safetensors",
        "black-forest-labs/FLUX.1-schnell",
        "169573f54945f633efbe126a40a5676b4228363c",
        335304388,
        "afc8e28272cd15db3919bacdb6918ce9c1ed22e96cb12c4d5ed0fba823529e38",
        gated=True,
    ),
    Asset(
        "clip_l.safetensors",
        "comfyanonymous/flux_text_encoders",
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5",
        246144152,
        "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd",
    ),
    Asset(
        "t5xxl_fp16.safetensors",
        "comfyanonymous/flux_text_encoders",
        "6af2a98e3f615bdfa612fbd85da93d1ed5f69ef5",
        9787841024,
        "6e480b09fae049a72d2a8c5fbccb8d3e92febeb233bbe9dfe7256958a9167635",
    ),
    Asset(
        "Juggernaut-XI-byRunDiffusion.safetensors",
        "RunDiffusion/Juggernaut-XI-v11",
        "f836518c7a75f47a9e26bc661ff9509ef2ceb7c0",
        7105350536,
        "33e58e86686f6b386c526682b5da9228ead4f91d994abd4b053442dc5b42719e",
        gated=True,
    ),
)


def download_asset(client: httpx.Client, directory: Path, asset: Asset, token: str) -> None:
    destination = directory / asset.name
    if destination.is_symlink():
        raise ValueError(f"{asset.name} must be a regular file")
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != asset.size:
            raise ValueError(f"{asset.name} does not match the expected size")
        return
    if asset.gated and not token:
        raise ValueError(f"{asset.name} requires HF_TOKEN and access to {asset.repo}")
    auth = {"Authorization": f"Bearer {token}"} if asset.gated else {}
    digest = hashlib.sha256()
    size = 0
    with NamedTemporaryFile(dir=directory, prefix=".download-", delete=False) as staged:
        temporary = Path(staged.name)
        try:
            with client.stream(
                "GET", asset.url, headers={**auth, "Accept-Encoding": "identity"}
            ) as response:
                if response.status_code != 200:
                    raise ValueError(
                        f"download of {asset.name} returned HTTP {response.status_code}"
                    )
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    size += len(chunk)
                    if size > asset.size:
                        raise ValueError(f"download of {asset.name} exceeds expected size")
                    staged.write(chunk)
                    digest.update(chunk)
            if size != asset.size or digest.hexdigest() != asset.sha256:
                raise ValueError(f"download of {asset.name} failed integrity verification")
            staged.close()
            temporary.chmod(0o644)
            temporary.replace(destination)
        except httpx.RequestError as exc:
            raise ValueError(f"download of {asset.name} failed during transfer") from exc
        finally:
            temporary.unlink(missing_ok=True)


def download_generation(directory: Path, *, cuda: bool, token: str = "") -> Path:
    assets = CUDA_ASSETS if cuda else CPU_ASSETS
    _download_assets(directory, assets, token)
    return directory


def _download_assets(directory: Path, assets: tuple[Asset, ...], token: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".imagies-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            for asset in assets:
                download_asset(client, directory, asset, token)


class GenerationDownloads:
    def __init__(self, directory: Path, registry: Path, prefix: str, token: str) -> None:
        config = json.loads(registry.read_text())
        self._files = {
            name: tuple(
                arg.removeprefix(_ASSET_ROOT) for arg in item["args"] if arg.startswith(_ASSET_ROOT)
            )
            for name, item in config["models"].items()
        }
        self._default = config["default_model"]
        self._directory = directory
        self._prefix = prefix
        self._token = token
        self._assets = {asset.name: asset for asset in CUDA_ASSETS}

    def _missing_files(self, model: str) -> list[str]:
        missing = []
        for name in self._files[model]:
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Invalid model asset path for {model}")
            path = self._directory / name
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(name)
        return missing

    def ensure(self, model: str | None) -> None:
        if model is None:
            key = self._default
        else:
            key = model.strip().removeprefix("openai/")
            stripped = key.removeprefix(self._prefix)
            if stripped in self._files:
                key = stripped
        if key not in self._files:
            return
        missing = self._missing_files(key)
        if missing:
            unmanaged = [name for name in missing if name not in self._assets]
            if unmanaged:
                raise ValueError(f"No download source for {', '.join(unmanaged)}")
            _download_assets(
                self._directory, tuple(self._assets[name] for name in missing), self._token
            )

    def prefetch(self) -> None:
        for model in self._files:
            missing = self._missing_files(model)
            if any(name not in self._assets for name in missing):
                _log.warning("skipping unregistered startup download", extra={"model": model})
                continue
            if not self._token and any(self._assets[name].gated for name in missing):
                _log.warning(
                    "skipping gated startup download without HF_TOKEN", extra={"model": model}
                )
                continue
            self.ensure(model)
