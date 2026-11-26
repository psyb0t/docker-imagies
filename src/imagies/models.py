from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from imagies.constants import GENERATION_MODELS_DIR, MODELS_DIR
from imagies.generation_models import download_generation

_CLIP_REVISION = "bf8895cff1da285aa497db31204d14464adcaa7c"
_CLIP_BASE = f"https://huggingface.co/openai/clip-vit-base-patch32/resolve/{_CLIP_REVISION}"
_CLIP_FILES = (
    "config.json",
    "merges.txt",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
_CLIP_SHA256 = "99d28a652e6ec46629ab7047a0ac82c69b1fe11e0ce672c43af65d3a9a3fc05d"
_CHUNK_BYTES = 1024 * 1024
_MAX_WEIGHT = 620 * _CHUNK_BYTES
_MAX_CONFIG = 4 * _CHUNK_BYTES
_DOWNLOAD_ENV = "IMAGIES_MODEL_DOWNLOAD"
_DOWNLOAD_MODES = {"startup", "on-demand"}


def model_download_mode() -> str:
    mode = os.environ.get(_DOWNLOAD_ENV, "").strip()
    if mode and mode not in _DOWNLOAD_MODES:
        raise ValueError(f"{_DOWNLOAD_ENV} must be startup or on-demand")
    return mode


def has_clip_files(models: Path) -> bool:
    return all(_nonempty_file(models / "clip" / name) for name in _CLIP_FILES)


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def download_clip(models: Path) -> Path:
    destination = models / "clip"
    if has_clip_files(models):
        return destination
    models.mkdir(parents=True, exist_ok=True)
    with (models / ".imagies-clip-download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not has_clip_files(models):
            _download_missing_clip_files(models, destination)
    return destination


def _download_missing_clip_files(models: Path, destination: Path) -> None:
    destination.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models, prefix=".clip-") as temporary:
        for name in _CLIP_FILES:
            target = destination / name
            if _nonempty_file(target):
                continue
            staged = Path(temporary) / name
            weight = name == "model.safetensors"
            digest = hashlib.sha256()
            downloaded = 0
            with (
                urlopen(f"{_CLIP_BASE}/{name}", timeout=60) as response,
                staged.open("wb") as file,
            ):
                for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                    downloaded += len(chunk)
                    if downloaded > (_MAX_WEIGHT if weight else _MAX_CONFIG):
                        raise ValueError(f"{name} exceeds its download limit")
                    file.write(chunk)
                    digest.update(chunk)
            if not downloaded or (weight and digest.hexdigest() != _CLIP_SHA256):
                raise ValueError(f"failed integrity check for {name}")
            staged.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model as regular local files.")
    parser.add_argument("model", choices=["clip", "generation-cpu", "generation-cuda"])
    options = parser.parse_args()
    if options.model == "clip":
        print(download_clip(MODELS_DIR))
        return
    print(
        download_generation(
            GENERATION_MODELS_DIR,
            cuda=options.model == "generation-cuda",
            token=os.environ.get("HF_TOKEN", ""),
        )
    )


if __name__ == "__main__":
    main()
