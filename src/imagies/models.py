from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

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


def download_clip(models: Path) -> Path:
    destination = models / "clip"
    if destination.exists():
        raise FileExistsError("CLIP directory already exists")
    models.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=models, prefix=".clip-") as temporary:
        source = Path(temporary) / "clip"
        source.mkdir()
        for name in _CLIP_FILES:
            weight = name == "model.safetensors"
            digest = hashlib.sha256()
            downloaded = 0
            with (
                urlopen(f"{_CLIP_BASE}/{name}", timeout=60) as response,
                (source / name).open("wb") as file,
            ):
                for chunk in iter(lambda: response.read(_CHUNK_BYTES), b""):
                    downloaded += len(chunk)
                    if downloaded > (_MAX_WEIGHT if weight else _MAX_CONFIG):
                        raise ValueError(f"{name} exceeds its download limit")
                    file.write(chunk)
                    digest.update(chunk)
            if not downloaded or (weight and digest.hexdigest() != _CLIP_SHA256):
                raise ValueError(f"failed integrity check for {name}")
        source.rename(destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a model as regular local files.")
    parser.add_argument("model", choices=["clip", "generation-cpu", "generation-cuda"])
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("IMAGIES_DATA_DIR", "/data"))
    )
    options = parser.parse_args()
    if options.model == "clip":
        print(download_clip(options.data_dir / "models"))
        return
    configured = os.environ.get("IMAGIES_GENERATION_MODELS_DIR")
    directory = Path(configured) if configured else options.data_dir / "models" / "generation"
    print(
        download_generation(
            directory, cuda=options.model == "generation-cuda", token=os.environ.get("HF_TOKEN", "")
        )
    )


if __name__ == "__main__":
    main()
