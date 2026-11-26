from __future__ import annotations

import re
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal, Self
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, model_validator

from imagies.errors import ImageMissing, InvalidImage

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
_MAX_PIXELS = 16_000_000
_IMAGE_ID = re.compile(r"[0-9a-f]{32}")
_FORMATS = ("PNG", "JPEG", "WEBP")
_SAVE_FORMAT = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
ImageId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
Dimension = Annotated[int, Field(strict=True, ge=1, le=4096)]
Format = Literal["png", "jpeg", "webp"]


class Asset(BaseModel):
    id: ImageId
    width: int
    height: int
    format: Format


class Transform(BaseModel):
    image_id: ImageId
    width: Dimension | None = None
    height: Dimension | None = None
    fit: Literal["contain", "cover", "pad", "stretch"] = "contain"
    crop: tuple[int, int, int, int] | None = Field(
        default=None, description="Crop rectangle as x, y, width, height before resize."
    )
    format: Format = "png"

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        if (self.width is None) != (self.height is None):
            raise ValueError("width and height must be provided together")
        if self.crop and (min(self.crop) < 0 or min(self.crop[2:]) == 0):
            raise ValueError("crop requires nonnegative coordinates and positive dimensions")
        return self


class Montage(BaseModel):
    image_ids: list[ImageId] = Field(min_length=2, max_length=32)
    columns: int = Field(default=3, strict=True, ge=1, le=8)
    cell_width: Dimension = 320
    cell_height: Dimension = 180


class Match(BaseModel):
    image_ids: list[ImageId] = Field(min_length=1, max_length=32)
    queries: list[Annotated[str, Field(min_length=1, max_length=160)]] = Field(
        min_length=1, max_length=16
    )


class Detect(BaseModel):
    image_ids: list[ImageId] = Field(min_length=1, max_length=32)
    confidence: float = Field(default=0.3, strict=True, gt=0, lt=1)


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, image_id: str) -> Path:
        if not _IMAGE_ID.fullmatch(image_id):
            raise ImageMissing("image not found")
        path = self.root / image_id
        if not path.is_file():
            raise ImageMissing("image not found")
        return path

    def save(self, image: Image.Image, format: Format) -> Asset:
        if image.width * image.height > _MAX_PIXELS:
            raise InvalidImage("image exceeds the pixel limit")
        self.root.mkdir(parents=True, exist_ok=True)
        image_id = uuid4().hex
        with NamedTemporaryFile(dir=self.root, prefix=".write-", delete=False) as output:
            temporary = Path(output.name)
            try:
                clean = image.convert("RGB" if format == "jpeg" else "RGBA")
                clean.info.clear()
                clean.save(output, format=_SAVE_FORMAT[format])
                output.close()
                temporary.replace(self.root / image_id)
            finally:
                temporary.unlink(missing_ok=True)
        return Asset(id=image_id, width=image.width, height=image.height, format=format)

    def ingest(self, content: bytes) -> Asset:
        try:
            with Image.open(BytesIO(content), formats=_FORMATS) as source:
                if source.width * source.height > _MAX_PIXELS:
                    raise InvalidImage("image exceeds the pixel limit")
                if getattr(source, "is_animated", False):
                    raise InvalidImage("only still images are accepted")
                return self.save(ImageOps.exif_transpose(source), "png")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise InvalidImage("expected a valid PNG, JPEG or WebP still image") from exc

    def transform(self, spec: Transform) -> Asset:
        with Image.open(self.path(spec.image_id)) as source:
            image = source.convert("RGBA")
        if spec.crop:
            x, y, width, height = spec.crop
            if x + width > image.width or y + height > image.height:
                raise InvalidImage("crop extends outside the image")
            image = image.crop((x, y, x + width, y + height))
        if spec.width is not None and spec.height is not None:
            size = (spec.width, spec.height)
            if spec.width * spec.height > _MAX_PIXELS:
                raise InvalidImage("canvas exceeds the pixel limit")
            if spec.fit == "cover":
                image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
            elif spec.fit == "pad":
                image = ImageOps.pad(image, size, Image.Resampling.LANCZOS)
            elif spec.fit == "stretch":
                image = image.resize(size, Image.Resampling.LANCZOS)
            else:
                image.thumbnail(size, Image.Resampling.LANCZOS)
        return self.save(image, spec.format)

    def montage(self, spec: Montage) -> Asset:
        paths = self.paths(spec.image_ids)
        columns = min(spec.columns, len(paths))
        rows = (len(paths) + columns - 1) // columns
        width, height = columns * spec.cell_width, rows * spec.cell_height
        if width * height > _MAX_PIXELS:
            raise InvalidImage("canvas exceeds the pixel limit")
        canvas = Image.new("RGBA", (width, height))
        for index, path in enumerate(paths):
            with Image.open(path) as source:
                tile = ImageOps.pad(
                    source.convert("RGBA"),
                    (spec.cell_width, spec.cell_height),
                    Image.Resampling.LANCZOS,
                )
                y, x = divmod(index, columns)
                canvas.paste(tile, (x * spec.cell_width, y * spec.cell_height))
        return self.save(canvas, "png")

    def paths(self, image_ids: Sequence[str]) -> list[Path]:
        return [self.path(image_id) for image_id in image_ids]

    def format(self, image_id: str) -> str:
        with Image.open(self.path(image_id)) as image:
            return {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp"}[str(image.format)]
