from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from imagies.errors import ModelUnavailable

_SIDE = 640
_NMS_IOU = 0.45
_MAX_CANDIDATES = 1000
_MAX_DETECTIONS = 100


class Vision:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = Lock()
        self.models: dict[str, Any] = {}

    def available(self) -> list[str]:
        clip = self.root / "clip"
        yolo = self.root / "yolo"
        engines = []
        if (
            find_spec("torch")
            and find_spec("transformers")
            and (clip / "model.safetensors").is_file()
            and (clip / "config.json").is_file()
        ):
            engines.append("clip")
        if (
            find_spec("onnxruntime")
            and (yolo / "model.onnx").is_file()
            and (yolo / "labels.txt").is_file()
        ):
            engines.append("yolo")
        return engines

    def _load(self, engine: str) -> Any:
        if engine not in self.available():
            raise ModelUnavailable(f"{engine} dependencies or model files are missing")
        if engine == "yolo":
            import onnxruntime

            root = self.root / "yolo"
            labels = (root / "labels.txt").read_text(encoding="utf-8").splitlines()
            if not labels or any(not label.strip() for label in labels):
                raise ValueError("YOLO labels.txt must contain one non-empty label per class")
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            session = onnxruntime.InferenceSession(
                str(root / "model.onnx"),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            inputs = session.get_inputs()
            if len(inputs) != 1 or len(inputs[0].shape) != 4 or inputs[0].type != "tensor(float)":
                raise ValueError("expected one YOLO float32 input in NCHW layout")
            batch, channels, height, width = inputs[0].shape
            if (isinstance(batch, int) and batch != 1) or (
                isinstance(channels, int) and channels != 3
            ):
                raise ValueError("YOLO input must accept one RGB image")
            size = tuple(value if isinstance(value, int) else _SIDE for value in (width, height))
            if any(not 32 <= value <= 1280 or value % 32 for value in size):
                raise ValueError("YOLO input dimensions must be multiples of 32 from 32 to 1280")
            return session, labels, inputs[0].name, size
        import torch
        from transformers import CLIPModel, CLIPProcessor

        root = self.root / "clip"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        processor = CLIPProcessor.from_pretrained(root, local_files_only=True)
        model = (
            CLIPModel.from_pretrained(root, local_files_only=True, use_safetensors=True)
            .eval()
            .to(device)
        )
        return model, processor, device

    def match(self, paths: list[Path], queries: list[str]) -> list[dict[str, Any]]:
        with self.lock:
            if "clip" not in self.models:
                self.models["clip"] = self._load("clip")
            return _clip_matches(self.models["clip"], paths, queries)

    def detect(self, paths: list[Path], confidence: float) -> list[dict[str, Any]]:
        with self.lock:
            if "yolo" not in self.models:
                self.models["yolo"] = self._load("yolo")
            network, labels, input_name, size = self.models["yolo"]
            return [
                {
                    "detections": _detect(
                        network, labels, path, confidence, input_name=input_name, size=size
                    )
                }
                for path in paths
            ]


def _detect(
    network: Any,
    labels: list[str],
    path: Path,
    threshold: float,
    *,
    input_name: str,
    size: tuple[int, int],
) -> list[dict[str, Any]]:
    import numpy as np

    with Image.open(path) as image:
        width, height = image.size
        scale = min(size[0] / width, size[1] / height)
        resized_w, resized_h = max(1, round(width * scale)), max(1, round(height * scale))
        offset = ((size[0] - resized_w) // 2, (size[1] - resized_h) // 2)
        canvas = Image.new("RGB", size, (114, 114, 114))
        canvas.paste(
            image.convert("RGB").resize((resized_w, resized_h), Image.Resampling.BILINEAR), offset
        )
    tensor = np.ascontiguousarray(
        np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255
    )
    (output,) = network.run(None, {input_name: tensor})
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != len(labels) + 4:
        raise ValueError("expected YOLOv8 ONNX detection output [1, 4 + classes, candidates]")
    rows = output[0].T
    classes = rows[:, 4:].argmax(axis=1)
    scores = rows[:, 4:].max(axis=1)
    eligible = np.flatnonzero((scores >= threshold) & np.isfinite(rows).all(axis=1))
    indices = eligible[np.argsort(-scores[eligible], kind="stable")][:_MAX_CANDIDATES]
    centers = rows[:, :2]
    halves = rows[:, 2:4] / 2
    top_left, bottom_right = centers - halves, centers + halves
    areas = np.prod(np.maximum(0, bottom_right - top_left), axis=1)
    detections = []
    while indices.size and len(detections) < _MAX_DETECTIONS:
        index = indices[0]
        remaining = indices[1:]
        intersection = np.prod(
            np.maximum(
                0,
                np.minimum(bottom_right[index], bottom_right[remaining])
                - np.maximum(top_left[index], top_left[remaining]),
            ),
            axis=1,
        )
        union = areas[index] + areas[remaining] - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        indices = remaining[(classes[remaining] != classes[index]) | (iou <= _NMS_IOU)]
        resized = (resized_w, resized_h)
        left, top = np.clip((top_left[index] - offset) / resized, 0, 1)
        right, bottom = np.clip((bottom_right[index] - offset) / resized, 0, 1)
        if right > left and bottom > top:
            detections.append(
                {
                    "label": labels[classes[index]],
                    "confidence": float(scores[index]),
                    "box": list(map(float, (left, top, right, bottom))),
                }
            )
    return detections


def _clip_matches(model_parts: Any, paths: list[Path], queries: list[str]) -> list[dict[str, Any]]:
    import torch

    model, processor, device = model_parts
    result = []
    with torch.inference_mode():
        encoded = processor(text=queries, padding=True, truncation=True, return_tensors="pt").to(
            device
        )
        text = torch.nn.functional.normalize(model.get_text_features(**encoded), dim=-1)
        for offset in range(0, len(paths), 8):
            batch = paths[offset : offset + 8]
            images = []
            for path in batch:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            encoded = processor(images=images, return_tensors="pt").to(device)
            vectors = torch.nn.functional.normalize(model.get_image_features(**encoded), dim=-1)
            for scores in (vectors @ text.T).tolist():
                matches = [
                    {"query": query, "score": score}
                    for query, score in zip(queries, scores, strict=True)
                ]
                result.append(
                    {"matches": sorted(matches, key=lambda match: match["score"], reverse=True)}
                )
    return result
