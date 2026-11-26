---
title: Deployment
description: Choose an Imagies Docker image, install local model weights, and configure authentication and persistent storage.
---

# Deployment

The four builds target Linux x86-64. An NVIDIA build needs a compatible GPU and the NVIDIA Container Toolkit.

| Image tag | Build command | What it runs |
| --- | --- | --- |
| `latest` | `make build` | Editing, CPU CLIP and CPU YOLO |
| `latest-cuda` | `make build-cuda` | Editing, CUDA CLIP and CPU YOLO |
| `latest-generation` | `make build-generation` | Standard vision and CPU Stable Diffusion |
| `latest-generation-cuda` | `make build-generation-cuda` | NVIDIA vision and CUDA Stable Diffusion |

```sh
docker run --rm --init \
  -p 127.0.0.1:8000:8000 \
  -v imagies:/data \
  psyb0t/imagies:latest
```

Pass `--gpus all` to a CUDA image. For a smaller local build, `make build EDITING_ONLY=1` leaves out CLIP and YOLO. The same flag on a generation build keeps the generation worker.

## Model files

```sh
docker run --rm -v imagies:/data psyb0t/imagies:latest \
  python -m imagies.models clip

docker run --rm -v imagies:/data psyb0t/imagies:latest-generation \
  python -m imagies.models generation-cpu
```

Use `generation-cuda` with the CUDA generation image to install its larger catalog. Gated models require accepting their terms and passing `HF_TOKEN` to the downloader. Installed generation weights work offline. To bring your own YOLOv8 detection export, put the ONNX file at `/data/models/yolo/model.onnx` and put one class per line in `labels.txt`. The float32 input is NCHW and the output is the raw pre-NMS detection tensor.

## Configuration

| Environment variable | Behavior |
| --- | --- |
| `IMAGIES_AUTH_TOKEN` | Optional shared bearer token. Set it before serving clients beyond loopback. |
| `IMAGIES_DATA_DIR` | Defaults to `/data`. Holds staged files in `images/` and model files in `models/`. |
| `IMAGIES_MCP_ALLOWED_HOSTS` | Comma-separated allowed MCP Host values. Include the port when the client sends one. |
| `IMAGIES_GENERATION_MODELS_DIR` | Overrides the generation weights directory, normally `/data/models/generation`. |
| `IMAGIES_GENERATION` | Set to `1` by generation images. Enables the generation worker and endpoints. |
| `IDLE_TIMEOUT` | Generation model idle timeout, default `5m`. |
| `LOAD_TIMEOUT` | Generation model load timeout, default `10m`. |
| `HF_TOKEN` | Optional downloader token for gated weights. |

The data volume is writable by container UID and GID `10001`. Keep it attached when you replace the container and back it up if you need the staged files. All authenticated callers share the same staging directory. The service exposes health on `/healthz` without a token.
