---
title: Deployment
description: Choose an Imagies Docker image, install local model weights, and configure authentication and persistent storage.
---

# Deployment

Both images target Linux x86-64 and include editing, vision and image generation. An NVIDIA build needs a compatible GPU and the NVIDIA Container Toolkit.

| Image tag | Build command | What it runs |
| --- | --- | --- |
| `latest` | `make build` | CPU CLIP, YOLO and Stable Diffusion |
| `latest-cuda` | `make build-cuda` | CUDA CLIP and Stable Diffusion, CPU YOLO |

```sh
docker run --rm --init \
  -p 127.0.0.1:8000:8000 \
  -v imagies:/data \
  psyb0t/imagies:latest
```

Pass `--gpus all` to a CUDA image. When upgrading from `latest-generation` or `latest-generation-cuda`, switch to `latest` or `latest-cuda` respectively and keep your existing storage.

## Storage mounts

Mount a named volume or a host directory at `/data`. Imagies saves staged images under `/data/images` and looks for model files under `/data/models`. Image staging and downloads need write access for container UID and GID `10001`.

To reuse an existing directory of generation weights, add this mount alongside the `/data` volume. Change the source to your host directory.

```sh
--mount "type=bind,source=$PWD/weights,target=/data/models/generation,readonly"
```

Use the same mount without `readonly` when running the model downloader.

## Model files

```sh
docker run --rm -v imagies:/data psyb0t/imagies:latest \
  python -m imagies.models clip

docker run --rm -v imagies:/data psyb0t/imagies:latest \
  python -m imagies.models generation-cpu
```

Use `generation-cuda` with the CUDA generation image to install its larger catalog. Gated models require accepting their terms and passing `HF_TOKEN` to the downloader. Installed generation weights work offline. To bring your own YOLOv8 detection export, put the ONNX file at `/data/models/yolo/model.onnx` and put one class per line in `labels.txt`. The float32 input is NCHW and the output is the raw pre-NMS detection tensor.

## Configuration

| Environment variable | Behavior |
| --- | --- |
| `IMAGIES_AUTH_TOKEN` | Optional shared bearer token. Set it before serving clients beyond loopback. |
| `IMAGIES_MCP_ALLOWED_HOSTS` | Comma-separated allowed MCP Host values. Include the port when the client sends one. |
| `IMAGIES_GENERATION` | Defaults to `1`. Set it to `0` to skip the worker and generation endpoints. |
| `IDLE_TIMEOUT` | Generation model idle timeout, default `5m`. |
| `LOAD_TIMEOUT` | Generation model load timeout, default `10m`. |
| `HF_TOKEN` | Optional downloader token for gated weights. |

Keep the data volume attached when you replace the container. All authenticated callers share the staged images. The service exposes health on `/healthz` without a token.
