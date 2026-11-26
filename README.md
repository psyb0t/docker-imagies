# Imagies

Self-hosted still-image tools. Crop, resize, convert and tile images, match them against text with CLIP, or find objects with YOLO. REST and MCP run on the same port.

[Documentation](https://psyb0t.github.io/docker-imagies/) covers deployment, model files, the REST API, and MCP.

## Quick start

```sh
make build
docker run --rm --init --cap-drop=ALL \
  --security-opt=no-new-privileges:true --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --memory=4g --cpus=2 --pids-limit=256 \
  -p 127.0.0.1:8000:8000 -v imagies:/data psyb0t/imagies
```

REST docs are at `http://localhost:8000/docs`, OpenAPI is at `/openapi.json`, and streamable HTTP MCP is at `/v1/mcp`. The loopback example uses no token. Set `IMAGIES_AUTH_TOKEN` before serving other clients. Use the same bearer token for REST and MCP. For MCP through a reverse proxy or another container, add the exact request host, including the port if present, to `IMAGIES_MCP_ALLOWED_HOSTS`. Separate multiple hosts with commas.

Stage a still image with `POST /v1/images` and a raw PNG, JPEG or WebP body. The response gives its ID. Pass it to an operation and download results from `GET /v1/images/{image_id}`. Delete files you no longer need with `DELETE /v1/images/{image_id}`. All authenticated callers share the staging directory.

| REST endpoint | MCP tool | What it does |
| --- | --- | --- |
| `POST /v1/images/transform` | `transform` | Crop, fit an image to a canvas, and convert formats. |
| `POST /v1/images/montage` | `montage` | Tile a list of images into a PNG grid. |
| `POST /v1/images/match` | `match` | Rank text queries for each image by cosine similarity. |
| `POST /v1/images/detect` | `detect` | Find objects and return labels, confidence and normalized `[x1, y1, x2, y2]` boxes. |

MCP tools take the matching REST JSON body inside a `spec` argument. Stage and download files via REST. For video, extract timestamped frames with [Flickies](https://github.com/psyb0t/docker-flickies) and send the stills to Imagies.

## Model files

Imagies loads models on demand. Processing works offline once the models are installed. Models use regular files under `/data/models` that you can copy to another installation. List available models with `GET /v1/engines`.

To install the pinned CLIP files into your data volume, run:

```sh
docker run --rm -v imagies:/data psyb0t/imagies python -m imagies.models clip
```

The installer checks the safetensors checksum before putting the files in place. To install a compatible Hugging Face Transformers CLIP model manually, supply its configuration, tokenizer, image processor and safetensors files in `/data/models/clip`.

For YOLO, put a YOLOv8 detection ONNX export at `/data/models/yolo/model.onnx`. Use a float32 input of `[1, 3, height, width]` and raw output of `[1, 4 + class_count, candidates]` before non-maximum suppression. Input dimensions must be multiples of 32 from 32 to 1280. Dynamic dimensions default to 640. Add `labels.txt` with one class label per line in model order. Check the license on your weights.

Use `make build EDITING_ONLY=1` for a smaller image with the still-image editing tools. Run `make check` for lint and tests.

The standard and NVIDIA vision images get PyTorch from Torchbase. For CLIP on NVIDIA, run `make build-cuda`, then use the Quick start command with `--gpus all` and `psyb0t/imagies:latest-cuda`. CLIP uses CUDA when Docker exposes a compatible GPU. YOLO uses the ONNX Runtime CPU provider.

## Image generation

Imagies has CPU and NVIDIA builds with stable-diffusion.cpp. The CPU image supports SD Turbo and SDXL Turbo. The CUDA image also supports SDXL Lightning, FLUX.1 Schnell and Juggernaut XI. Imagies loads one generation model at a time. `IDLE_TIMEOUT` defaults to `5m` and `LOAD_TIMEOUT` to `10m`.

```sh
make build-generation
docker run --rm -v imagies:/data psyb0t/imagies:latest-generation \
  python -m imagies.models generation-cpu
docker run --rm --init --cap-drop=ALL --security-opt=no-new-privileges:true \
  --read-only --tmpfs /tmp:rw,noexec,nosuid,size=256m --memory=12g --cpus=4 \
  --pids-limit=512 -p 127.0.0.1:8000:8000 -v imagies:/data \
  psyb0t/imagies:latest-generation
```

For NVIDIA, use `make build-generation-cuda`, the `:latest-generation-cuda` image, and `generation-cuda` to install models. Run the service with `--gpus all`. This image runs both CLIP matching and native image generation on CUDA. Override `CUDA_ARCHITECTURES` at build time to target a specific compute capability, such as `make build-generation-cuda CUDA_ARCHITECTURES=86` for an RTX A5000. Set `EDITING_ONLY=1` on either generation build to include generation and editing without CLIP or YOLO. Provide `HF_TOKEN` to the downloader for missing gated FLUX and Juggernaut files after accepting their terms. Installed weights work offline.

The downloader writes regular GGUF and safetensors files to `/data/models/generation` and checks the hashes of new downloads. Set `IMAGIES_GENERATION_MODELS_DIR` to use existing weights. CPU and CUDA can share that directory. Only the downloader needs write access to it.

`GET /v1/models` lists the model IDs. Pass one to `POST /v1/images/generations` with an OpenAI-compatible request. Model controls live at `/sdcpp/v1/status`, `/sdcpp/v1/load?model=sd-turbo`, `/sdcpp/v1/unload` and `/sdcpp/v1/cancel`. The MCP tool `generate` stages one result as an Imagies image ID.
