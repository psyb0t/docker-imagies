# Changelog

## [0.2.0]

- Ship editing, CLIP, YOLO and generation in both `latest` (CPU) and `latest-cuda` (NVIDIA). Switch from `latest-generation` or `latest-generation-cuda` to those tags and keep your mounted data.
- Describe each MCP tool and document the model, prompt and size arguments for image generation.
- Imagies uses fixed paths for data and generation weights. Mount existing storage at `/data` or mount just the generation weights at `/data/models/generation`. Replace the old path environment variables and the downloader's `--data-dir` option with mounts.

## [0.1.0]

- Added image staging, crop, resize, format conversion and grids.
- Added CLIP matching and YOLO object detection with local model files.
- Exposed editing and analysis over REST and streamable HTTP MCP.
- Added CPU and NVIDIA stable-diffusion.cpp image generation with model load, unload and cancel controls.
- Added OpenAI-compatible REST, MCP generation and a downloader for reusable model files.
- Switched vision builds to Torchbase and enabled CUDA CLIP matching in the NVIDIA images.
- Added CI for Python and the Go worker, default-branch Docker builds, and version tags that promote the images built for the same commit.
- Added a searchable documentation site with deployment guides, MCP examples, and a REST reference generated from FastAPI.
- Added engine-scoped routes for discovery, status, model selection, unload, and cancellation. Older Aigate versions can still use the legacy stable-diffusion.cpp routes.
