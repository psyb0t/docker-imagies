# Changelog

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
