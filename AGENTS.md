# Imagies

Imagies is a FastAPI service for still-image editing and analysis, with streamable HTTP MCP. Use `make check` for the checks and `make build` for the runtime image.

- `src/imagies/images.py` owns still-image operations and file staging.
- `src/imagies/vision.py` owns CLIP and YOLO inference. Keep model loading local and optional.
- `src/imagies/models.py` installs explicitly requested model files.
- Keep REST and MCP on the same operation functions in `src/imagies/server.py`.
- FastAPI generates OpenAPI from the routes and models.
- `generation/worker` supervises the bundled stable-diffusion.cpp binary. FastAPI owns the public generation routes and starts the worker on loopback.
