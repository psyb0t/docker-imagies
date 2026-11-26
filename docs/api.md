---
title: REST API
description: Stage a still image, transform it, and download the result using the Imagies HTTP API.
---

# REST API

Use the reference in the sidebar for parameters, request bodies, and responses. The [OpenAPI JSON](https://psyb0t.github.io/docker-imagies/reference/openapi.json) includes the optional generation endpoints. A running instance advertises its own enabled routes at `/openapi.json`.

## Stage an image

Send up to 20 MiB of PNG, JPEG, or WebP bytes as the request body. If the server requires authentication, add the same `Authorization: Bearer <token>` header to each request.

```sh
curl --fail-with-body http://localhost:8000/v1/images \
  -H 'Content-Type: image/png' \
  --data-binary @example.png
```

Copy the returned `id`. You can download the staged image from `GET /v1/images/{image_id}` and delete it with `DELETE /v1/images/{image_id}`. Edit or group staged images at `/v1/images/transform` and `/v1/images/montage`. Check `GET /v1/engines` before using CLIP matching or YOLO detection. A missing image or model returns HTTP 404.

## Resize a staged image

Replace the example ID with the uploaded image ID. The response gives a new ID for the resized image.

```sh
curl --fail-with-body http://localhost:8000/v1/images/transform \
  -H 'Content-Type: application/json' \
  --data '{"image_id":"00000000000000000000000000000000","width":640,"height":480,"fit":"contain","format":"webp"}'
```

## Generate an image

On a generation image, use `GET /v1/models` to find the exact model ID. The OpenAI-compatible generation route returns the generated image as base64 when you request `b64_json`.

```sh
curl --fail-with-body http://localhost:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  --data '{"model":"imagies-sd-turbo","prompt":"a yellow teapot on a wooden table","size":"256x256","response_format":"b64_json"}'
```
