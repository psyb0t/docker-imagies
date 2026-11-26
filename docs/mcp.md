---
title: MCP
description: Connect a streamable HTTP MCP client to Imagies and use staged images with its tools.
---

# MCP

Connect a streamable HTTP MCP client to `http://localhost:8000/v1/mcp`. Add the bearer token to the client headers when `IMAGIES_AUTH_TOKEN` is set. If you connect by a reverse proxy or container service name, add the Host header the server will receive to `IMAGIES_MCP_ALLOWED_HOSTS`.

| Tool | Arguments | Result |
| --- | --- | --- |
| `transform` | `spec` with a staged image ID and edit options | Staged output image |
| `montage` | `spec` with image IDs and grid options | Staged contact sheet |
| `match` | `spec` with image IDs and text queries | Ranked matches by image |
| `detect` | `spec` with image IDs and an optional confidence threshold | Object labels, scores, and normalized boxes |
| `generate` | `model`, `prompt`, and optional `size` | Staged output image, available on generation builds |

Upload inputs and download staged results using the [REST API](api.md). The `spec` argument takes the same JSON object as the matching REST endpoint.

For example, call `transform` with these arguments. Replace the example image ID with the ID returned by the upload.

```json
{
  "spec": {
    "image_id": "00000000000000000000000000000000",
    "width": 640,
    "height": 480,
    "fit": "contain",
    "format": "webp"
  }
}
```
