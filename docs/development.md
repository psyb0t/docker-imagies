---
title: Development
description: Run Imagies checks, refresh the generated OpenAPI file, and build the docs locally.
---

# Development

Run `make check` for Python lint and tests and Go worker vet and race tests. Run `make docs-openapi` after changing a protocol route or schema and `make docs-build` to build the site into `website/build`. The OpenAPI file is exported from FastAPI with the generation endpoints enabled.

The default-branch pipeline builds and scans each Docker variant after code checks pass. A version tag waits for the matching default-branch run, promotes the images it built and then creates the GitHub release.
