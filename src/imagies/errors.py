from __future__ import annotations


class InvalidImage(ValueError):
    pass


class ImageMissing(LookupError):
    pass


class ModelUnavailable(LookupError):
    pass


class ModelProcessError(RuntimeError):
    pass
