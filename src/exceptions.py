"""Exceptions raised in `src` package context."""


class srcError(Exception):
    """Base Exception."""


class MissingConfigError(srcError):
    """Raised when a model does not have a config field."""


class ModelNotFoundError(srcError):
    """Raised when a model is not registered in the models module."""
