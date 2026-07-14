"""P0 protocol compilation services; later stages consume locked registries only."""

from .registry_compiler import compile_registry

__all__ = ["compile_registry"]
