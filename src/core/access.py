"""Public access-control API for later CLI modules."""

from .access_matrix import assert_access, compile_access_matrix

__all__ = ["assert_access", "compile_access_matrix"]
