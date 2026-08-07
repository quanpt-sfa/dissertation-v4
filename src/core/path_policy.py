"""Cross-platform policy for paths rooted beneath an explicit runtime directory."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath


class RelativePathError(ValueError):
    """Raised when a configured child path is not portable or escapes its root."""


def portable_relative_path(value: str | os.PathLike[str], *, context: str) -> Path:
    """Return a normalized relative path accepted on Windows and POSIX.

    Configured paths are serialized with forward slashes, but backslashes are
    accepted as input and normalized. Absolute paths, drive-qualified paths,
    UNC paths, traversal components and empty paths are rejected before any
    filesystem lookup occurs.
    """

    text = os.fspath(value).strip()
    if not text or "\x00" in text:
        raise RelativePathError(f"{context}: non-empty relative path required")

    windows = PureWindowsPath(text)
    posix = PurePosixPath(text)
    if windows.is_absolute() or windows.drive or windows.root or posix.is_absolute():
        raise RelativePathError(
            f"{context}: absolute or drive-qualified path is forbidden: {text!r}"
        )

    normalized = text.replace("\\", "/")
    raw_parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise RelativePathError(
            f"{context}: normalized child path cannot contain empty, dot, or parent components"
        )

    return Path(*raw_parts)


def resolve_relative_path(
    root: Path,
    value: str | os.PathLike[str],
    *,
    context: str,
) -> Path:
    """Join a portable relative path beneath ``root`` without resolving a missing leaf.

    The lexical path is retained for stable artifact naming. Existing parents
    and an existing leaf are resolved individually so a symlink or Windows
    junction cannot redirect the path outside the declared root.
    """

    root_path = root.expanduser().resolve(strict=False)
    relative = portable_relative_path(value, context=context)
    candidate = root_path.joinpath(relative)

    try:
        candidate.relative_to(root_path)
    except ValueError as exc:  # defensive; traversal is rejected above
        raise RelativePathError(
            f"{context}: child path escapes root: root={root_path}, child={candidate}"
        ) from exc

    current = root_path
    for part in relative.parts:
        current = current / part
        if current.exists() or current.is_symlink():
            resolved = current.resolve(strict=True)
            try:
                resolved.relative_to(root_path)
            except ValueError as exc:
                raise RelativePathError(
                    f"{context}: existing path component escapes root: "
                    f"root={root_path}, component={current}, resolved={resolved}"
                ) from exc

    return candidate
