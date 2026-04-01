"""Path helpers for portable project file handling."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value: str | Path, base_dir: str | Path | None = None) -> Path:
    """Resolve an absolute or project-relative path value to an absolute path."""
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()

    root = PROJECT_ROOT if base_dir is None else Path(base_dir).expanduser().resolve()
    return (root / path).resolve()


def to_portable_path(path_value: str | Path, base_dir: str | Path | None = None) -> str:
    """Return a project-relative path when possible, otherwise an absolute path."""
    resolved_path = resolve_path(path_value, base_dir=base_dir)
    try:
        return str(resolved_path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved_path)
