"""Project paths and root discovery."""

from __future__ import annotations

from pathlib import Path


def project_root(start: Path | None = None) -> Path:
    """Return the repository root (directory that contains configs/)."""
    here = (start or Path(__file__)).resolve()
    candidates = [here, *here.parents]
    for cand in candidates:
        if (cand / "configs" / "base.yaml").exists():
            return cand
    # Fallback: src/ is one level below the project root.
    return Path(__file__).resolve().parents[1]


def resolve_under_root(path: str | Path, root: Path | None = None) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (root or project_root()) / p
