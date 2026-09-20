"""Recursive discovery of HELECAR-D candidate CSV files."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from data.schema import (
    ALIAS_LOOKUP,
    EXPECTED_ANALYSED_CANONICAL,
    RAW_INDICATOR_NORMALIZED,
    normalize_header,
)
from paths import project_root

TRAJECTORY_RE = re.compile(r"(?:^|[/\\_-])(T[123])(?:[/\\_-]|$)", re.IGNORECASE)
TRIP_STEM_RE = re.compile(r"(T[123]_[0-9]{1,2}_[0-9]{1,2}_[0-9]{4}(?:_[0-9]+)?)")


@dataclass
class CandidateFile:
    path: Path
    kind: str  # analysed | raw | unknown
    score: int
    columns: list[str]
    canonical_hits: list[str]
    trip_id: str
    trajectory: str
    n_rows: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "score": self.score,
            "n_columns": len(self.columns),
            "columns": self.columns,
            "canonical_hits": self.canonical_hits,
            "trip_id": self.trip_id,
            "trajectory": self.trajectory,
            "n_rows": self.n_rows,
        }


@dataclass
class DiscoveryResult:
    root: Path
    candidates: list[CandidateFile] = field(default_factory=list)
    selected: list[CandidateFile] = field(default_factory=list)
    skipped_non_helec: list[str] = field(default_factory=list)

    @property
    def n_selected(self) -> int:
        return len(self.selected)


def _default_skip_dirs() -> set[str]:
    return {
        ".git",
        ".venv",
        "venv",
        "env",
        "outputs",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "node_modules",
        ".eggs",
        "dist",
        "build",
    }


def infer_trip_id(path: Path) -> str:
    stem = path.name
    while stem.lower().endswith(".csv"):
        stem = stem[:-4]
    match = TRIP_STEM_RE.search(stem)
    if match:
        return match.group(1)
    return stem


def infer_trajectory(path: Path, trip_id: str) -> str:
    blob = f"{path.as_posix()}/{trip_id}"
    match = TRAJECTORY_RE.search(blob.replace("\\", "/"))
    if match:
        return match.group(1).upper()
    return "unknown"


def classify_kind(path: Path, headers: list[str]) -> str:
    path_l = path.as_posix().lower()
    normalized = {normalize_header(h) for h in headers}
    raw_hits = normalized & RAW_INDICATOR_NORMALIZED
    if "raw" in path_l.split("/"):
        return "raw"
    if "analysed" in path_l or "analyzed" in path_l:
        return "analysed"
    if raw_hits:
        return "raw"
    canon_hits = [ALIAS_LOOKUP[normalize_header(h)] for h in headers if normalize_header(h) in ALIAS_LOOKUP]
    if len(set(canon_hits)) >= 10:
        return "analysed"
    return "unknown"


def score_headers(headers: list[str]) -> tuple[int, list[str]]:
    hits: list[str] = []
    for header in headers:
        key = normalize_header(header)
        if key in ALIAS_LOOKUP:
            hits.append(ALIAS_LOOKUP[key])
    ordered = [name for name in EXPECTED_ANALYSED_CANONICAL if name in set(hits)]
    return len(ordered), ordered


def _iter_csv_files(roots: Iterable[Path], skip_dirs: set[str]) -> Iterable[Path]:
    skip_l = {d.lower() for d in skip_dirs}
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() == ".csv":
            yield root
            continue
        root = root.resolve()
        for path in root.rglob("*.csv"):
            try:
                rel = path.resolve().relative_to(root)
            except ValueError:
                rel = path
            dir_parts = {p.lower() for p in rel.parts[:-1]}
            if dir_parts & skip_l:
                continue
            yield path


def _read_headers(path: Path) -> list[str] | None:
    try:
        frame = pd.read_csv(path, nrows=0)
    except Exception:
        return None
    return [str(c) for c in frame.columns]


def discover_dataset(
    config: dict[str, Any] | None = None,
    search_roots: list[Path] | None = None,
    prefer_analysed: bool = True,
    min_score: int = 8,
) -> DiscoveryResult:
    """Locate candidate HELECAR-D CSVs. Never modifies files."""
    cfg = config or {}
    disc = cfg.get("discovery", {})
    skip_names = disc.get("skip_dir_names", None)
    skip = set(skip_names) if skip_names is not None else _default_skip_dirs()
    prefer = disc.get("prefer_analysed", prefer_analysed)
    root = project_root()
    if search_roots is None:
        raw_roots = cfg.get("paths", {}).get("data_search_roots", ["."])
        search_roots = [root / r if not Path(r).is_absolute() else Path(r) for r in raw_roots]

    result = DiscoveryResult(root=root)
    seen: set[Path] = set()
    for csv_path in _iter_csv_files(search_roots, skip):
        resolved = csv_path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        headers = _read_headers(resolved)
        if headers is None:
            result.skipped_non_helec.append(str(resolved))
            continue
        score, hits = score_headers(headers)
        if score < min_score:
            result.skipped_non_helec.append(str(resolved))
            continue
        trip_id = infer_trip_id(resolved)
        cand = CandidateFile(
            path=resolved,
            kind=classify_kind(resolved, headers),
            score=score,
            columns=headers,
            canonical_hits=hits,
            trip_id=trip_id,
            trajectory=infer_trajectory(resolved, trip_id),
        )
        result.candidates.append(cand)

    result.candidates.sort(key=lambda c: (c.trajectory, c.trip_id, c.kind, str(c.path)))
    result.selected = _select_trips(result.candidates, prefer_analysed=prefer)
    return result


def _select_trips(candidates: list[CandidateFile], prefer_analysed: bool) -> list[CandidateFile]:
    by_trip: dict[str, list[CandidateFile]] = {}
    for cand in candidates:
        by_trip.setdefault(cand.trip_id, []).append(cand)

    selected: list[CandidateFile] = []
    for trip_id in sorted(by_trip):
        group = by_trip[trip_id]
        if prefer_analysed:
            analysed = [c for c in group if c.kind == "analysed"]
            chosen_pool = analysed or group
        else:
            chosen_pool = group
        # Prefer higher score, then shorter path (often the analysed file).
        chosen_pool = sorted(chosen_pool, key=lambda c: (-c.score, len(str(c.path)), str(c.path)))
        selected.append(chosen_pool[0])
    selected.sort(key=lambda c: (c.trajectory, c.trip_id))
    return selected


def directory_tree_lines(paths: list[Path], root: Path) -> list[str]:
    """Compact listing of discovered files relative to the project root."""
    lines: list[str] = []
    for path in sorted(paths):
        try:
            rel = path.resolve().relative_to(root.resolve())
        except ValueError:
            rel = path
        lines.append(str(rel).replace("\\", "/"))
    return lines
