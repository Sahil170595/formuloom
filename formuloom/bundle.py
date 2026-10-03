from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from formuloom.schema import CellEntry, DiffFile

BundleMode = Literal["predict", "score"]

INIT_FILE = "init.xlsx"
COMPLETE_FILE = "complete.xlsx"
RAW_DIFF_FILE = "raw_diff.json"
INSTRUCTIONS_FILE = "instructions.md"
SUBSET_FILE = "subset.json"

PREDICT_ALLOWLIST: frozenset[str] = frozenset({INIT_FILE, COMPLETE_FILE, RAW_DIFF_FILE, INSTRUCTIONS_FILE})

SCORE_ALLOWLIST: frozenset[str] = PREDICT_ALLOWLIST | frozenset({SUBSET_FILE})


class BundleError(Exception):
    pass


class MissingBundleFileError(BundleError):
    pass


class MalformedBundleError(BundleError):
    pass


class LeakageError(BundleError):
    pass


def _allowlist_for(mode: BundleMode) -> frozenset[str]:
    if mode == "predict":
        return PREDICT_ALLOWLIST
    return SCORE_ALLOWLIST


def _read_text(path: Path, allowlist: frozenset[str]) -> str:
    if path.name not in allowlist:
        raise LeakageError(
            f"refusing to open {path.name!r}: not permitted for this bundle mode " f"(allowed: {sorted(allowlist)})"
        )
    return path.read_text(encoding="utf-8")


def _load_diff(path: Path, allowlist: frozenset[str]) -> DiffFile:
    if not path.exists():
        raise MissingBundleFileError(f"required bundle file missing: {path}")
    raw = _read_text(path, allowlist)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MalformedBundleError(f"{path.name} is not valid JSON: {exc}") from exc
    try:
        return DiffFile.model_validate(payload)
    except ValidationError as exc:
        raise MalformedBundleError(f"{path.name} does not match the diff schema: {exc}") from exc


@dataclass(frozen=True)
class TaskBundle:

    task_dir: Path
    mode: BundleMode
    raw_diff: DiffFile
    golden: DiffFile | None
    instructions: str
    init_path: Path
    complete_path: Path

    @property
    def name(self) -> str:
        return self.task_dir.name

    @classmethod
    def load(cls, task_dir: Path | str, mode: BundleMode) -> TaskBundle:
        task_dir = Path(task_dir)
        if not task_dir.is_dir():
            raise MissingBundleFileError(f"task directory does not exist: {task_dir}")

        allowlist = _allowlist_for(mode)

        init_path = task_dir / INIT_FILE
        complete_path = task_dir / COMPLETE_FILE
        raw_diff_path = task_dir / RAW_DIFF_FILE
        instructions_path = task_dir / INSTRUCTIONS_FILE

        for required in (init_path, complete_path, raw_diff_path, instructions_path):
            if not required.exists():
                raise MissingBundleFileError(f"required bundle file missing: {required}")

        raw_diff = _load_diff(raw_diff_path, allowlist)
        instructions = _read_text(instructions_path, allowlist)

        golden: DiffFile | None = None
        if mode == "score":
            subset_path = task_dir / SUBSET_FILE
            if not subset_path.exists():
                raise MissingBundleFileError(f"score mode requires the golden subset: {subset_path}")
            golden = _load_diff(subset_path, allowlist)

        return cls(
            task_dir=task_dir,
            mode=mode,
            raw_diff=raw_diff,
            golden=golden,
            instructions=instructions,
            init_path=init_path,
            complete_path=complete_path,
        )

    def candidate_cells(self) -> dict[str, list[CellEntry]]:
        out: dict[str, list[CellEntry]] = {}
        for sheet_name, sheet in self.raw_diff.sheets.items():
            seen: set[str] = set()
            cells: list[CellEntry] = []
            for group in (sheet.groups.intermediate, sheet.groups.final):
                if group is None:
                    continue
                for entry in group.cells:
                    if entry.cell in seen:
                        continue
                    seen.add(entry.cell)
                    cells.append(entry)
            out[sheet_name] = cells
        return out
