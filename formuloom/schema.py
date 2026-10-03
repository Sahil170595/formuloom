from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

SPEC_VERSION: int = 2

TASK_TOLERANCE: float = 0.01

CellType = Literal["currency", "date", "number", "percentage", "text"]

ScoreMode = Literal["strict", "annotated"]
GroupingMode = Literal["auto", "row", "cell", "block"]
UniversePolicy = Literal["raw", "raw_union_diff"]
ReasoningEffort = Literal["none", "low", "medium"]

_CELL_REF_RE = re.compile(r"([A-Za-z]{1,3})([1-9][0-9]{0,6})")
_MAX_EXCEL_ROW = 1_048_576
_MAX_EXCEL_COLUMN = 16_384


def _cell_coordinates(ref: str) -> tuple[str, int, int]:
    match = _CELL_REF_RE.fullmatch(ref)
    if match is None:
        raise ValueError(f"not a plain A1 cell ref: {ref!r}")
    letters = match.group(1).upper()
    row = int(match.group(2))
    column = 0
    for char in letters:
        column = column * 26 + (ord(char) - ord("A") + 1)
    if row > _MAX_EXCEL_ROW or column > _MAX_EXCEL_COLUMN:
        raise ValueError(f"cell ref outside Excel bounds (A1:XFD1048576): {ref!r}")
    return letters, row, column


def cell_column_letters(ref: str) -> str:
    return _cell_coordinates(ref)[0]


def cell_row(ref: str) -> int:
    return _cell_coordinates(ref)[1]


def cell_column_index(ref: str) -> int:
    return _cell_coordinates(ref)[2]


def cell_sort_key(ref: str) -> tuple[int, int]:
    _, row, column = _cell_coordinates(ref)
    return row, column


class CellEntry(BaseModel):

    model_config = ConfigDict(extra="forbid")

    cell: str
    cell_type: CellType

    @field_validator("cell")
    @classmethod
    def canonicalize_cell(cls, value: str) -> str:
        letters, row, _ = _cell_coordinates(value)
        return f"{letters}{row}"


class CellGroup(BaseModel):

    model_config = ConfigDict(extra="forbid")

    weight: float
    cells: list[CellEntry] = Field(default_factory=list)


class SheetGroups(BaseModel):

    model_config = ConfigDict(extra="forbid")

    intermediate: CellGroup | None = None
    final: CellGroup | None = None


class SheetDiff(BaseModel):

    model_config = ConfigDict(extra="forbid")

    sheet_weight: float
    groups: SheetGroups


class DiffFile(BaseModel):

    model_config = ConfigDict(extra="forbid")

    spec_version: int = SPEC_VERSION
    task_tolerance: float = TASK_TOLERANCE
    sheets: dict[str, SheetDiff]

    def final_refs(self, sheet: str) -> set[str]:
        sheet_diff = self.sheets.get(sheet)
        if sheet_diff is None or sheet_diff.groups.final is None:
            return set()
        return {entry.cell for entry in sheet_diff.groups.final.cells}

    def intermediate_refs(self, sheet: str) -> set[str]:
        sheet_diff = self.sheets.get(sheet)
        if sheet_diff is None or sheet_diff.groups.intermediate is None:
            return set()
        return {entry.cell for entry in sheet_diff.groups.intermediate.cells}

    def final_types(self, sheet: str) -> dict[str, CellType]:
        sheet_diff = self.sheets.get(sheet)
        if sheet_diff is None or sheet_diff.groups.final is None:
            return {}
        return {entry.cell: entry.cell_type for entry in sheet_diff.groups.final.cells}

    def intermediate_types(self, sheet: str) -> dict[str, CellType]:
        sheet_diff = self.sheets.get(sheet)
        if sheet_diff is None or sheet_diff.groups.intermediate is None:
            return {}
        return {entry.cell: entry.cell_type for entry in sheet_diff.groups.intermediate.cells}


class VariantConfig(BaseModel):

    model_config = ConfigDict(extra="forbid")

    name: str
    grouping: GroupingMode = "auto"
    universe: UniversePolicy = "raw"
    compression: bool = True
    features_in_prompt: bool = False
    section_split: bool = False
    full_dump: bool = False
    prompt_version: str = "v1"
    model: str = "gpt-5.4-mini"
    reasoning_effort: ReasoningEffort = "low"
    voting_k: int = Field(default=1, ge=1)
    selective_voting: bool = False
    use_builtin_tooling: bool = False
    use_task_profile: bool = True
    adjudicate: bool = False
    adjudicator_model: str | None = None
    ensemble_prompts: tuple[str, ...] = ()
    cascade: bool = False
    route_variants: tuple[str, ...] = ()
    compose_intersect: tuple[str, ...] = ()
    scout_route: bool = False


class TaskScore(BaseModel):

    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float


class SheetBreakdown(BaseModel):

    model_config = ConfigDict(extra="forbid")

    sheet: str
    tp: int
    fp: int
    fn: int
    golden_intermediate: int = 0


class TaskScoreDetail(BaseModel):

    model_config = ConfigDict(extra="forbid")

    task: str
    mode: ScoreMode
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    per_sheet: list[SheetBreakdown]


class MetricTriple(BaseModel):

    model_config = ConfigDict(extra="forbid")

    precision: float
    recall: float
    f1: float


class Aggregates(BaseModel):

    model_config = ConfigDict(extra="forbid")

    micro: MetricTriple
    macro: MetricTriple
