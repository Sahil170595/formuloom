from __future__ import annotations

from collections.abc import Mapping, Sequence, Set

from formuloom.constants import GROUP_WEIGHT_FINAL, GROUP_WEIGHT_INTERMEDIATE, SPEC_VERSION, TASK_TOLERANCE
from formuloom.schema import CellEntry, CellGroup, DiffFile, SheetDiff, SheetGroups, cell_sort_key

_EMPTY_FINAL_INTERMEDIATE_WEIGHT = 1.0
_EMPTY_FINAL_FINAL_WEIGHT = 0.0


class AssembleError(Exception):
    pass


def _raw_universe(raw_diff: DiffFile, sheet: str) -> list[CellEntry]:
    sheet_diff = raw_diff.sheets.get(sheet)
    if sheet_diff is None:
        return []
    seen: set[str] = set()
    cells: list[CellEntry] = []
    for group in (sheet_diff.groups.intermediate, sheet_diff.groups.final):
        if group is None:
            continue
        for entry in group.cells:
            if entry.cell in seen:
                continue
            seen.add(entry.cell)
            cells.append(entry)
    return cells


def build_sheet_universe(raw_diff: DiffFile, sheet: str, extra_cells: Sequence[CellEntry] = ()) -> list[CellEntry]:
    raw_cells = _raw_universe(raw_diff, sheet)
    seen = {c.cell for c in raw_cells}
    extras: list[CellEntry] = []
    for entry in extra_cells:
        if entry.cell in seen:
            continue
        seen.add(entry.cell)
        extras.append(entry)
    extras.sort(key=lambda c: cell_sort_key(c.cell))
    return raw_cells + extras


def assemble_sheet_diff(
    raw_diff: DiffFile,
    sheet: str,
    predicted_final: Set[str],
    *,
    extra_cells: Sequence[CellEntry] = (),
) -> SheetDiff:
    if sheet not in raw_diff.sheets:
        raise AssembleError(f"unknown sheet {sheet!r}: not present in raw_diff (sheets: {sorted(raw_diff.sheets)})")
    sheet_weight = raw_diff.sheets[sheet].sheet_weight
    universe = build_sheet_universe(raw_diff, sheet, extra_cells)
    universe_refs = {c.cell for c in universe}

    unknown = predicted_final - universe_refs
    if unknown:
        raise AssembleError(
            f"sheet {sheet!r}: predicted-final ref(s) {sorted(unknown)} are not in the "
            f"candidate universe ({len(universe)} cells) -- upstream wiring bug"
        )

    final_cells = [c for c in universe if c.cell in predicted_final]
    intermediate_cells = [c for c in universe if c.cell not in predicted_final]

    if final_cells:
        intermediate_weight, final_weight = GROUP_WEIGHT_INTERMEDIATE, GROUP_WEIGHT_FINAL
    else:
        intermediate_weight, final_weight = _EMPTY_FINAL_INTERMEDIATE_WEIGHT, _EMPTY_FINAL_FINAL_WEIGHT

    return SheetDiff(
        sheet_weight=sheet_weight,
        groups=SheetGroups(
            intermediate=CellGroup(weight=intermediate_weight, cells=intermediate_cells),
            final=CellGroup(weight=final_weight, cells=final_cells),
        ),
    )


def assemble_diff_file(
    raw_diff: DiffFile,
    predicted_final: Mapping[str, Set[str]],
    *,
    extra_cells: Mapping[str, Sequence[CellEntry]] | None = None,
) -> DiffFile:
    extra_cells = extra_cells or {}
    unknown_sheets = set(predicted_final) - set(raw_diff.sheets)
    if unknown_sheets:
        raise AssembleError(
            f"predicted_final references unknown sheet(s) {sorted(unknown_sheets)}: not in "
            f"raw_diff (sheets: {sorted(raw_diff.sheets)})"
        )

    sheets: dict[str, SheetDiff] = {}
    for sheet in raw_diff.sheets:
        sheets[sheet] = assemble_sheet_diff(
            raw_diff,
            sheet,
            predicted_final.get(sheet, frozenset()),
            extra_cells=extra_cells.get(sheet, ()),
        )
    return DiffFile(spec_version=SPEC_VERSION, task_tolerance=TASK_TOLERANCE, sheets=sheets)
