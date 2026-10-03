"""Precision-oriented support-sheet-safe arm, used by V11."""

from .v1 import (
    CELL_SCHEMA_DESCRIPTION,
    EXEMPLARS,
    ROW_SCHEMA_DESCRIPTION,
    ClassifyMode,
)
from .v2 import SYSTEM_PROMPT as BASE
from .v2 import (
    TASK_PROFILE_SYSTEM_PROMPT as TASK_PROFILE_SYSTEM_PROMPT,
)

SYSTEM_PROMPT = BASE + """
Support-sheet guard: graph sinks in assumption grids can be unused parameters,
not deliverables. Verify the sheet's purpose before accepting terminal cells.
Source statement totals can still be useful outputs despite having no formula.
Demand destination evidence before promoting local ratios or scenario endpoints.
Review large FINAL proposals for a mistaken supporting-table interpretation."""


def static_prefix(mode: ClassifyMode) -> str:
    schema = ROW_SCHEMA_DESCRIPTION if mode == "row" else CELL_SCHEMA_DESCRIPTION
    return f"{SYSTEM_PROMPT}\n{schema}\n{EXEMPLARS}"
