"""Recall-oriented row-evidence arm, used by V9."""

from .v1 import (
    CELL_SCHEMA_DESCRIPTION,
    EXEMPLARS,
    ROW_SCHEMA_DESCRIPTION,
    ClassifyMode,
)
from .v3 import SYSTEM_PROMPT as BASE
from .v3 import (
    TASK_PROFILE_SYSTEM_PROMPT as TASK_PROFILE_SYSTEM_PROMPT,
)

SYSTEM_PROMPT = BASE + """
The feeder prior is defeasible: clear evidence that a row is a requested section
result overrides the sheet's supporting role. Judge this from structure and row
content, not a sheet-name whitelist. Keep signal-bearing ambiguous outputs."""


def static_prefix(mode: ClassifyMode) -> str:
    schema = ROW_SCHEMA_DESCRIPTION if mode == "row" else CELL_SCHEMA_DESCRIPTION
    return f"{SYSTEM_PROMPT}\n{schema}\n{EXEMPLARS}"
