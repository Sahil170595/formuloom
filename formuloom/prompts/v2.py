"""Graph-role lens for the public policy; no imported worked examples."""

from .v1 import (
    CELL_SCHEMA_DESCRIPTION,
    EXEMPLARS,
    ROW_SCHEMA_DESCRIPTION,
    ClassifyMode,
)
from .v1 import SYSTEM_PROMPT as BASE
from .v1 import (
    TASK_PROFILE_SYSTEM_PROMPT as TASK_PROFILE_SYSTEM_PROMPT,
)

SYSTEM_PROMPT = BASE + """
Distinguish input (no formula), working (formula with readers), output (formula
without readers), and label roles. Use deps_out, src_sheets, src_sections, agg,
lex and style hints as evidence. A working row can be a requested section output.
For genuine uncertainty with output evidence, prefer FINAL."""


def static_prefix(mode: ClassifyMode) -> str:
    schema = ROW_SCHEMA_DESCRIPTION if mode == "row" else CELL_SCHEMA_DESCRIPTION
    return f"{SYSTEM_PROMPT}\n{schema}\n{EXEMPLARS}"
