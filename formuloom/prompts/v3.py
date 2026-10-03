"""Workbook-scope conservative prompt arm."""

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
First inspect the workbook map. A feeder schedule is more likely supporting work
than a reader-facing result. Prefer INTERMEDIATE on ambiguous feeder rows. Apply
the uncertainty preference only when there is evidence of a section destination."""


def static_prefix(mode: ClassifyMode) -> str:
    schema = ROW_SCHEMA_DESCRIPTION if mode == "row" else CELL_SCHEMA_DESCRIPTION
    return f"{SYSTEM_PROMPT}\n{schema}\n{EXEMPLARS}"
