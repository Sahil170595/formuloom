"""Density and profile-checklist extension."""

from .v1 import (
    CELL_SCHEMA_DESCRIPTION,
    EXEMPLARS,
    ROW_SCHEMA_DESCRIPTION,
    ClassifyMode,
)
from .v4 import SYSTEM_PROMPT as BASE
from .v4 import (
    TASK_PROFILE_SYSTEM_PROMPT as TASK_PROFILE_SYSTEM_PROMPT,
)

SYSTEM_PROMPT = BASE + """
If nearly every candidate is FINAL, review whether supporting calculations were
included. Compare against the task profile when present, but it is a checklist,
not an exhaustive whitelist of possible section outputs."""


def static_prefix(mode: ClassifyMode) -> str:
    schema = ROW_SCHEMA_DESCRIPTION if mode == "row" else CELL_SCHEMA_DESCRIPTION
    return f"{SYSTEM_PROMPT}\n{schema}\n{EXEMPLARS}"
