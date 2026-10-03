from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from . import v1, v2, v3, v4, v5, v6, v11
from .v1 import ClassifyMode

__all__ = ["ClassifyMode", "PromptVersion", "get_prompt_version"]


@dataclass(frozen=True)
class PromptVersion:

    name: str
    static_prefix: Callable[[ClassifyMode], str]
    task_profile_system_prompt: str


_REGISTRY: dict[str, PromptVersion] = {
    "v1": PromptVersion(
        name="v1",
        static_prefix=v1.static_prefix,
        task_profile_system_prompt=v1.TASK_PROFILE_SYSTEM_PROMPT,
    ),
    "v2": PromptVersion(
        name="v2",
        static_prefix=v2.static_prefix,
        task_profile_system_prompt=v2.TASK_PROFILE_SYSTEM_PROMPT,
    ),
    "v3": PromptVersion(
        name="v3",
        static_prefix=v3.static_prefix,
        task_profile_system_prompt=v3.TASK_PROFILE_SYSTEM_PROMPT,
    ),
    "v4": PromptVersion(
        name="v4",
        static_prefix=v4.static_prefix,
        task_profile_system_prompt=v4.TASK_PROFILE_SYSTEM_PROMPT,
    ),
    "v5": PromptVersion(
        name="v5",
        static_prefix=v5.static_prefix,
        task_profile_system_prompt=v5.TASK_PROFILE_SYSTEM_PROMPT,
    ),
    "v6": PromptVersion(
        name="v6",
        static_prefix=v6.static_prefix,
        task_profile_system_prompt=v6.TASK_PROFILE_SYSTEM_PROMPT,
    ),
    "v11": PromptVersion(
        name="v11",
        static_prefix=v11.static_prefix,
        task_profile_system_prompt=v11.TASK_PROFILE_SYSTEM_PROMPT,
    ),
}


def get_prompt_version(version: str) -> PromptVersion:
    try:
        return _REGISTRY[version]
    except KeyError as exc:
        raise ValueError(f"unknown prompt version {version!r} (available: {sorted(_REGISTRY)})") from exc
