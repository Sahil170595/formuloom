from __future__ import annotations

from formuloom.schema import VariantConfig

RULES_ONLY_VARIANT: str = "V0"

V4_VOTING_K: int = 3

_V1_KWARGS: dict[str, object] = {
    "grouping": "auto",
    "universe": "raw",
    "compression": True,
    "features_in_prompt": False,
    "section_split": False,
    "full_dump": False,
    "prompt_version": "v1",
    "model": "gpt-5.4-mini",
    "reasoning_effort": "low",
    "voting_k": 1,
    "selective_voting": False,
    "use_builtin_tooling": False,
}


def _variant(name: str, **overrides: object) -> VariantConfig:
    kwargs = {**_V1_KWARGS, **overrides, "name": name}
    return VariantConfig(**kwargs)  # type: ignore[arg-type]  # dict-spread defeats overload narrowing


VARIANTS: dict[str, VariantConfig] = {
    "V8": VariantConfig(
        name="V8", grouping="row", model="offline-dawid-skene", prompt_version="none", use_task_profile=False
    ),
    "V0": VariantConfig(
        name="V0",
        grouping="row",
        universe="raw",
        compression=True,
        features_in_prompt=False,
        section_split=False,
        full_dump=False,
        prompt_version="v1",
        voting_k=1,
    ),
    "V1": _variant("V1"),
    "V2": _variant("V2", features_in_prompt=True),
    "V3": _variant("V3", features_in_prompt=True, section_split=True),
    "V4": _variant("V4", features_in_prompt=True, voting_k=V4_VOTING_K, selective_voting=True),
    "V5": _variant("V5", compression=False, full_dump=True),
    "V6": _variant("V6", grouping="cell", use_builtin_tooling=True),
    "V7": _variant("V7", features_in_prompt=True, prompt_version="v4", voting_k=V4_VOTING_K, selective_voting=False),
    "V9": _variant(
        "V9",
        features_in_prompt=True,
        prompt_version="v4",
        voting_k=V4_VOTING_K,
        selective_voting=False,
        use_task_profile=False,
    ),
    "V11": _variant(
        "V11",
        features_in_prompt=True,
        prompt_version="v11",
        model="gpt-5.4",
        voting_k=V4_VOTING_K,
        selective_voting=False,
        adjudicate=True,
    ),
    "V12": _variant("V12", features_in_prompt=True, ensemble_prompts=("v3", "v4", "v6"), voting_k=1),
    "V14": _variant(
        "V14",
        grouping="row",
        features_in_prompt=True,
        prompt_version="v4",
        use_task_profile=False,
        ensemble_prompts=("v3", "v4", "v6"),
        adjudicator_model="gpt-5.4",
        cascade=True,
        voting_k=1,
    ),
    "V13": VariantConfig(
        name="V13",
        compose_intersect=("V9", "V11"),
        model="V9+V11 (composite)",
        prompt_version="compose-intersect",
    ),
    "V15": VariantConfig(
        name="V15",
        route_variants=("V11", "V9"),
        model="V11+V9 (router)",
        prompt_version="v15-router",
    ),
    "V16": VariantConfig(
        name="V16",
        scout_route=True,
        model="scout-routed (V9/V11/V13)",
        prompt_version="scout-route",
    ),
}


def get_variant(name: str) -> VariantConfig:
    try:
        return VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f"unknown variant {name!r} (available: {sorted(VARIANTS)})") from exc


def is_rules_only(name: str) -> bool:
    return name == RULES_ONLY_VARIANT


def is_offline(name: str) -> bool:
    return name in {RULES_ONLY_VARIANT, "V8"}
