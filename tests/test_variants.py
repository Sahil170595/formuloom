from __future__ import annotations

import pytest

from formuloom.schema import VariantConfig
from formuloom.variants import RULES_ONLY_VARIANT, VARIANTS, get_variant, is_rules_only

EXPECTED_NAMES = {"V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V11", "V12", "V13", "V14", "V15", "V16"}


def test_registry_has_exactly_the_planned_variants() -> None:
    assert set(VARIANTS) == EXPECTED_NAMES


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_every_variant_is_a_real_variant_config_with_matching_name(name: str) -> None:
    config = VARIANTS[name]
    assert isinstance(config, VariantConfig)
    assert config.name == name


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES))
def test_no_variant_uses_the_cut_block_grouping_mode(name: str) -> None:

    assert VARIANTS[name].grouping in ("row", "cell", "auto")


def test_v0_is_rules_only_no_grouping_ambiguity() -> None:
    v0 = VARIANTS["V0"]
    assert v0.grouping == "row"
    assert v0.universe == "raw"
    assert is_rules_only("V0") is True


@pytest.mark.parametrize("name", sorted(EXPECTED_NAMES - {"V0"}))
def test_only_v0_is_rules_only(name: str) -> None:
    assert is_rules_only(name) is False


def test_v1_is_row_level_compressed_no_features_no_split_no_voting() -> None:
    v1 = VARIANTS["V1"]
    assert v1.compression is True
    assert v1.features_in_prompt is False
    assert v1.section_split is False
    assert v1.full_dump is False
    assert v1.voting_k == 1
    assert v1.selective_voting is False


def test_v2_is_v1_plus_features_in_prompt() -> None:
    v1, v2 = VARIANTS["V1"], VARIANTS["V2"]
    assert v2.features_in_prompt is True

    assert v2.grouping == v1.grouping
    assert v2.compression == v1.compression
    assert v2.section_split == v1.section_split
    assert v2.full_dump == v1.full_dump
    assert v2.voting_k == v1.voting_k


def test_v3_is_v2_plus_section_split() -> None:
    v2, v3 = VARIANTS["V2"], VARIANTS["V3"]
    assert v3.features_in_prompt == v2.features_in_prompt is True
    assert v3.section_split is True
    assert v2.section_split is False


def test_v4_is_v2_plus_selective_voting_k3() -> None:
    v2, v4 = VARIANTS["V2"], VARIANTS["V4"]
    assert v4.features_in_prompt == v2.features_in_prompt is True
    assert v4.voting_k == 3
    assert v4.selective_voting is True
    assert v2.voting_k == 1
    assert v2.selective_voting is False


def test_v5_is_full_dump_no_compression() -> None:
    v5 = VARIANTS["V5"]
    assert v5.full_dump is True
    assert v5.compression is False
    assert v5.features_in_prompt is False
    assert v5.section_split is False


def test_v6_is_builtin_tooling_cell_mode() -> None:

    v6 = VARIANTS["V6"]
    assert v6.use_builtin_tooling is True
    assert v6.grouping == "cell"
    assert v6.features_in_prompt is False
    assert is_rules_only("V6") is False


def test_get_variant_returns_registered_config() -> None:
    assert get_variant("V2") is VARIANTS["V2"]


def test_v9_is_v7_without_the_task_profile() -> None:

    v7, v9 = VARIANTS["V7"], VARIANTS["V9"]
    assert v9.use_task_profile is False
    assert v7.use_task_profile is True

    assert v9.features_in_prompt is True
    assert v9.prompt_version == "v4"
    assert v9.voting_k == 3
    assert v9.selective_voting is False
    assert v9.adjudicate is False
    assert v9.model == v7.model


def test_v11_is_full_cost_v11_prompt_with_adjudication() -> None:
    v11 = VARIANTS["V11"]
    assert v11.features_in_prompt is True
    assert v11.prompt_version == "v11"
    assert v11.model == "gpt-5.4"
    assert v11.voting_k == 3
    assert v11.selective_voting is False
    assert v11.adjudicate is True


def test_v12_is_diverse_prompt_ensemble() -> None:

    v12 = VARIANTS["V12"]
    assert v12.features_in_prompt is True
    assert v12.ensemble_prompts == ("v3", "v4", "v6")
    assert v12.voting_k == 1
    assert v12.selective_voting is False
    assert is_rules_only("V12") is False


def test_v13_is_consensus_intersection_of_v9_and_v11() -> None:

    v13 = VARIANTS["V13"]
    assert v13.compose_intersect == ("V9", "V11")

    for name in v13.compose_intersect:
        assert name in VARIANTS
        assert VARIANTS[name].compose_intersect == ()
    assert is_rules_only("V13") is False


def test_v15_is_structural_router_over_v11_and_v9() -> None:
    v15 = VARIANTS["V15"]
    assert v15.route_variants == ("V11", "V9")
    assert v15.model == "V11+V9 (router)"
    assert v15.prompt_version == "v15-router"
    assert v15.ensemble_prompts == ()
    assert v15.compose_intersect == ()
    for name in v15.route_variants:
        assert name in VARIANTS
        assert VARIANTS[name].route_variants == ()


def test_no_non_composite_variant_sets_compose_intersect() -> None:

    for name in EXPECTED_NAMES - {"V13"}:
        assert VARIANTS[name].compose_intersect == ()


def test_get_variant_unknown_name_raises_with_available_list() -> None:
    with pytest.raises(ValueError, match="unknown variant"):
        get_variant("V99")


def test_rules_only_variant_constant_matches_registry_key() -> None:
    assert RULES_ONLY_VARIANT in VARIANTS
    assert is_rules_only(RULES_ONLY_VARIANT) is True


def test_v16_is_scout_routed() -> None:

    v16 = VARIANTS["V16"]
    assert v16.scout_route is True
    assert v16.compose_intersect == ()
    assert v16.cascade is False
    assert is_rules_only("V16") is False


def test_no_non_scout_route_variant_sets_scout_route() -> None:

    for name in EXPECTED_NAMES - {"V16"}:
        assert VARIANTS[name].scout_route is False


def test_v14_is_consensus_cascade() -> None:

    v14 = VARIANTS["V14"]
    assert v14.cascade is True
    assert v14.grouping == "row"
    assert v14.features_in_prompt is True
    assert v14.use_task_profile is False
    assert v14.ensemble_prompts == ("v3", "v4", "v6")
    assert v14.adjudicator_model == "gpt-5.4"
    assert v14.model == "gpt-5.4-mini"
    assert v14.voting_k == 1
    assert is_rules_only("V14") is False
