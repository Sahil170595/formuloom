import pytest

from formuloom.prompts import get_prompt_version
from formuloom.prompts.v1 import EXEMPLARS
from formuloom.prompts.v5_adjudicate import ADJUDICATE_JSON_SCHEMA, build_adjudication_input


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5", "v6", "v11"])
@pytest.mark.parametrize("mode", ["row", "cell"])
def test_fresh_prompt_contract_is_repeatable_and_mode_specific(version, mode):
    prompt = get_prompt_version(version)
    prefix = prompt.static_prefix(mode)
    assert prefix == prompt.static_prefix(mode)
    assert "final_rows" in prefix if mode == "row" else "final_cells" in prefix
    assert EXEMPLARS in prefix
    assert "data" in prefix


def test_distinct_policies_produce_distinct_ensemble_inputs():
    assert len({get_prompt_version(v).static_prefix("row") for v in ("v3", "v4", "v6")}) == 3


def test_prune_only_payload_contains_only_proposed_rows():
    text = build_adjudication_input(
        sheet="Synthetic",
        workbook_map_line="map",
        section_titles=[],
        likely_capstone_outputs=[],
        proposed_row_lines=[(2, "row 2 | total"), (8, "row 8 | net")],
    )
    assert "row 2 | total" in text and "row 8 | net" in text
    assert ADJUDICATE_JSON_SCHEMA["properties"]["rows"]["items"]["properties"]["verdict"]["enum"] == ["keep", "drop"]


def test_unknown_prompt_fails_loudly():
    with pytest.raises(ValueError):
        get_prompt_version("missing")
