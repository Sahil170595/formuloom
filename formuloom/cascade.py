from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from formuloom.adjudicate import adjudicate_sheet, build_adjudication_input
from formuloom.assemble import assemble_diff_file
from formuloom.bundle import TaskBundle
from formuloom.classify import ClassifyError, UsageRecord, build_async_client
from formuloom.compose import PipelineResult
from formuloom.encode import SheetContext, sheet_map_line
from formuloom.ensemble import PromptVoteRecord, classify_sheet_diverse
from formuloom.features import build_task_features
from formuloom.schema import DiffFile, ReasoningEffort, VariantConfig
from formuloom.settings import Settings
from formuloom.workbook import load_workbook_data

logger = logging.getLogger(__name__)

CASCADE_PROMPT_VERSIONS: tuple[str, ...] = ("v3", "v4", "v6")

ENSEMBLE_SAMPLES_PER_PROMPT: int = 3

ENSEMBLE_MODEL: str = "gpt-5.4-mini"

ENSEMBLE_EFFORT: ReasoningEffort = "low"

CASCADE_JUDGE_MODEL: str = "gpt-5.4"

CASCADE_JUDGE_EFFORT: ReasoningEffort = "low"

CASCADE_JUDGE_PROMPT_VERSION: str = "cascade"

HIGH_CONSENSUS: int = 7

LOW_CONSENSUS: int = 2

BAND_KEEP_VOTES: int = 4

TIER_STATS_SUBDIR: str = "cascade"


def row_vote_counts(candidates: Sequence[int], per_vote: Sequence[PromptVoteRecord]) -> dict[int, int]:
    counts = {row: 0 for row in candidates}
    for record in per_vote:
        for ref in record.final_refs:
            row = int(ref)
            if row in counts:
                counts[row] += 1
    return counts


def assign_tiers(
    candidates: Sequence[int], vote_counts: Mapping[int, int], *, high: int, low: int
) -> tuple[list[int], list[int], list[int]]:
    if high <= low:
        raise ValueError(f"require high ({high}) > low ({low}) for a non-empty consensus band")
    final: list[int] = []
    intermediate: list[int] = []
    band: list[int] = []
    for row in candidates:
        v = vote_counts.get(row, 0)
        if v >= high:
            final.append(row)
        elif v <= low:
            intermediate.append(row)
        else:
            band.append(row)
    return final, intermediate, band


def resolve_band(
    band_rows: Sequence[int],
    vote_counts: Mapping[int, int],
    n_votes: int,
    judge_keep: frozenset[int] | set[int],
    *,
    degraded: bool,
    band_keep_votes: int,
) -> set[int]:
    if degraded:
        return {row for row in band_rows if 2 * vote_counts.get(row, 0) >= n_votes}
    return {row for row in band_rows if row in judge_keep or vote_counts.get(row, 0) >= band_keep_votes}


@dataclass(frozen=True)
class SheetCascadeResult:

    sheet: str
    final_cells: frozenset[str]
    usage: UsageRecord
    n_votes: int
    n_high: int
    n_low: int
    n_band: int
    band_kept: int
    band_dropped: int
    band_degraded: bool

    def stats_payload(self) -> dict[str, object]:
        return {
            "sheet": self.sheet,
            "n_votes": self.n_votes,
            "n_high": self.n_high,
            "n_low": self.n_low,
            "n_band": self.n_band,
            "band_kept": self.band_kept,
            "band_dropped": self.band_dropped,
            "band_degraded": self.band_degraded,
        }


async def cascade_sheet(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    task: str,
    sheet: str,
    variant_name: str,
    context: SheetContext,
    row_to_cells: Mapping[int, Sequence[str]],
    workbook_map_line: str,
    run_dir: Path,
    prompt_versions: Sequence[str] = CASCADE_PROMPT_VERSIONS,
    samples_per_prompt: int = ENSEMBLE_SAMPLES_PER_PROMPT,
    ensemble_model: str = ENSEMBLE_MODEL,
    ensemble_effort: ReasoningEffort = ENSEMBLE_EFFORT,
    judge_model: str = CASCADE_JUDGE_MODEL,
    judge_effort: ReasoningEffort = CASCADE_JUDGE_EFFORT,
    judge_prompt_version: str = CASCADE_JUDGE_PROMPT_VERSION,
    high_consensus: int = HIGH_CONSENSUS,
    low_consensus: int = LOW_CONSENSUS,
    band_keep_votes: int = BAND_KEEP_VOTES,
) -> SheetCascadeResult:
    if context.grouping != "row":
        raise ValueError(f"cascade_sheet requires row-mode context, got grouping={context.grouping!r} for {sheet!r}")

    candidates = context.rows
    if not candidates:
        return SheetCascadeResult(sheet, frozenset(), UsageRecord(), 0, 0, 0, 0, 0, 0, False)

    ensemble = await classify_sheet_diverse(
        client,
        settings,
        task=task,
        sheet=sheet,
        variant_name=variant_name,
        prompt_versions=tuple(prompt_versions),
        mode="row",
        context=context.text,
        candidates=candidates,
        run_dir=run_dir,
        model=ensemble_model,
        reasoning_effort=ensemble_effort,
        samples_per_prompt=samples_per_prompt,
    )
    n_votes = ensemble.n_votes
    vote_counts = row_vote_counts(candidates, ensemble.per_vote)
    high_rows, low_rows, band_rows = assign_tiers(candidates, vote_counts, high=high_consensus, low=low_consensus)

    final_rows: set[int] = set(high_rows)
    usage = ensemble.usage
    band_kept: set[int] = set()
    degraded = False

    if band_rows:
        band_set = set(band_rows)
        adjudication_input = build_adjudication_input(
            sheet=sheet,
            workbook_map_line=workbook_map_line,
            section_titles=[s.title for s in context.sections if s.title],
            likely_capstone_outputs=[],
            proposed_row_lines=[(row, line) for row, line in context.row_lines if row in band_set],
        )
        adjudication = await adjudicate_sheet(
            client,
            settings,
            task=task,
            sheet=sheet,
            variant_name=variant_name,
            prompt_version=judge_prompt_version,
            input_text=adjudication_input,
            proposed_rows=band_rows,
            run_dir=run_dir,
            model=judge_model,
            reasoning_effort=judge_effort,
        )
        usage = usage.combined_with(adjudication.usage)
        degraded = adjudication.degraded
        band_kept = resolve_band(
            band_rows,
            vote_counts,
            n_votes,
            set(adjudication.keep_rows),
            degraded=adjudication.degraded,
            band_keep_votes=band_keep_votes,
        )
        final_rows |= band_kept

    final_cells = frozenset(ref for row in final_rows for ref in row_to_cells.get(row, ()))
    return SheetCascadeResult(
        sheet=sheet,
        final_cells=final_cells,
        usage=usage,
        n_votes=n_votes,
        n_high=len(high_rows),
        n_low=len(low_rows),
        n_band=len(band_rows),
        band_kept=len(band_kept),
        band_dropped=len(band_rows) - len(band_kept),
        band_degraded=degraded,
    )


def _write_tier_stats(run_dir: Path, task: str, results: Sequence[SheetCascadeResult]) -> None:
    total_high = sum(r.n_high for r in results)
    total_low = sum(r.n_low for r in results)
    total_band = sum(r.n_band for r in results)
    total_rows = total_high + total_low + total_band
    payload = {
        "task": task,
        "totals": {
            "rows": total_rows,
            "high_consensus_final": total_high,
            "low_consensus_intermediate": total_low,
            "band_escalated": total_band,
            "band_kept": sum(r.band_kept for r in results),
            "band_dropped": sum(r.band_dropped for r in results),
            "band_degraded_sheets": sum(1 for r in results if r.band_degraded),
            "consensus_decided_fraction": (total_high + total_low) / total_rows if total_rows else 1.0,
            "band_fraction": total_band / total_rows if total_rows else 0.0,
        },
        "per_sheet": [r.stats_payload() for r in results],
    }
    path = run_dir / TIER_STATS_SUBDIR / f"{task}.tiers.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


class CascadePipeline:

    def __init__(self, config: VariantConfig, settings: Settings) -> None:
        self._config = config
        self._settings = settings

    async def predict(self, bundle: TaskBundle, *, run_dir: Path, best_effort: bool) -> PipelineResult:
        config = self._config
        settings = self._settings
        client = build_async_client(settings)
        try:
            complete = load_workbook_data(bundle.complete_path)
            task_features = build_task_features(bundle, complete)
            candidates_by_sheet = bundle.candidate_cells()
            sheet_names = [s for s in bundle.raw_diff.sheets if candidates_by_sheet.get(s)]

            prompt_versions = config.ensemble_prompts or CASCADE_PROMPT_VERSIONS
            judge_model = config.adjudicator_model or CASCADE_JUDGE_MODEL

            async def _one_sheet(sheet_name: str) -> tuple[str, SheetCascadeResult | None, str | None]:
                try:
                    context = SheetContext.build(
                        bundle=bundle,
                        complete=complete,
                        features=task_features,
                        sheet_name=sheet_name,
                        variant=config,
                        profile=None,
                    )
                    sheet_features = task_features.sheets[sheet_name]
                    row_to_cells = {rf.row: rf.candidate_refs for rf in sheet_features.rows}
                    result = await cascade_sheet(
                        client,
                        settings,
                        task=bundle.name,
                        sheet=sheet_name,
                        variant_name=config.name,
                        context=context,
                        row_to_cells=row_to_cells,
                        workbook_map_line=sheet_map_line(sheet_name, sheet_features),
                        run_dir=run_dir,
                        prompt_versions=prompt_versions,
                        ensemble_model=config.model,
                        ensemble_effort=config.reasoning_effort,
                        judge_model=judge_model,
                        judge_effort=config.reasoning_effort,
                    )
                    return sheet_name, result, None
                except ClassifyError as exc:
                    if not best_effort:
                        raise
                    logger.error("sheet %r failed (--best-effort: degrading to all-intermediate): %s", sheet_name, exc)
                    return sheet_name, None, str(exc)

            outcomes = await asyncio.gather(*(_one_sheet(s) for s in sheet_names))

            predicted_final: dict[str, set[str]] = {}
            failures: dict[str, str] = {}
            usage = UsageRecord()
            tier_results: list[SheetCascadeResult] = []
            for sheet_name, result, error in outcomes:
                if result is not None:
                    predicted_final[sheet_name] = set(result.final_cells)
                    usage = usage.combined_with(result.usage)
                    tier_results.append(result)
                else:
                    predicted_final[sheet_name] = set()
                if error is not None:
                    failures[sheet_name] = error

            diff: DiffFile = assemble_diff_file(bundle.raw_diff, predicted_final)
            _write_tier_stats(run_dir, bundle.name, tier_results)
            return PipelineResult(diff=diff, failures=failures, usage=usage)
        finally:
            await client.close()
