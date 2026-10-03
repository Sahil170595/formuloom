from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from formuloom.assemble import assemble_diff_file
from formuloom.bundle import TaskBundle
from formuloom.classify import UsageRecord
from formuloom.schema import DiffFile, VariantConfig


@dataclass(frozen=True)
class PipelineResult:

    diff: DiffFile
    failures: dict[str, str]
    usage: UsageRecord


class Pipeline(Protocol):

    async def predict(self, bundle: TaskBundle, *, run_dir: Path, best_effort: bool) -> PipelineResult: ...


def final_map_from_diff(diff: DiffFile) -> dict[str, set[str]]:
    return {sheet: diff.final_refs(sheet) for sheet in diff.sheets}


def intersect_predicted_final(per_constituent: Sequence[Mapping[str, Set[str]]]) -> dict[str, set[str]]:
    if not per_constituent:
        return {}
    all_sheets: set[str] = set()
    for constituent in per_constituent:
        all_sheets |= set(constituent)
    merged: dict[str, set[str]] = {}
    for sheet in all_sheets:
        per_sheet_sets = [set(constituent.get(sheet, frozenset())) for constituent in per_constituent]
        merged[sheet] = set.intersection(*per_sheet_sets)
    return merged


def _merge_failures(named_failures: Sequence[tuple[str, Mapping[str, str]]]) -> dict[str, str]:
    merged: dict[str, str] = {}
    for variant_name, failures in named_failures:
        for sheet, message in failures.items():
            prefixed = f"{variant_name}: {message}"
            merged[sheet] = f"{merged[sheet]} | {prefixed}" if sheet in merged else prefixed
    return merged


class ComposePipeline:

    def __init__(
        self,
        constituents: Sequence[VariantConfig],
        pipeline_factory: Callable[[VariantConfig], Pipeline],
    ) -> None:
        if not constituents:
            raise ValueError("ComposePipeline requires at least one constituent variant")
        for constituent in constituents:
            if constituent.compose_intersect:
                raise ValueError(
                    f"constituent {constituent.name!r} is itself a compose variant; nested composition "
                    "is not supported (constituents must be plain variants)"
                )
        self._constituents = tuple(constituents)
        self._factory = pipeline_factory

    async def predict(self, bundle: TaskBundle, *, run_dir: Path, best_effort: bool) -> PipelineResult:
        sub_results: list[PipelineResult] = []
        for constituent in self._constituents:
            pipeline = self._factory(constituent)
            sub_results.append(await pipeline.predict(bundle, run_dir=run_dir, best_effort=best_effort))

        per_constituent_finals = [final_map_from_diff(result.diff) for result in sub_results]
        merged_final = intersect_predicted_final(per_constituent_finals)
        diff = assemble_diff_file(bundle.raw_diff, merged_final)

        usage = UsageRecord()
        for result in sub_results:
            usage = usage.combined_with(result.usage)
        failures = _merge_failures(
            [
                (constituent.name, result.failures)
                for constituent, result in zip(self._constituents, sub_results, strict=True)
            ]
        )
        return PipelineResult(diff=diff, failures=failures, usage=usage)
