from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from formuloom.classify import ClassifyError, ClassifySheetResult, UsageRecord, classify_sheet
from formuloom.prompts import ClassifyMode
from formuloom.schema import ReasoningEffort
from formuloom.settings import Settings

logger = logging.getLogger(__name__)

MAJORITY_TIE_TO_FINAL: bool = True


@dataclass(frozen=True)
class PromptVoteRecord:

    prompt_version: str
    sample_index: int
    final_refs: tuple[int, ...] | tuple[str, ...]
    usage: UsageRecord

    def dissent_count(self, majority_refs: frozenset[str]) -> int:
        own = {str(r) for r in self.final_refs}
        return len(own ^ majority_refs)


@dataclass(frozen=True)
class DiverseEnsembleResult:

    sheet: str
    mode: ClassifyMode
    prompt_versions: tuple[str, ...]
    n_votes: int
    n_failed_votes: int
    majority_refs: tuple[int, ...] | tuple[str, ...]
    agreement_rate: float
    per_vote: tuple[PromptVoteRecord, ...]
    usage: UsageRecord


@dataclass(frozen=True)
class _VoteSpec:

    position: int
    prompt_version: str
    sample_index: int
    cache_suffix: str


def _plan_votes(prompt_versions: Sequence[str], samples_per_prompt: int) -> list[_VoteSpec]:
    specs: list[_VoteSpec] = []
    for position, prompt_version in enumerate(prompt_versions):
        for sample_index in range(samples_per_prompt):
            specs.append(
                _VoteSpec(
                    position=position,
                    prompt_version=prompt_version,
                    sample_index=sample_index,
                    cache_suffix=f"ens{position}s{sample_index}",
                )
            )
    return specs


def _ensemble_vote(
    mode: ClassifyMode,
    candidates: Sequence[int] | Sequence[str],
    votes: Sequence[ClassifySheetResult],
) -> tuple[tuple[int, ...] | tuple[str, ...], float]:
    n = len(votes)
    counts: dict[str, int] = {str(c): 0 for c in candidates}
    for vote in votes:
        for ref in vote.final_refs:
            key = str(ref)
            if key in counts:
                counts[key] += 1
    final_str = [c for c, count in counts.items() if 2 * count >= n] if n else []
    unanimous = sum(1 for count in counts.values() if count == 0 or count == n)
    agreement_rate = unanimous / len(counts) if counts else 1.0
    majority: tuple[int, ...] | tuple[str, ...] = (
        tuple(sorted(int(c) for c in final_str)) if mode == "row" else tuple(sorted(final_str))
    )
    return majority, agreement_rate


async def classify_sheet_diverse(
    client: AsyncOpenAI,
    settings: Settings,
    *,
    task: str,
    sheet: str,
    variant_name: str,
    prompt_versions: Sequence[str],
    mode: ClassifyMode,
    context: str,
    candidates: Sequence[int] | Sequence[str],
    run_dir: Path,
    model: str | None = None,
    reasoning_effort: ReasoningEffort | None = None,
    samples_per_prompt: int = 1,
) -> DiverseEnsembleResult:
    if not prompt_versions:
        raise ValueError("prompt_versions must be non-empty")
    if samples_per_prompt < 1:
        raise ValueError(f"samples_per_prompt must be >= 1, got {samples_per_prompt}")

    specs = _plan_votes(prompt_versions, samples_per_prompt)
    raw = await asyncio.gather(
        *(
            classify_sheet(
                client,
                settings,
                task=task,
                sheet=sheet,
                variant_name=variant_name,
                prompt_version=spec.prompt_version,
                mode=mode,
                context=context,
                candidates=candidates,
                run_dir=run_dir,
                model=model,
                reasoning_effort=reasoning_effort,
                cache_suffix=spec.cache_suffix,
            )
            for spec in specs
        ),
        return_exceptions=True,
    )

    votes: list[ClassifySheetResult] = []
    records: list[PromptVoteRecord] = []
    failures: list[BaseException] = []
    for spec, outcome in zip(specs, raw, strict=True):
        if isinstance(outcome, ClassifySheetResult):
            votes.append(outcome)
            records.append(
                PromptVoteRecord(
                    prompt_version=spec.prompt_version,
                    sample_index=spec.sample_index,
                    final_refs=outcome.final_refs,
                    usage=outcome.usage,
                )
            )
        elif isinstance(outcome, BaseException):
            logger.warning(
                "ensemble vote dropped for sheet %r (prompt=%s sample=%d): %s: %s",
                sheet,
                spec.prompt_version,
                spec.sample_index,
                type(outcome).__name__,
                outcome,
            )
            failures.append(outcome)
        else:  # pragma: no cover - gather only yields results or exceptions
            raise TypeError(f"unexpected gather outcome for sheet {sheet!r}: {outcome!r}")

    if not votes:
        first = failures[0]
        if isinstance(first, ClassifyError):
            raise first
        raise ClassifyError(f"all {len(failures)} ensemble votes failed for sheet {sheet!r}: {first}") from first

    majority_refs, agreement_rate = _ensemble_vote(mode, candidates, votes)
    total_usage = UsageRecord()
    for vote in votes:
        total_usage = total_usage.combined_with(vote.usage)

    return DiverseEnsembleResult(
        sheet=sheet,
        mode=mode,
        prompt_versions=tuple(prompt_versions),
        n_votes=len(votes),
        n_failed_votes=len(failures),
        majority_refs=majority_refs,
        agreement_rate=agreement_rate,
        per_vote=tuple(records),
        usage=total_usage,
    )
