# Formuloom

**Workbook change classification with structural features, weak supervision,
multi-method consensus, adjudication, and sheet-level routing.**

Formuloom takes two workbooks, a supplied universe of changed cells, and a short
description of the workbook's purpose. It partitions those candidates into
**final outputs** and **intermediates**. This is not a formula calculator or a
general workbook differ: it consumes `raw_diff.json` and classifies its cells.

This repository contains the full Python pipeline, not just a browser exercise.
The independently written public examples are small synthetic workbooks; no
historical workbook corpus, reference labels, provider responses, or run records
are included. Public prompt wording and examples have been independently
rewritten. Historical performance does not transfer to this release.

The separate browser edition is proposed in [demo PR #55](https://github.com/Sahil170595/Banterblogs/pull/55).
It is pending merge and is **not a claimed live deployment**. Its reduced browser
logic is not this Python pipeline. The portfolio collection is [/work](https://chimeraforge.vercel.app/work).

## Run Without a Provider

Python 3.11+ is declared; release checks used Python 3.13.1. From the repository
root, with dependencies already available:

```sh
python -B -m formuloom.reproduce --out output/first-run
python -B -m formuloom predict examples/synthetic --variant V8 --run-dir output/v8
python -B -m formuloom score examples/synthetic --pred output/v8/generated.json --mode both
python -B -m formuloom eval examples --variant V8 --offline-only --out output/eval.json
```

`reproduce` refuses existing output directories. It generates a **fresh XLSX
pair**, runs V0, V8, and the explicitly substituted V15 mechanical composition,
then loads synthetic reference labels for scoring. It writes predictions,
per-cell error reports, row features, formula-graph warnings, LF votes and
reliabilities, EM convergence diagnostics, routing decisions, and a versioned
`report.json`. It performs no HTTP requests or provider calls.

For a new environment, the usual installation is `python -m pip install -e
".[dev]"`. No dependency installation or wheel build was performed for the
release checks: the existing Python environment was reused. The SDK is currently
a required import dependency even when using offline variants; a key is not.
See [reproduction details](docs/REPRODUCTION.md) for the tested versions and checks.

## Underlying System

The retained implementation treats output identification as a semantic decision,
not an Excel syntax property. An aggregated row may still feed another schedule;
a direct reference may be a legitimate deliverable on a summary sheet. A
dependency sink is evidence, not a definition of a final output.

The system has five main boundaries:

1. **Bundle and schema validation.** `TaskBundle` admits only the workbook pair,
   supplied candidate diff, and instructions during prediction. `subset.json`
   is accessible only in score mode. Typed schemas validate references and diff
   structure. Assembly preserves the deduplicated candidate universe and rejects
   predicted references outside it.
2. **Workbook extraction and structure.** Separate formula and cached-value
   loads preserve both views. Extraction includes formatting, defined names,
   charts, print areas, tables, and merged regions. Formula tokenization builds
   precedent/dependent graphs, including cross-sheet edges and bounded recursive
   named-range resolution. Row features combine labels, formula sketches,
   aggregation, dependency counts, presentation, and a financial-output lexicon.
3. **Context encoding.** The encoder chooses row or cell granularity using
   structural signals, compresses repeated formulas and adjacent references, and
   can split context at sections. More than 300 candidates forces row grouping;
   section splitting is not a hard model-context limit. An optional task-profile
   call summarizes workbook purpose before sheet classification.
4. **Inference and composition.** Recipes combine deterministic rules, EM weak
   labels, typed model responses, repeated votes, prompt diversity, intersection,
   adjudication, cascades, and structural routing. Provider requests have bounded
   concurrency, retry policy, response validation, and atomic artifact writes.
5. **Evaluation and diagnosis.** Both scoring modes expose TP/FP/FN, task and
   sheet breakdowns, micro/macro aggregation, and false-positive/negative cell
   reports. Auxiliary modules implement posterior/cost diagnostics, bootstrap
   estimates, conformal threshold helpers, and controlled workbook perturbations.

### V8: Actual Weak-Label EM

Seven labeling functions vote `final`, `intermediate`, or abstain: formula sinks,
aggregation, output vocabulary, input-colored cells, unlabeled constants,
bold/bordered output rows, and single-reference pass-throughs. The retained
Dawid-Skene-style model performs log-space expectation steps and smoothed
maximization steps for class prevalence and each LF's two class-conditional
reliabilities. It is implemented locally, not a call to a hosted model or an
imported labeling package.

Accuracy initialization is 0.7, LF smoothing strength is 5, class-prior pseudo
strength is 2, and fitting allows at most 100 iterations with a 1e-4 convergence
criterion. A majority-alignment guard handles class-label reversal. Diagnostics
record conflicts, all-abstain rows, convergence, and whether a flip occurred.

The public `weak.py` adapter is new wiring around this retained algorithm: it
pools candidate rows across one workbook, fixes prevalence at 0.30, and thresholds
the inferred probabilities at 0.5. These are illustrative defaults, not
independently calibrated prevalence or probabilities. Call `predict_v8` with a
different prior/threshold to study sensitivity. Correlated labeling functions
violate the conditional-independence assumption; a tiny workbook cannot identify
reliability well. Label-free fitting is not proof of semantic correctness.

### V15: Real Multi-Method Routing

V15 constructs **both** V11 and V9 predictions, then chooses an arm per sheet.
V11 uses a precision-oriented prompt, three votes, structural guards, and
selective adjudication. V9 uses three recall-oriented votes without a task-profile
call. The router considers candidate density/count, input coloring, dependency
direction, row widths, and formula diversity; it can empty feeder/reference
sheets or prune particular funding and percentage rows.

Examples of actual retained decision boundaries include a dense standalone
sheet with at least 1,000 candidates and density at least 0.25, and a sparse
cross-sheet output route below 0.04 density with at least 250 cross-sheet
precedents. These are engineering heuristics, **not validated universal
thresholds**. A small feeder-sheet rule uses input coloring strictly below 0.5;
the public fixture at exactly 0.5 exercises that boundary without being emptied.

The implementation executes the two arms sequentially before routing. It is
therefore not a lazy dispatcher that saves the unselected arm's API cost.
Graph incompleteness or a slightly changed layout can switch decisions sharply.

The no-provider reproduction uses this **same router and assembly**, but replaces
V11 with V0 rules and V9 with the union of V0 and V8. It is named
`V15-mechanical`, never reported as a V15 provider benchmark. The small XLSX
example selects the default arm on all sheets; unit tests independently exercise
high-recall routing, emptying, pruning, and their boundary counterexamples.

### Consensus, Adjudication, and Other Recipes

| Recipe | Implemented mechanics |
| --- | --- |
| V0 | Deterministic aggregation/output-lexicon rules |
| V1-V5 | Compressed/features/section/full-dump context variations; selective or repeated voting |
| V6 | Cell-oriented CLI recipe; separate built-in-tooling module, caveat below |
| V7, V9 | Three recall-oriented samples, differing task-profile settings |
| V11 | Precision guards, three votes, selectively enabled adjudication |
| V12 | Three different prompt policies; row ties resolve toward final |
| V13 | Intersection of V9 and V11 |
| V14 | Nine samples across three prompts; vote bands plus ambiguous-row adjudication |
| V15 | Structural selection and pruning of V11/V9 predictions |
| V16 | Typed scout assessment selecting V9/V11/V13 per sheet |

Single-prompt self-consistency uses strict majority, so an even-vote tie is
intermediate; this differs deliberately from the prompt-diverse ensemble's tie
policy. V14 accepts at least seven of nine votes, rejects at most two, and judges
the ambiguous band; its keep logic and degraded-majority path are separately
tested. Adjudication reasons over explicit candidates, can add rows when allowed,
and filters unknown outputs.

The separate `tooling_v6.py` contains workbook upload and provider container-tool
support. **The generic V6 CLI recipe does not dispatch through that module**;
setting its registry flag does not establish code-interpreter execution. The
module is retained as an API, not a verified end-to-end service. External file
and container lifecycle is the caller's responsibility.

## Fresh Synthetic Findings

The three-sheet example has 18 candidate cells and 10 reference final cells.
Its explicit reference policy treats Operations revenue/contribution rows and
all Summary candidates as deliverables, but Assumptions as inputs. References
are authored for that policy, not independently labeled benchmark evidence.

| Current run | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V0 | 4 | 0 | 6 | 1.0 | 0.4 | 0.571429 |
| V8 | 4 | 0 | 6 | 1.0 | 0.4 | 0.571429 |
| V15-mechanical | 4 | 0 | 6 | 1.0 | 0.4 | 0.571429 |

Strict and annotated modes agree here because the reference labels exhaust the
candidate universe. The identical predictions are **not** evidence that these
methods are equivalent. The example exposes their shared limitations: valid
summary deliverables can be pass-throughs, and reference roles need not follow
aggregation keywords. Six misses are preserved in the generated error reports.
No historical metrics or paid-provider measurements are published here.
The generated [synthetic report](docs/synthetic-report.json) preserves these
counts; `reproduce` recomputes them rather than reading the report as an oracle.

## Failure Cases and Operational Limits

- **Cached values are not recalculation.** openpyxl does not execute formulas.
  The fixture generator writes known arithmetic caches for its own formulas;
  this is not an Excel engine. Missing/stale caches in other workbooks remain a
  limitation. Numeric workbook tolerances are not formula-verification results.
- **Graph parsing is best-effort.** Structured/external/dynamic references and
  incomplete named-range expansion can leave edges unknown. Warnings must be
  inspected; a graph sink can be an extraction failure. Financial vocabulary,
  color, and layout policies can fail on other domains.
- **No arbitrary execution or hosted server.** Workbook inputs are parsed as
  data; formulas are not executed. The library is not a sandbox for malicious
  XML archives, and large workbooks/ranges have no comprehensive resource quota.
- **Trust cache provenance.** Workbook caches use pickle and must be local,
  trusted artifacts. Never load a cache supplied by someone else. Provider cache
  keys do not hash every workbook/model/context parameter; use a fresh run
  directory after input/config changes. `eval --repeat` can reuse cached replies
  and is not independent stochastic replication.
- **Degradation is not a verdict.** `--best-effort` records sheet failures and
  assigns all-intermediate. Adjudication can retain the incoming candidates on a
  judge failure and logs degradation; the generic CLI does not propagate every
  such degraded flag into its failure manifest. Review the run logs/artifacts
  before treating a result as accepted. Defaults are not a universal fail-closed
  evaluation policy.
- **Evaluation semantics matter.** Strict scoring counts every predicted final
  outside reference finals as FP. Annotated scoring counts errors only against
  explicitly opposite reference labels; missing annotations can make it look
  better. Empty-empty sets score perfectly by convention. Conformal helpers need
  a separate labeled calibration set and exchangeability assumptions; no public
  coverage guarantee is asserted for correlated workbook rows.
- **Cost estimates are historical constants.** Model IDs and pricing entries are
  retained configuration, not a promise of current availability/billing. Some
  legacy cost-normalized diagnostics return infinity for zero cost; the focused
  reproduction report uses finite strict JSON instead.

## Optional Provider Execution

```sh
# Explicit opt-in; incurs provider charges and uploads encoded workbook content.
# Set OPENAI_API_KEY in your environment, not in tracked files.
python -B -m formuloom predict examples/synthetic --variant V15 --run-dir output/provider-v15
python -B -m formuloom score examples/synthetic --pred output/provider-v15/generated.json
```

Review the newly authored prompts and typed registry before enabling this path.
Models are selected by each variant in `variants.py`; a general model setting
does not override those pinned constituent IDs. Change the registry deliberately
if needed, and choose a new run directory. Provider parsing/retry/cache behavior
is covered by fake-client unit tests; **no live provider request was made for this
release**, and no new provider-quality claim is made.

## Module Map and Credits

| Modules | Responsibility |
| --- | --- |
| `bundle`, `schema`, `constants`, `assemble` | Validated input contract and exact output partition |
| `workbook`, `features`, `encode` | Formula/value extraction, graph and presentation features, context encoding |
| `labelmodel`, `weak` | Retained weak-label EM and new offline adapter |
| `classify`, `prompts`, `adjudicate`, `ensemble` | Structured provider inference and consensus/judging |
| `compose`, `cascade`, `v15_router`, `scout`, `variants` | Multi-method execution and routing recipes |
| `score`, `metrics_extra`, `perturb` | Evaluation, diagnostic statistics, synthetic robustness transforms |
| `tooling_v6`, `settings`, `cli` | Provider-tool API, runtime settings, predict/score/eval/compare commands |
| `fixtures`, `offline`, `reproduce` | Fresh synthetic XLSX and provider-free inspectable reproduction |

Project code is MIT licensed; dependency licenses and method credits remain
separate. See [LICENSE](LICENSE) and [third-party notes](THIRD_PARTY.md). This
release contains no vendored third-party code or third-party workbook assets.
