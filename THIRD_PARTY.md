# Third-Party Notes and Method Credits

No third-party source code or workbook corpus is vendored in this repository.
The project MIT license applies to project-authored code and independently
authored synthetic fixtures, not to dependencies or publications.

Runtime dependencies are openpyxl (workbook/XML handling), Pydantic and
pydantic-settings (validation/configuration), Typer (CLI), the OpenAI Python SDK
(optional provider execution), and Tenacity (retry policies). Development checks
use pytest, Ruff, Black, and mypy; Hatchling is the declared package build backend.
These are separately distributed packages with their own licenses and notices;
retain those notices when redistributing a bundled environment. No dependency
license is replaced by this project's MIT license.

The label model is a locally implemented binary, abstention-aware EM model in
the tradition of Dawid and Skene's [Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm](https://academic.oup.com/jrsssc/article/28/1/20/6953573)
(1979). Labeling-function aggregation is also
part of the broader weak-supervision/data-programming literature, including
[Snorkel](https://arxiv.org/abs/1711.10160). Formuloom does not vendor or depend on Snorkel and does not claim its
correlation modeling or guarantees.

Self-consistency, prompt-diverse voting, bootstrap resampling, and split-conformal
thresholding are established techniques. The modules implement their own limited
versions; neither their names nor test coverage establishes the assumptions
needed for general statistical guarantees. The workbook domain policies, recipe
composition, and synthetic public examples are project-specific.

Materials whose redistribution rights were not established are not included.
Public prompt policies and demonstrations are independently authored, rather
than copies of externally supplied prose or examples.
