# Julia versus R benchmarks

This suite compares steady-state Copulas.jl operations with equivalent calls
from R's `copula` package. It is independent of the Tachometer suite: Tachometer
tracks Julia regressions, while this directory holds a controlled
cross-language experiment.

## Design

- `cases.toml` is the single source of truth for names, parameters, dimensions,
  workloads, fixtures, and correctness tolerances.
- `data/` contains deterministic CSV fixtures. Rows are observations on disk;
  the Julia runner transposes them before timing.
- `julia/run.jl` and `r/run.R` load data and construct models before timing.
  Package loading, Julia compilation, fixture parsing, and model construction
  are excluded.
- Both runners emit the same JSON schema. `scripts/compare_results.jl` refuses
  to publish ratios unless all cases exist, runner regimes match, and numerical
  validation succeeds.
- Sampling uses independent seeded RNG streams because Julia and R do not share
  an RNG implementation. Sampling output is checked locally for validity;
  deterministic density, ranking, and fitting output is compared numerically.
- Time ratios use per-language median elapsed time. Allocation measurements are
  retained in raw files for diagnostics but are not compared across runtimes.

The full workflow alternates three Julia and R process rounds to reduce ordering
and host-load bias. Pull requests that modify this directory run one reduced
smoke round. Release and manually dispatched runs use the full workloads.

## Cases

| Target | Julia | R |
|---|---|---|
| Clayton, Gumbel, and Gaussian sampling | `rand` | `rCopula` |
| Gumbel and Gaussian log-density | `logpdf` | `dCopula(..., log=TRUE)` |
| Pseudo-observations | `pseudos` | `pobs` |
| Gumbel inverse-Kendall fitting | `fit(..., method=:itau)` | `fitCopula(..., method="itau")` |
| Gaussian maximum-likelihood fitting | `fit(..., method=:mle)` | `fitCopula(..., method="ml")` |

The Gaussian fitting algorithms are equivalent estimators but not identical
implementations: Copulas.jl computes the latent-normal correlation estimate,
whereas R numerically optimizes the copula likelihood. The validation tolerance
in `cases.toml` accounts for their difference on the reduced smoke sample.

## Reproducing a run

Install Julia 1.12.1 and R 4.6.1. Restore the locked environments, then run:

```sh
julia --startup-file=no --project=benchmark/cross_language/julia \
  -e 'using Pkg; Pkg.instantiate()'
Rscript --vanilla -e 'renv::restore(project="benchmark/cross_language/r", prompt=FALSE)'

CROSS_BENCH_MODE=full CROSS_BENCH_ROUNDS=3 \
  bash benchmark/cross_language/scripts/run_suite.sh
```

The rendered table is written to `results/comparison.md`, with raw per-round
records and machine metadata under `results/raw/`. GitHub-hosted runner results
are indicative; strong absolute-performance claims should be repeated on
controlled hardware.

`scripts/generate_data.jl` regenerates the fixtures. Fixture changes must be
reviewed together with all four CSV files because they alter the experiment.
