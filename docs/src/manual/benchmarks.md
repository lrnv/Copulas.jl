```@meta
CurrentModule = Copulas
```

# Performance benchmarks

Copulas.jl uses two complementary benchmark systems.

## Performance over time

[Tachometer](https://github.com/KristofferC/Tachometer.jl) runs the native Julia
suite on pull requests, compares changes with their merge base, and records
default-branch results. Its dashboard is published alongside this documentation
at [the Copulas.jl benchmark dashboard](https://lrnv.github.io/Copulas.jl/benchmarks/).

## Comparison with R

The cross-language suite compares Copulas.jl with R's
[`copula`](https://cran.r-project.org/package=copula) package. The initial set
covers representative common operations rather than every family and dimension:

- Clayton, Gumbel, and Gaussian sampling;
- Gumbel and Gaussian log-density evaluation;
- pseudo-observations;
- Gumbel inverse-Kendall fitting;
- Gaussian maximum-likelihood fitting.

Julia and R run as independent native processes on the same machine. The suite
excludes process startup, package loading, fixture parsing, model construction,
and Julia compilation. Full runs alternate three Julia and R process rounds and
report the median steady-state time for each language.

Before reporting timings, deterministic outputs are compared numerically on
shared committed fixtures. Sampling uses independent seeded streams and receives
local range and finiteness checks. A failed correctness check invalidates the
entire timing report.

The complete specification, locked environments, raw-result schema, and local
reproduction instructions live in
[`benchmark/cross_language`](https://github.com/lrnv/Copulas.jl/tree/main/benchmark/cross_language).
Full reports are produced on releases and through the manually dispatched
**Julia vs R benchmarks** workflow. The workflow attaches the raw JSON and
rendered table as artifacts and includes the table in its job summary.

!!! note "Interpreting results"
    GitHub-hosted runners are useful for reproducibility but are shared
    infrastructure. Their ratios are indicative, not universal claims about all
    hardware. Allocation figures are retained separately but are not compared
    because Julia and R memory instrumentation measures different things.

No full reference snapshot has been published yet. A snapshot should be added
to `benchmark/cross_language/results/` only after its correctness report and
runner metadata have been reviewed.
