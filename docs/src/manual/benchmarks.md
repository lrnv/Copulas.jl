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
Every documentation build runs the comparison first and embeds that run's table
below. Pull-request previews use the reduced smoke workload; default-branch and
release-tag documentation use the full workload with three alternating process
rounds. The raw JSON and rendered table are retained as artifacts of the same
workflow. The manually dispatched **Julia vs R benchmarks** workflow remains
available for diagnostic runs that do not publish documentation.

<!-- BEGIN JULIA_VS_R_RESULTS -->

### Results from this documentation build

The documentation workflow replaces this text with the validated results that
it measured immediately before building the site. A local documentation build
keeps this placeholder unless the result-injection script is run first.

<!-- END JULIA_VS_R_RESULTS -->

!!! note "Interpreting results"
    GitHub-hosted runners are useful for reproducibility but are shared
    infrastructure. Their ratios are indicative, not universal claims about all
    hardware. Allocation figures are retained separately but are not compared
    because Julia and R memory instrumentation measures different things.
