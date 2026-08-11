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

The documentation build compares Copulas.jl with R's
[`copula`](https://cran.r-project.org/package=copula) package. The initial set
covers a few representative operations rather than every family and dimension:

- Clayton and Gaussian sampling;
- Gumbel log-density evaluation;
- pseudo-observations;
- Gumbel inverse-Kendall fitting.

The Julia documentation process measures the Julia operations directly and
invokes one `Rscript` process for their R equivalents. Both languages are warmed
up before five batched timing samples are collected. Package loading, Julia
compilation, and R process startup are excluded. The resulting table is inserted
immediately before Documenter builds the site, so every preview and release page
shows numbers measured by its own workflow run.

<!-- BEGIN JULIA_VS_R_RESULTS -->

### Results from this documentation build

The documentation workflow replaces this text with the timing results that
it measured immediately before building the site. Local documentation builds
keep this placeholder unless `COPULAS_DOCS_BENCHMARKS=true` is set.

<!-- END JULIA_VS_R_RESULTS -->

!!! note "Interpreting results"
    These are lightweight, indicative measurements from shared GitHub-hosted
    runners, not a controlled benchmarking study. Small differences should not
    be interpreted as universal performance claims.
