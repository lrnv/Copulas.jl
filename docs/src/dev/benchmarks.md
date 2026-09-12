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

The local comparison below contrasts Copulas.jl with R's
[`copula`](https://cran.r-project.org/package=copula) package. The table uses
three representative models from their shared API: Clayton (lower-tail
Archimedean), Gumbel (upper-tail Archimedean), and Gaussian (elliptical), in several different dimensions.

For each model, the table measures sampling, PDF and CDF evaluation, and fitting
by inversion of Kendall's tau. This is a useful common subset, not an exhaustive
survey of either package.

The comparison measures the Julia operations directly and invokes one
`Rscript` process for their R equivalents. Both languages are warmed up before
five timing samples are collected, with inexpensive operations batched to
improve timer resolution. Package loading, Julia compilation, and R process
startup are excluded.

### Local comparison

The following snapshot was generated on 12 September 2026 from Copulas.jl v0.1.43.

| Machine property | Value |
|---|---|
| Computer | Lenovo `20XWCTO1WW` |
| Processor | Intel Core i7-1165G7 @ 2.80 GHz, 4 physical cores / 8 logical processors |
| Memory | 32 GiB LPDDR4x-4267 |
| Operating system | Windows 10 Professional 64-bit, build 19045 |
| Julia | 1.13.0, Copulas.jl 0.1.42 |
| R | 4.2.1 (UCRT), `copula` 1.1-7 |

| Model | Dimension | Parameter | Operation | Workload | Julia median | R median | R / Julia |
|---|---:|---:|---|---:|---:|---:|---:|
| Clayton | 2 | θ=2 | Sampling | 10,000 draws | 1.90 ms | 6.0 ms | 3.16× |
| Clayton | 2 | θ=2 | PDF | 10,000 points | 849.18 μs | 78.0 ms | 91.85× |
| Clayton | 2 | θ=2 | CDF | 1,000 points | 105.69 μs | 500.0 μs | 4.73× |
| Clayton | 2 | θ=2 | Fit (inverse Kendall's τ) | 2,000 observations | 1.45 ms | 2.0 ms | 1.38× |
| Gumbel | 2 | θ=2 | Sampling | 10,000 draws | 16.58 ms | 10.0 ms | 0.60× |
| Gumbel | 2 | θ=2 | PDF | 10,000 points | 6.40 ms | 149.0 ms | 23.29× |
| Gumbel | 2 | θ=2 | CDF | 1,000 points | 550.05 μs | 600.0 μs | 1.09× |
| Gumbel | 2 | θ=2 | Fit (inverse Kendall's τ) | 2,000 observations | 713.30 μs | 1.0 ms | 1.40× |
| Gaussian | 2 | ρ=0.5 | Sampling | 10,000 draws | 896.07 μs | 6.0 ms | 6.70× |
| Gaussian | 2 | ρ=0.5 | PDF | 10,000 points | 9.36 ms | 43.0 ms | 4.59× |
| Gaussian | 2 | ρ=0.5 | CDF | 1,000 points | 329.47 ms | 270.0 ms | 0.82× |
| Gaussian | 2 | ρ=0.5 | Fit (inverse Kendall's τ) | 2,000 observations | 588.60 μs | 3.0 ms | 5.10× |
| Gumbel | 5 | θ=2 | Sampling | 10,000 draws | 7.42 ms | 16.0 ms | 2.15× |
| Gumbel | 5 | θ=2 | PDF | 10,000 points | 29.64 ms | 141.0 ms | 4.76× |
| Gaussian | 10 | ρ=0.35 | Sampling | 10,000 draws | 4.83 ms | 30.0 ms | 6.21× |
| Gaussian | 10 | ρ=0.35 | PDF | 10,000 points | 32.82 ms | 61.0 ms | 1.86× |

Five timing samples are summarized by their median steady-state time per
evaluation. The ratio is R time divided by Julia time, so values above one
favor Julia. These figures characterize this machine and software environment;
they are not universal performance guarantees.

::: details Reproduce the comparison locally

Install R's `copula` package, set `RSCRIPT` when `Rscript` is not on `PATH`, and
run the launcher from the repository root:

```powershell
$env:RSCRIPT = "C:\Program Files\R\R-4.2.1\bin\Rscript.exe"
julia --project=docs benchmark/julia_vs_r.jl
```

On Unix-like systems where `Rscript` is on `PATH`, only the second command is
needed. The launcher evaluates exactly the source below; edit the constants if
you want different workloads or more timing samples.

<!-- benchmark-source-start -->
```julia
using Copulas
using Dates
using Distributions
using Markdown
using Random

const BENCHMARK_SAMPLES = 5
const BENCHMARK_MODELS = [
    ("clayton", "Clayton", "θ=2"),
    ("gumbel", "Gumbel", "θ=2"),
    ("gaussian", "Gaussian", "ρ=0.5"),
]
const BENCHMARK_OPERATIONS = [
    ("sampling", "Sampling", "10,000 draws"),
    ("pdf", "PDF", "10,000 points"),
    ("cdf", "CDF", "1,000 points"),
    ("fitting", "Fit (inverse Kendall's τ)", "2,000 observations"),
]
const BENCHMARK_ROWS = vcat(
    [(key="$model/$operation", model=label, dimension=2, parameter=parameter,
      operation=operation_label, workload=workload,
      batch=operation == "cdf" ? (model == "gaussian" ? 1 : 100) : 10)
     for (model, label, parameter) in BENCHMARK_MODELS
     for (operation, operation_label, workload) in BENCHMARK_OPERATIONS],
    [
        (key="gumbel_d5/sampling", model="Gumbel", dimension=5, parameter="θ=2", operation="Sampling", workload="10,000 draws", batch=10),
        (key="gumbel_d5/pdf", model="Gumbel", dimension=5, parameter="θ=2", operation="PDF", workload="10,000 points", batch=10),
        (key="gaussian_d10/sampling", model="Gaussian", dimension=10, parameter="ρ=0.35", operation="Sampling", workload="10,000 draws", batch=10),
        (key="gaussian_d10/pdf", model="Gaussian", dimension=10, parameter="ρ=0.35", operation="PDF", workload="10,000 points", batch=10),
    ],
)

function median_seconds(f, batch)
    f() # warm-up: exclude compilation and one-time initialization
    samples = Vector{Float64}(undef, BENCHMARK_SAMPLES)
    for sample in eachindex(samples)
        GC.gc()
        start = time_ns()
        for _ in 1:batch
            f()
        end
        samples[sample] = (time_ns() - start) / 1e9 / batch
    end
    sort!(samples)
    return samples[cld(length(samples), 2)]
end

function julia_benchmarks()
    rng = Xoshiro(23)
    specs = [
        (id="clayton", model=ClaytonCopula{2}(2.0), type=ClaytonCopula),
        (id="gumbel", model=GumbelCopula{2}(2.0), type=GumbelCopula),
        (id="gaussian", model=GaussianCopula{2}(0.5), type=GaussianCopula),
    ]
    points = clamp.(rand(rng, 2, 10_000), 1e-6, 1 - 1e-6)
    cdf_points = @view points[:, 1:1_000]
    functions = Dict{String,Function}()
    for spec in specs
        fit_data = rand(rng, spec.model, 2_000)
        functions["$(spec.id)/sampling"] = let model=spec.model; () -> rand(rng, model, 10_000); end
        functions["$(spec.id)/pdf"] = let model=spec.model; () -> pdf(model, points); end
        functions["$(spec.id)/cdf"] = let model=spec.model; () -> cdf(model, cdf_points); end
        functions["$(spec.id)/fitting"] = let type=spec.type, data=fit_data; () -> fit(type, data; method=:itau); end
    end
    for spec in [(id="gumbel_d5", model=GumbelCopula{5}(2.0)),
                 (id="gaussian_d10", model=GaussianCopula{10}(0.35))]
        multivariate_points = clamp.(rand(rng, length(spec.model), 10_000), 1e-6, 1 - 1e-6)
        functions["$(spec.id)/sampling"] = let model=spec.model; () -> rand(rng, model, 10_000); end
        functions["$(spec.id)/pdf"] = let model=spec.model, data=multivariate_points; () -> pdf(model, data); end
    end
    return Dict(row.key => median_seconds(functions[row.key], row.batch) for row in BENCHMARK_ROWS)
end

const R_BENCHMARK_SCRIPT = raw"""
suppressPackageStartupMessages(library(copula))

samples <- 5L
measure <- function(f, batch) {
    invisible(f())
    timings <- replicate(samples, {
        invisible(gc())
        elapsed <- system.time(for (i in seq_len(batch)) invisible(f()))[["elapsed"]]
        elapsed / batch
    })
    median(timings)
}

set.seed(23)
models <- list(
    clayton = list(model = claytonCopula(2, dim = 2), fit_model = claytonCopula(dim = 2)),
    gumbel = list(model = gumbelCopula(2, dim = 2), fit_model = gumbelCopula(dim = 2)),
    gaussian = list(model = normalCopula(0.5, dim = 2), fit_model = normalCopula(dim = 2))
)
points <- matrix(runif(2 * 10000, min = 1e-6, max = 1 - 1e-6), ncol = 2)
cdf_points <- points[seq_len(1000), , drop = FALSE]
results <- c()
for (name in names(models)) {
    spec <- models[[name]]
    fit_data <- rCopula(2000, spec$model)
    results[[paste0(name, "/sampling")]] <- measure(function() rCopula(10000, spec$model), 10L)
    results[[paste0(name, "/pdf")]] <- measure(function() dCopula(points, spec$model), 10L)
    cdf_batch <- if (name == "gaussian") 1L else 100L
    results[[paste0(name, "/cdf")]] <- measure(function() pCopula(cdf_points, spec$model), cdf_batch)
    results[[paste0(name, "/fitting")]] <- measure(function() suppressWarnings(
        fitCopula(spec$fit_model, fit_data, method = "itau", estimate.variance = FALSE)
    ), 10L)
}
multivariate_models <- list(
    gumbel_d5 = list(model = gumbelCopula(2, dim = 5), dimension = 5L),
    gaussian_d10 = list(model = normalCopula(0.35, dim = 10, dispstr = "ex"), dimension = 10L)
)
for (name in names(multivariate_models)) {
    spec <- multivariate_models[[name]]
    multivariate_points <- matrix(
        runif(spec$dimension * 10000, min = 1e-6, max = 1 - 1e-6),
        ncol = spec$dimension
    )
    results[[paste0(name, "/sampling")]] <- measure(function() rCopula(10000, spec$model), 10L)
    results[[paste0(name, "/pdf")]] <- measure(function() dCopula(multivariate_points, spec$model), 10L)
}

for (name in names(results)) cat(name, sprintf("%.17g", results[[name]]), sep = "\t", fill = TRUE)
cat("__R_VERSION__", R.version.string, sep = "\t", fill = TRUE)
cat("__COPULA_VERSION__", as.character(packageVersion("copula")), sep = "\t", fill = TRUE)
"""

function r_benchmarks()
    rscript = get(ENV, "RSCRIPT", "Rscript")
    command = Cmd([rscript, "--vanilla", "-"])
    output = read(pipeline(command; stdin=IOBuffer(R_BENCHMARK_SCRIPT)), String)
    values = Dict{String,String}()
    for line in split(chomp(output), '\n')
        fields = split(chomp(line), '\t'; limit=2)
        length(fields) == 2 || error("Unexpected R benchmark output: $line")
        values[fields[1]] = fields[2]
    end
    timings = Dict(row.key => parse(Float64, values[row.key]) for row in BENCHMARK_ROWS)
    return timings, values["__R_VERSION__"], values["__COPULA_VERSION__"]
end

function format_time(seconds)
    nanoseconds = seconds * 1e9
    nanoseconds < 1e3 && return "$(round(nanoseconds; digits=1)) ns"
    nanoseconds < 1e6 && return "$(round(nanoseconds / 1e3; digits=2)) μs"
    nanoseconds < 1e9 && return "$(round(nanoseconds / 1e6; digits=2)) ms"
    return "$(round(seconds; digits=2)) s"
end

function benchmark_comparison()
    julia_times = julia_benchmarks()
    r_times, r_version, copula_version = r_benchmarks()

    report = IOBuffer()
    println(report, "### Results from this documentation build")
    println(report)
    commit = get(ENV, "GITHUB_SHA", "local")
    commit_label = if commit == "local"
        "local working tree"
    else
        server = get(ENV, "GITHUB_SERVER_URL", "https://github.com")
        repository = get(ENV, "GITHUB_REPOSITORY", "lrnv/Copulas.jl")
        "[`$(first(commit, min(8, length(commit))))`]($server/$repository/commit/$commit)"
    end
    generated = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS") * "Z"
    println(report, "Generated at `$generated` from $commit_label.")
    println(report)
    println(report, "| Model | Dimension | Parameter | Operation | Workload | Julia median | R median | R / Julia |")
    println(report, "|---|---:|---:|---|---:|---:|---:|---:|")
    for row in BENCHMARK_ROWS
        ratio = r_times[row.key] / julia_times[row.key]
        println(report, "| $(row.model) | $(row.dimension) | $(row.parameter) | $(row.operation) | $(row.workload) | $(format_time(julia_times[row.key])) | $(format_time(r_times[row.key])) | $(round(ratio; digits=2))× |")
    end
    println(report)
    println(report, "Five timing samples; median steady-state time per evaluation. Cheap operations are batched to improve timer resolution. Runner: `$(Sys.KERNEL)` / `$(Sys.ARCH)`. Julia $(VERSION) with Copulas.jl $(pkgversion(Copulas)); $r_version with copula $copula_version.")

    return Markdown.parse(String(take!(report)))
end
benchmark_comparison()
```
<!-- benchmark-source-end -->

:::

!!! note "Interpreting results"
    These are lightweight, indicative measurements from one local machine, not
    a controlled cross-platform benchmarking study. Small differences should
    not be interpreted as universal performance claims. Power management,
    thermal state, background activity and package versions can all affect the
    result.
