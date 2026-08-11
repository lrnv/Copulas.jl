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
[`copula`](https://cran.r-project.org/package=copula) package. The table uses
three representative models from their shared API: Clayton (lower-tail
Archimedean), Gumbel (upper-tail Archimedean), and Gaussian (elliptical). For
each bivariate model it measures sampling, PDF and CDF evaluation, and fitting
by inversion of Kendall's tau. This is a useful common subset, not an exhaustive
survey of either package. Clayton and Gumbel use ``\theta=2``; Gaussian uses
``\rho=0.5``.

The Julia documentation process measures the Julia operations directly and
invokes one `Rscript` process for their R equivalents. Both languages are warmed
up before five timing samples are collected, with inexpensive operations batched
to improve timer resolution. Package loading, Julia compilation, and R process
startup are excluded. The resulting table is inserted as part of Documenter's
evaluation of this page, so every preview and release page shows numbers measured
by its own workflow run.

```@example julia_vs_r
using Copulas # hide
using Dates # hide
using Distributions # hide
using Markdown # hide
using Random # hide

const BENCHMARK_SAMPLES = 5 # hide
const BENCHMARK_MODELS = [("clayton", "Clayton"), ("gumbel", "Gumbel"), ("gaussian", "Gaussian")] # hide
const BENCHMARK_OPERATIONS = [ # hide
    ("sampling", "Sampling", "10,000 draws"), # hide
    ("pdf", "PDF", "10,000 points"), # hide
    ("cdf", "CDF", "1,000 points"), # hide
    ("fitting", "Fit (inverse Kendall's τ)", "2,000 observations"), # hide
] # hide
const BENCHMARK_ROWS = [ # hide
    (key="$model/$operation", model=label, operation=operation_label, workload=workload, # hide
     batch=operation == "cdf" ? (model == "gaussian" ? 1 : 100) : 10) # hide
    for (model, label) in BENCHMARK_MODELS # hide
    for (operation, operation_label, workload) in BENCHMARK_OPERATIONS # hide
] # hide

function median_seconds(f, batch) # hide
    f() # warm-up: exclude compilation and one-time initialization # hide
    samples = Vector{Float64}(undef, BENCHMARK_SAMPLES) # hide
    for sample in eachindex(samples) # hide
        GC.gc() # hide
        start = time_ns() # hide
        for _ in 1:batch # hide
            f() # hide
        end # hide
        samples[sample] = (time_ns() - start) / 1e9 / batch # hide
    end # hide
    sort!(samples) # hide
    return samples[cld(length(samples), 2)] # hide
end # hide

function julia_benchmarks() # hide
    rng = Xoshiro(23) # hide
    specs = [ # hide
        (id="clayton", model=ClaytonCopula(2, 2.0), type=ClaytonCopula), # hide
        (id="gumbel", model=GumbelCopula(2, 2.0), type=GumbelCopula), # hide
        (id="gaussian", model=GaussianCopula(2, 0.5), type=GaussianCopula), # hide
    ] # hide
    points = clamp.(rand(rng, 2, 10_000), 1e-6, 1 - 1e-6) # hide
    cdf_points = @view points[:, 1:1_000] # hide
    functions = Dict{String,Function}() # hide
    for spec in specs # hide
        fit_data = rand(rng, spec.model, 2_000) # hide
        functions["$(spec.id)/sampling"] = let model=spec.model; () -> rand(rng, model, 10_000); end # hide
        functions["$(spec.id)/pdf"] = let model=spec.model; () -> pdf(model, points); end # hide
        functions["$(spec.id)/cdf"] = let model=spec.model; () -> cdf(model, cdf_points); end # hide
        functions["$(spec.id)/fitting"] = let type=spec.type, data=fit_data; () -> fit(type, data; method=:itau); end # hide
    end # hide
    return Dict(row.key => median_seconds(functions[row.key], row.batch) for row in BENCHMARK_ROWS) # hide
end # hide

const R_BENCHMARK_SCRIPT = raw""" # hide
suppressPackageStartupMessages(library(copula)) # hide

samples <- 5L # hide
measure <- function(f, batch) { # hide
    invisible(f()) # hide
    timings <- replicate(samples, { # hide
        invisible(gc()) # hide
        elapsed <- system.time(for (i in seq_len(batch)) invisible(f()))[["elapsed"]] # hide
        elapsed / batch # hide
    }) # hide
    median(timings) # hide
} # hide

set.seed(23) # hide
models <- list( # hide
    clayton = list(model = claytonCopula(2, dim = 2), fit_model = claytonCopula(dim = 2)), # hide
    gumbel = list(model = gumbelCopula(2, dim = 2), fit_model = gumbelCopula(dim = 2)), # hide
    gaussian = list(model = normalCopula(0.5, dim = 2), fit_model = normalCopula(dim = 2)) # hide
) # hide
points <- matrix(runif(2 * 10000, min = 1e-6, max = 1 - 1e-6), ncol = 2) # hide
cdf_points <- points[seq_len(1000), , drop = FALSE] # hide
results <- c() # hide
for (name in names(models)) { # hide
    spec <- models[[name]] # hide
    fit_data <- rCopula(2000, spec$model) # hide
    results[[paste0(name, "/sampling")]] <- measure(function() rCopula(10000, spec$model), 10L) # hide
    results[[paste0(name, "/pdf")]] <- measure(function() dCopula(points, spec$model), 10L) # hide
    cdf_batch <- if (name == "gaussian") 1L else 100L # hide
    results[[paste0(name, "/cdf")]] <- measure(function() pCopula(cdf_points, spec$model), cdf_batch) # hide
    results[[paste0(name, "/fitting")]] <- measure(function() suppressWarnings( # hide
        fitCopula(spec$fit_model, fit_data, method = "itau", estimate.variance = FALSE) # hide
    ), 10L) # hide
} # hide

for (name in names(results)) cat(name, sprintf("%.17g", results[[name]]), sep = "\t", fill = TRUE) # hide
cat("__R_VERSION__", R.version.string, sep = "\t", fill = TRUE) # hide
cat("__COPULA_VERSION__", as.character(packageVersion("copula")), sep = "\t", fill = TRUE) # hide
""" # hide

function r_benchmarks() # hide
    rscript = get(ENV, "RSCRIPT", "Rscript") # hide
    command = Cmd([rscript, "--vanilla", "-"]) # hide
    output = read(pipeline(command; stdin=IOBuffer(R_BENCHMARK_SCRIPT)), String) # hide
    values = Dict{String,String}() # hide
    for line in split(chomp(output), '\n') # hide
        fields = split(chomp(line), '\t'; limit=2) # hide
        length(fields) == 2 || error("Unexpected R benchmark output: $line") # hide
        values[fields[1]] = fields[2] # hide
    end # hide
    timings = Dict(row.key => parse(Float64, values[row.key]) for row in BENCHMARK_ROWS) # hide
    return timings, values["__R_VERSION__"], values["__COPULA_VERSION__"] # hide
end # hide

function format_time(seconds) # hide
    nanoseconds = seconds * 1e9 # hide
    nanoseconds < 1e3 && return "$(round(nanoseconds; digits=1)) ns" # hide
    nanoseconds < 1e6 && return "$(round(nanoseconds / 1e3; digits=2)) μs" # hide
    nanoseconds < 1e9 && return "$(round(nanoseconds / 1e6; digits=2)) ms" # hide
    return "$(round(seconds; digits=2)) s" # hide
end # hide

function benchmark_comparison() # hide
    julia_times = julia_benchmarks() # hide
    r_times, r_version, copula_version = r_benchmarks() # hide

    report = IOBuffer() # hide
    println(report, "### Results from this documentation build") # hide
    println(report) # hide
    commit = get(ENV, "GITHUB_SHA", "local") # hide
    commit_label = if commit == "local" # hide
        "local working tree" # hide
    else # hide
        server = get(ENV, "GITHUB_SERVER_URL", "https://github.com") # hide
        repository = get(ENV, "GITHUB_REPOSITORY", "lrnv/Copulas.jl") # hide
        "[`$(first(commit, min(8, length(commit))))`]($server/$repository/commit/$commit)" # hide
    end # hide
    generated = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS") * "Z" # hide
    println(report, "Generated at `$generated` from $commit_label.") # hide
    println(report) # hide
    println(report, "| Model | Operation | Workload | Julia median | R median | R / Julia |") # hide
    println(report, "|---|---|---:|---:|---:|---:|") # hide
    for row in BENCHMARK_ROWS # hide
        ratio = r_times[row.key] / julia_times[row.key] # hide
        println(report, "| $(row.model) | $(row.operation) | $(row.workload) | $(format_time(julia_times[row.key])) | $(format_time(r_times[row.key])) | $(round(ratio; digits=2))× |") # hide
    end # hide
    println(report) # hide
    println(report, "Five timing samples; median steady-state time per evaluation. Cheap operations are batched to improve timer resolution. Runner: `$(Sys.KERNEL)` / `$(Sys.ARCH)`. Julia $(VERSION) with Copulas.jl $(pkgversion(Copulas)); $r_version with copula $copula_version.") # hide

    return Markdown.parse(String(take!(report))) # hide
end # hide
get(ENV, "COPULAS_DOCS_BENCHMARKS", "false") == "true" ? # hide
    benchmark_comparison() : # hide
    Markdown.parse("Results are generated when this page is built in CI.") # hide
```

!!! note "Interpreting results"
    These are lightweight, indicative measurements from shared GitHub-hosted
    runners, not a controlled benchmarking study. Small differences should not
    be interpreted as universal performance claims.
