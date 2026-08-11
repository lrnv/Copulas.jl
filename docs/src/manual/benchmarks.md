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
as part of Documenter's evaluation of this page, so every preview and release
page shows numbers measured by its own workflow run.

```@example julia_vs_r
using Copulas # hide
using Dates # hide
using Distributions # hide
using Markdown # hide
using Random # hide

const BENCHMARK_SAMPLES = 5 # hide
const BENCHMARK_BATCH = 10 # hide
const BENCHMARK_ROWS = [ # hide
    ("sampling/clayton_d5", "Clayton sampling (`d=5`, `n=10,000`)"), # hide
    ("sampling/gaussian_d10", "Gaussian sampling (`d=10`, `n=10,000`)"), # hide
    ("density/gumbel_d5", "Gumbel log-density (`d=5`, `n=10,000`)"), # hide
    ("data/pseudos_5x10000", "Pseudo-observations (`d=5`, `n=10,000`)"), # hide
    ("fitting/gumbel_itau", "Gumbel inverse-Kendall fit (`n=2,000`)"), # hide
] # hide

function median_seconds(f) # hide
    f() # warm-up: exclude compilation and one-time initialization # hide
    samples = Vector{Float64}(undef, BENCHMARK_SAMPLES) # hide
    for sample in eachindex(samples) # hide
        GC.gc() # hide
        start = time_ns() # hide
        for _ in 1:BENCHMARK_BATCH # hide
            f() # hide
        end # hide
        samples[sample] = (time_ns() - start) / 1e9 / BENCHMARK_BATCH # hide
    end # hide
    sort!(samples) # hide
    return samples[cld(length(samples), 2)] # hide
end # hide

function julia_benchmarks() # hide
    rng = Xoshiro(23) # hide
    clayton = ClaytonCopula(5, 2.0) # hide
    gaussian = GaussianCopula(10, 0.35) # hide
    gumbel = GumbelCopula(5, 2.0) # hide
    points = clamp.(rand(rng, 5, 10_000), 1e-6, 1 - 1e-6) # hide
    raw_data = randn(rng, 5, 10_000) # hide
    fit_data = rand(rng, GumbelCopula(2, 2.0), 2_000) # hide

    functions = Dict( # hide
        "sampling/clayton_d5" => () -> rand(rng, clayton, 10_000), # hide
        "sampling/gaussian_d10" => () -> rand(rng, gaussian, 10_000), # hide
        "density/gumbel_d5" => () -> logpdf(gumbel, points), # hide
        "data/pseudos_5x10000" => () -> pseudos(raw_data), # hide
        "fitting/gumbel_itau" => () -> fit(GumbelCopula, fit_data; method=:itau), # hide
    ) # hide
    return Dict(name => median_seconds(functions[name]) for (name, _) in BENCHMARK_ROWS) # hide
end # hide

const R_BENCHMARK_SCRIPT = raw""" # hide
suppressPackageStartupMessages(library(copula)) # hide

samples <- 5L # hide
batch <- 10L # hide
measure <- function(f) { # hide
    invisible(f()) # hide
    timings <- replicate(samples, { # hide
        invisible(gc()) # hide
        elapsed <- system.time(for (i in seq_len(batch)) invisible(f()))[["elapsed"]] # hide
        elapsed / batch # hide
    }) # hide
    median(timings) # hide
} # hide

set.seed(23) # hide
clayton <- claytonCopula(2, dim = 5) # hide
gaussian <- normalCopula(0.35, dim = 10, dispstr = "ex") # hide
gumbel <- gumbelCopula(2, dim = 5) # hide
points <- matrix(runif(5 * 10000, min = 1e-6, max = 1 - 1e-6), ncol = 5) # hide
raw_data <- matrix(rnorm(5 * 10000), ncol = 5) # hide
fit_data <- rCopula(2000, gumbelCopula(2, dim = 2)) # hide

results <- c( # hide
    "sampling/clayton_d5" = measure(function() rCopula(10000, clayton)), # hide
    "sampling/gaussian_d10" = measure(function() rCopula(10000, gaussian)), # hide
    "density/gumbel_d5" = measure(function() dCopula(points, gumbel, log = TRUE)), # hide
    "data/pseudos_5x10000" = measure(function() pobs(raw_data)), # hide
    "fitting/gumbel_itau" = measure(function() suppressWarnings( # hide
        fitCopula(gumbelCopula(dim = 2), fit_data, method = "itau", estimate.variance = FALSE) # hide
    )) # hide
) # hide

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
    timings = Dict(name => parse(Float64, values[name]) for (name, _) in BENCHMARK_ROWS) # hide
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
    println(report, "| Target | Julia median | R median | R / Julia |") # hide
    println(report, "|---|---:|---:|---:|") # hide
    for (name, label) in BENCHMARK_ROWS # hide
        ratio = r_times[name] / julia_times[name] # hide
        println(report, "| $label | $(format_time(julia_times[name])) | $(format_time(r_times[name])) | $(round(ratio; digits=2))× |") # hide
    end # hide
    println(report) # hide
    println(report, "Five samples of ten evaluations; median steady-state time per evaluation. Runner: `$(Sys.KERNEL)` / `$(Sys.ARCH)`. Julia $(VERSION) with Copulas.jl $(pkgversion(Copulas)); $r_version with copula $copula_version.") # hide

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
