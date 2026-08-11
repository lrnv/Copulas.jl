using Copulas
using Dates
using Distributions
using Printf
using Random

const BENCHMARK_SAMPLES = 5
const BENCHMARK_BATCH = 10
const BENCHMARK_ROWS = [
    ("sampling/clayton_d5", "Clayton sampling (`d=5`, `n=10,000`)"),
    ("sampling/gaussian_d10", "Gaussian sampling (`d=10`, `n=10,000`)"),
    ("density/gumbel_d5", "Gumbel log-density (`d=5`, `n=10,000`)"),
    ("data/pseudos_5x10000", "Pseudo-observations (`d=5`, `n=10,000`)"),
    ("fitting/gumbel_itau", "Gumbel inverse-Kendall fit (`n=2,000`)"),
]

function median_seconds(f)
    f() # warm-up: exclude compilation and one-time initialization
    samples = Vector{Float64}(undef, BENCHMARK_SAMPLES)
    for sample in eachindex(samples)
        GC.gc()
        start = time_ns()
        for _ in 1:BENCHMARK_BATCH
            f()
        end
        samples[sample] = (time_ns() - start) / 1e9 / BENCHMARK_BATCH
    end
    sort!(samples)
    return samples[cld(length(samples), 2)]
end

function julia_benchmarks()
    rng = Xoshiro(23)
    clayton = ClaytonCopula(5, 2.0)
    gaussian = GaussianCopula(10, 0.35)
    gumbel = GumbelCopula(5, 2.0)
    points = clamp.(rand(rng, 5, 10_000), 1e-6, 1 - 1e-6)
    raw_data = randn(rng, 5, 10_000)
    fit_data = rand(rng, GumbelCopula(2, 2.0), 2_000)

    functions = Dict(
        "sampling/clayton_d5" => () -> rand(rng, clayton, 10_000),
        "sampling/gaussian_d10" => () -> rand(rng, gaussian, 10_000),
        "density/gumbel_d5" => () -> logpdf(gumbel, points),
        "data/pseudos_5x10000" => () -> pseudos(raw_data),
        "fitting/gumbel_itau" => () -> fit(GumbelCopula, fit_data; method=:itau),
    )
    return Dict(name => median_seconds(functions[name]) for (name, _) in BENCHMARK_ROWS)
end

const R_BENCHMARK_SCRIPT = raw"""
suppressPackageStartupMessages(library(copula))

samples <- 5L
batch <- 10L
measure <- function(f) {
    invisible(f())
    timings <- replicate(samples, {
        invisible(gc())
        elapsed <- system.time(for (i in seq_len(batch)) invisible(f()))[["elapsed"]]
        elapsed / batch
    })
    median(timings)
}

set.seed(23)
clayton <- claytonCopula(2, dim = 5)
gaussian <- normalCopula(0.35, dim = 10, dispstr = "ex")
gumbel <- gumbelCopula(2, dim = 5)
points <- matrix(runif(5 * 10000, min = 1e-6, max = 1 - 1e-6), ncol = 5)
raw_data <- matrix(rnorm(5 * 10000), ncol = 5)
fit_data <- rCopula(2000, gumbelCopula(2, dim = 2))

results <- c(
    "sampling/clayton_d5" = measure(function() rCopula(10000, clayton)),
    "sampling/gaussian_d10" = measure(function() rCopula(10000, gaussian)),
    "density/gumbel_d5" = measure(function() dCopula(points, gumbel, log = TRUE)),
    "data/pseudos_5x10000" = measure(function() pobs(raw_data)),
    "fitting/gumbel_itau" = measure(function() suppressWarnings(
        fitCopula(gumbelCopula(dim = 2), fit_data, method = "itau", estimate.variance = FALSE)
    ))
)

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
    timings = Dict(name => parse(Float64, values[name]) for (name, _) in BENCHMARK_ROWS)
    return timings, values["__R_VERSION__"], values["__COPULA_VERSION__"]
end

function format_time(seconds)
    nanoseconds = seconds * 1e9
    nanoseconds < 1e3 && return @sprintf("%.1f ns", nanoseconds)
    nanoseconds < 1e6 && return @sprintf("%.2f μs", nanoseconds / 1e3)
    nanoseconds < 1e9 && return @sprintf("%.2f ms", nanoseconds / 1e6)
    return @sprintf("%.2f s", seconds)
end

function render_comparison!()
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
    generated = Dates.format(now(UTC), dateformat"yyyy-mm-ddTHH:MM:SS") * "Z"
    println(report, "Generated at `$generated` from $commit_label.")
    println(report)
    println(report, "| Target | Julia median | R median | R / Julia |")
    println(report, "|---|---:|---:|---:|")
    for (name, label) in BENCHMARK_ROWS
        ratio = r_times[name] / julia_times[name]
        println(report, "| $label | $(format_time(julia_times[name])) | $(format_time(r_times[name])) | $(@sprintf("%.2f×", ratio)) |")
    end
    println(report)
    println(report, "Five samples of ten evaluations; median steady-state time per evaluation. Runner: `$(Sys.KERNEL)` / `$(Sys.ARCH)`. Julia $(VERSION) with Copulas.jl $(pkgversion(Copulas)); $r_version with copula $copula_version.")

    begin_marker = "<!-- BEGIN JULIA_VS_R_RESULTS -->"
    end_marker = "<!-- END JULIA_VS_R_RESULTS -->"
    path = get(ENV, "COPULAS_BENCHMARK_DOCS_PATH", joinpath(@__DIR__, "src", "manual", "benchmarks.md"))
    docs = read(path, String)
    length(findall(begin_marker, docs)) == 1 || error("Expected one benchmark results block in $path")
    replacement = string(begin_marker, "\n\n", strip(String(take!(report))), "\n\n", end_marker)
    pattern = Regex(string("(?s)", begin_marker, ".*?", end_marker))
    write(path, replace(docs, pattern => replacement; count=1))
end

render_comparison!()
