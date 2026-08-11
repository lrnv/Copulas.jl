using BenchmarkTools
using Copulas
using Dates
using DelimitedFiles
using Distributions
using JSON3
using LinearAlgebra
using Random
using Statistics
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SPEC = TOML.parsefile(joinpath(ROOT, "cases.toml"))
const MODE = get(ENV, "CROSS_BENCH_MODE", "full")
const OUTPUT = get(ENV, "CROSS_BENCH_OUTPUT", joinpath(ROOT, "results", "julia.json"))
const SEED = Int(SPEC["suite"]["seed"])
const VALIDATION_POINTS = Int(SPEC["suite"]["validation_points"])
const REPETITIONS = Int(SPEC["suite"][MODE == "smoke" ? "smoke_repetitions" : "full_repetitions"])

MODE in ("smoke", "full") || error("CROSS_BENCH_MODE must be smoke or full")

function model(case)
    family = case["family"]
    d = Int(case["dimension"])
    parameter = Float64(case["parameter"])
    family == "clayton" && return ClaytonCopula(d, parameter)
    family == "gumbel" && return GumbelCopula(d, parameter)
    family == "gaussian" && return GaussianCopula(d, parameter)
    error("Unsupported family: $family")
end

function input_matrix(case)
    path = joinpath(ROOT, case["input"])
    rows = readdlm(path, ',', Float64)
    n = Int(case[MODE == "smoke" ? "smoke_n" : "n"])
    d = Int(case["dimension"])
    size(rows, 1) >= n || error("Not enough rows in $path")
    size(rows, 2) >= d || error("Not enough columns in $path")
    return permutedims(rows[1:n, 1:d])
end

upper_triangle(matrix) = [matrix[i, j] for j in 2:size(matrix, 2) for i in 1:j-1]

function prepare(case)
    operation = case["operation"]
    n = Int(case[MODE == "smoke" ? "smoke_n" : "n"])

    if operation == "sample"
        copula = model(case)
        rng = Xoshiro(SEED)
        f = () -> rand(rng, copula, n)
        check = rand(Xoshiro(SEED + 1), copula, min(n, 1_000))
        valid = all(isfinite, check) && all(0 .<= check .<= 1)
        return f, "stochastic_summary", vec(mean(check; dims=2)), valid
    elseif operation == "logdensity"
        copula = model(case)
        points = input_matrix(case)
        f = () -> logpdf(copula, points)
        values = logpdf(copula, @view points[:, 1:min(VALIDATION_POINTS, size(points, 2))])
        return f, "numeric", collect(values), all(isfinite, values)
    elseif operation == "pseudos"
        data = input_matrix(case)
        f = () -> pseudos(data)
        transformed = pseudos(data)
        values = vec(transformed[:, 1:min(VALIDATION_POINTS, size(transformed, 2))])
        valid = all(isfinite, transformed) && all(0 .< transformed .< 1)
        return f, "numeric", collect(values), valid
    elseif operation == "fit_itau"
        data = input_matrix(case)
        f = () -> fit(GumbelCopula, data; method=:itau)
        values = [Float64(f().G.θ)]
        return f, "numeric", values, all(isfinite, values)
    elseif operation == "fit_mle"
        data = input_matrix(case)
        f = () -> fit(GaussianCopula, data; method=:mle)
        values = Float64.(upper_triangle(f().Σ))
        return f, "numeric", values, all(isfinite, values)
    end
    error("Unsupported operation: $operation")
end

function measure(f)
    f()
    GC.gc()
    benchmark = @benchmarkable $f() evals=1 samples=REPETITIONS seconds=120
    trial = run(benchmark)
    med = median(trial)
    best = minimum(trial)
    return Dict(
        "median_ns" => Float64(med.time),
        "minimum_ns" => Float64(best.time),
        "memory_bytes" => Int(med.memory),
        "allocations" => Int(med.allocs),
        "iterations" => length(trial.times),
    )
end

results = Any[]
for case in SPEC["cases"]
    name = case["name"]
    println("Benchmarking $name")
    f, validation_kind, values, valid = prepare(case)
    valid || error("Local validation failed for $name")
    timing = measure(f)
    push!(results, Dict(
        "name" => name,
        "operation" => case["operation"],
        "timing" => timing,
        "validation" => Dict(
            "kind" => validation_kind,
            "valid" => valid,
            "values" => values,
        ),
    ))
end

record = Dict(
    "schema_version" => Int(SPEC["suite"]["schema_version"]),
    "language" => "Julia",
    "mode" => MODE,
    "generated_at" => string(now(UTC)),
    "runtime_version" => string(VERSION),
    "package_versions" => Dict(
        "Copulas" => string(pkgversion(Copulas)),
        "BenchmarkTools" => string(pkgversion(BenchmarkTools)),
    ),
    "environment" => Dict(
        "os" => Sys.iswindows() ? "Windows" : Sys.islinux() ? "Linux" : string(Sys.KERNEL),
        "architecture" => string(Sys.ARCH),
        "threads" => Threads.nthreads(),
        "git_commit" => get(ENV, "GITHUB_SHA", "local"),
    ),
    "benchmarks" => results,
)

mkpath(dirname(OUTPUT))
open(OUTPUT, "w") do io
    JSON3.pretty(io, record)
end
println("Wrote $OUTPUT")
