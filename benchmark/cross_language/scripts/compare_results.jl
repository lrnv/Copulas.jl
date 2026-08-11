using Dates
using JSON3
using Printf
using Statistics
using TOML

length(ARGS) == 3 || error("usage: compare_results.jl RESULTS_DIR OUTPUT_JSON OUTPUT_MARKDOWN")

const ROOT = normpath(joinpath(@__DIR__, ".."))
const SPEC = TOML.parsefile(joinpath(ROOT, "cases.toml"))
const RESULTS_DIR, OUTPUT_JSON, OUTPUT_MARKDOWN = ARGS

read_record(path) = JSON3.read(read(path, String), Dict{String,Any})

function records_for(language)
    prefix = lowercase(language) * "-"
    paths = sort(filter(path -> startswith(lowercase(basename(path)), prefix) && endswith(path, ".json"),
                        readdir(RESULTS_DIR; join=true)))
    isempty(paths) && error("No $language result files found in $RESULTS_DIR")
    records = read_record.(paths)
    all(record -> record["language"] == language, records) || error("Mislabeled $language result file")
    return records
end

function indexed(record)
    Dict(String(item["name"]) => item for item in record["benchmarks"])
end

function aggregate(records)
    modes = unique(String(record["mode"]) for record in records)
    length(modes) == 1 || error("Cannot aggregate mixed smoke/full records")
    environments = [(record["environment"]["os"], record["environment"]["architecture"]) for record in records]
    length(unique(environments)) == 1 || error("Cannot aggregate different runner regimes")
    maps = indexed.(records)
    names = sort(collect(keys(first(maps))))
    all(map -> sort(collect(keys(map))) == names, maps) || error("Benchmark names differ between rounds")
    aggregate_map = Dict{String,Any}()
    for name in names
        items = [map[name] for map in maps]
        medians = Float64[item["timing"]["median_ns"] for item in items]
        minimums = Float64[item["timing"]["minimum_ns"] for item in items]
        aggregate_map[name] = Dict(
            "name" => name,
            "operation" => first(items)["operation"],
            "median_ns" => median(medians),
            "minimum_ns" => minimum(minimums),
            "round_medians_ns" => medians,
            "validation" => first(items)["validation"],
        )
    end
    return Dict(
        "mode" => first(modes),
        "rounds" => length(records),
        "runtime_version" => first(records)["runtime_version"],
        "package_versions" => first(records)["package_versions"],
        "environment" => first(records)["environment"],
        "benchmarks" => aggregate_map,
    )
end

function format_time(ns)
    ns < 1e3 && return @sprintf("%.1f ns", ns)
    ns < 1e6 && return @sprintf("%.2f μs", ns / 1e3)
    ns < 1e9 && return @sprintf("%.2f ms", ns / 1e6)
    return @sprintf("%.2f s", ns / 1e9)
end

julia = aggregate(records_for("Julia"))
r = aggregate(records_for("R"))
julia["mode"] == r["mode"] || error("Julia and R modes differ")
julia["environment"]["os"] == r["environment"]["os"] || error("Julia and R ran on different operating systems")
julia["environment"]["architecture"] == r["environment"]["architecture"] || error("Julia and R ran on different architectures")

case_by_name = Dict(String(case["name"]) => case for case in SPEC["cases"])
names = sort(collect(keys(case_by_name)))
sort(collect(keys(julia["benchmarks"]))) == names || error("Julia results do not match cases.toml")
sort(collect(keys(r["benchmarks"]))) == names || error("R results do not match cases.toml")

comparisons = Any[]
valid = Ref(true)
for name in names
    case = case_by_name[name]
    j = julia["benchmarks"][name]
    rr = r["benchmarks"][name]
    jvalidation = j["validation"]
    rvalidation = rr["validation"]
    local_valid = Bool(jvalidation["valid"]) && Bool(rvalidation["valid"])
    comparable = jvalidation["kind"] == "numeric" && rvalidation["kind"] == "numeric"
    max_abs_error = nothing
    validation_valid = local_valid
    if comparable
        jvalues = Float64.(jvalidation["values"])
        rvalues = Float64.(rvalidation["values"])
        length(jvalues) == length(rvalues) || error("Validation vector length differs for $name")
        atol = Float64(get(case, "validation_atol", 0.0))
        rtol = Float64(get(case, "validation_rtol", 0.0))
        max_abs_error = maximum(abs.(jvalues .- rvalues); init=0.0)
        validation_valid &= all(isapprox.(jvalues, rvalues; atol=atol, rtol=rtol))
    end
    valid[] &= validation_valid
    push!(comparisons, Dict(
        "name" => name,
        "operation" => case["operation"],
        "julia_median_ns" => j["median_ns"],
        "r_median_ns" => rr["median_ns"],
        "r_over_julia" => rr["median_ns"] / j["median_ns"],
        "validation" => Dict(
            "valid" => validation_valid,
            "cross_language_numeric" => comparable,
            "max_abs_error" => max_abs_error,
        ),
    ))
end

report = Dict(
    "schema_version" => Int(SPEC["suite"]["schema_version"]),
    "generated_at" => string(now(UTC)),
    "mode" => julia["mode"],
    "valid" => valid[],
    "julia" => julia,
    "r" => r,
    "comparisons" => comparisons,
)

mkpath(dirname(OUTPUT_JSON))
open(OUTPUT_JSON, "w") do io
    JSON3.pretty(io, report)
end

mkpath(dirname(OUTPUT_MARKDOWN))
open(OUTPUT_MARKDOWN, "w") do io
    println(io, "# Julia and R benchmark comparison")
    println(io)
    println(io, "Mode: `$(report["mode"])`; rounds: Julia $(julia["rounds"]), R $(r["rounds"]); correctness: **$(valid[] ? "passed" : "failed")**.")
    println(io)
    println(io, "| Target | Julia median | R median | R / Julia | Validation |")
    println(io, "|---|---:|---:|---:|:---:|")
    for item in comparisons
        ratio = @sprintf("%.2f×", item["r_over_julia"])
        status = item["validation"]["valid"] ? "✓" : "✗"
        println(io, "| `$(item["name"])` | $(format_time(item["julia_median_ns"])) | $(format_time(item["r_median_ns"])) | $ratio | $status |")
    end
    println(io)
    println(io, "Steady-state execution only: startup, package loading, fixture parsing, model construction, and Julia compilation are excluded. Sampling rows receive local validity checks; deterministic density, rank, and fitting rows are checked numerically across languages.")
    println(io)
    println(io, "Runner: `$(julia["environment"]["os"])` / `$(julia["environment"]["architecture"])`. Julia $(julia["runtime_version"]) with Copulas.jl $(julia["package_versions"]["Copulas"]); $(r["runtime_version"]) with copula $(r["package_versions"]["copula"]).")
end

valid[] || error("Cross-language correctness validation failed; timings are not publishable")
println("Wrote $OUTPUT_JSON and $OUTPUT_MARKDOWN")
