# Test-suite orchestrator. See the developer guide's "Testing architecture"
# section for the four proof obligations implemented by contracts,
# mathematical oracles, specialization comparisons, and dispatch registries.
using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test, TOML

const rng = StableRNG(123)
const _TEST_RUN_STARTED = time()
const _TEST_PROGRESS_LAST = Ref(_TEST_RUN_STARTED)
const _TEST_TIMINGS = Dict{String,Any}()
function test_progress(parts...)
    # Logging must not generate a new method instance for every combination of
    # family names, symbols, dimensions, and fitting methods passed by tests.
    Base.@nospecialize parts
    now = time()
    @info "Test progress" path=join(string.(parts), " / ") elapsed=round(now - _TEST_PROGRESS_LAST[]; digits=2) total=round(now - _TEST_RUN_STARTED; digits=2)
    _TEST_PROGRESS_LAST[] = now
end

function timed_include(label, path)
    started = time()
    try
        timing = @timed include(path)
        _TEST_TIMINGS[string(label)] = Dict(
            "elapsed_seconds" => timing.time,
            "compile_seconds" => timing.compile_time,
            "recompile_seconds" => timing.recompile_time,
        )
        return timing.value
    finally
        get!(_TEST_TIMINGS, string(label), Dict(
            "elapsed_seconds" => time() - started,
            "compile_seconds" => -1.0,
            "recompile_seconds" => -1.0,
        ))
    end
end

function write_test_timings()
    path = get(ENV, "COPULAS_TEST_TIMINGS", "")
    isempty(path) && return
    mkpath(dirname(abspath(path)))
    report = Dict(
        "total_seconds" => time() - _TEST_RUN_STARTED,
        "files" => _TEST_TIMINGS,
    )
    open(path, "w") do io
        TOML.print(io, report; sorted=true)
    end
end
atexit(write_test_timings)

obligation_testfiles = (
    contracts = [
        "public_surface", "constructors", "copulas", "fitting", "sklar",
        "utilities", "distortions", "univariate_distributions",
        "public_compositions",
    ],
    correctness = [
        "generators", "tails", "measure_inverses", "mathematical",
        "statistical",
    ],
    equivalence = ["specializations"],
    routing = ["dispatch", "branches", "fitting"],
)

family_testfiles = [
    "archimedean",
    "conditioning",
    "constructors",
    "elliptical",
    "extreme_value_architecture",
    "extreme_value",
    "fitting",
    "liouville",
    "miscellaneous",
    "nataf",
    "nested_archimedean",
    "sklar",
    "subsetting",
]

extension_testfiles = (
    CopulasExpectationMaximizationExt="expectation_maximization",
    CopulasPlotsExt="plots",
)

function run_obligations(groups)
    Base.@nospecialize groups
    for obligation in groups
        files = getproperty(obligation_testfiles, obligation)
        @testset verbose=true "$obligation obligations" begin
            @testset verbose=true "$f.jl" for f in files
                test_progress("$obligation obligations", "$f.jl")
                timed_include("$obligation/$f.jl", joinpath(
                    @__DIR__, "obligations", string(obligation), "$f.jl"))
            end
        end
    end
end

function run_family_regressions()
    @testset verbose=true "family regressions" begin
        @testset verbose=true "$f.jl" for f in family_testfiles
            test_progress("family regressions", "$f.jl")
            timed_include("families/$f.jl",
                joinpath(@__DIR__, "families", "$f.jl"))
        end
    end
end

function run_extension_regressions()
    @testset verbose=true "extension regressions" begin
        declared = Set(keys(TOML.parsefile(
            joinpath(@__DIR__, "..", "Project.toml"))["extensions"]))
        represented = Set(string.(keys(extension_testfiles)))
        @test declared == represented
        @testset verbose=true "$(extension) ($(getproperty(extension_testfiles, extension)).jl)" for extension in keys(extension_testfiles)
            f = getproperty(extension_testfiles, extension)
            test_progress("extension regressions", "$f.jl")
            timed_include("extensions/$f.jl",
                joinpath(@__DIR__, "extensions", "$f.jl"))
        end
    end
end

# Fixtures define registries and helpers but contain no assertions.  Load them
# before opening the test hierarchy so they do not appear as an empty testset.
timed_include("infrastructure/fixtures.jl", joinpath(@__DIR__, "fixtures.jl"))

@testset verbose=true "Copulas.jl" begin
    test_progress("Aqua.jl")
    timed_include("infrastructure/Aqua.jl", joinpath(@__DIR__, "Aqua.jl"))
    run_obligations(keys(obligation_testfiles))
    run_family_regressions()
    run_extension_regressions()
end
