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
function test_progress(parts...)
    now = time()
    @info "Test progress" path=join(string.(parts), " / ") elapsed=round(now - _TEST_PROGRESS_LAST[]; digits=2) total=round(now - _TEST_RUN_STARTED; digits=2)
    _TEST_PROGRESS_LAST[] = now
end

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

# Fixtures define registries and helpers but contain no assertions.  Load them
# before opening the test hierarchy so they do not appear as an empty testset.
include(joinpath(@__DIR__, "fixtures.jl"))

@testset verbose=true "Copulas.jl" begin
    selection = isempty(ARGS) ? :all : Symbol(only(ARGS))

    if selection === :equivalence
        # Minimal dependency chain for the equivalence ledger: distortion
        # fixtures and mathematical oracle types are defined by these files.
        for (obligation, file) in (("contracts", "public_surface"),
                                   ("contracts", "distortions"),
                                   ("correctness", "tails"),
                                   ("correctness", "mathematical"),
                                   ("equivalence", "specializations"))
            test_progress("targeted", obligation, file)
            include(joinpath(@__DIR__, "obligations", obligation, "$file.jl"))
        end
    elseif selection === :routing_fitting
        test_progress("targeted", "routing", "fitting")
        include(joinpath(@__DIR__, "obligations", "routing", "fitting.jl"))
    elseif selection === :all
        test_progress("Aqua.jl")
        include(joinpath(@__DIR__, "Aqua.jl"))

        for (obligation, files) in pairs(obligation_testfiles)
            @testset verbose=true "$obligation obligations" begin
                @testset verbose=true "$f.jl" for f in files
                    test_progress("$obligation obligations", "$f.jl")
                    include(joinpath(
                        @__DIR__, "obligations", string(obligation), "$f.jl"))
                end
            end
        end

        @testset verbose=true "family regressions" begin
            @testset verbose=true "$f.jl" for f in family_testfiles
                test_progress("family regressions", "$f.jl")
                include(joinpath(@__DIR__, "families", "$f.jl"))
            end
        end

        @testset verbose=true "extension regressions" begin
            declared = Set(keys(TOML.parsefile(
                joinpath(@__DIR__, "..", "Project.toml"))["extensions"]))
            represented = Set(string.(keys(extension_testfiles)))
            @test declared == represented
            @testset verbose=true "$(extension) ($(getproperty(extension_testfiles, extension)).jl)" for extension in keys(extension_testfiles)
                f = getproperty(extension_testfiles, extension)
                test_progress("extension regressions", "$f.jl")
                include(joinpath(@__DIR__, "extensions", "$f.jl"))
            end
        end
    else
        error("unknown test selection: $selection")
    end
end
