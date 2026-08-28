# Test-suite orchestrator. See the developer guide's "Testing architecture"
# section for the four proof obligations implemented by contracts,
# mathematical oracles, specialization comparisons, and dispatch registries.
using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test

const rng = StableRNG(123)

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
    routing = ["dispatch", "fitting"],
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

extension_testfiles = ["expectation_maximization", "plots"]

# Fixtures define registries and helpers but contain no assertions.  Load them
# before opening the test hierarchy so they do not appear as an empty testset.
include(joinpath(@__DIR__, "fixtures.jl"))

@testset verbose=true "Copulas.jl" begin
    include(joinpath(@__DIR__, "Aqua.jl"))

    for (obligation, files) in pairs(obligation_testfiles)
        @testset verbose=true "$obligation obligations" begin
            @testset verbose=true "$f.jl" for f in files
                include(joinpath(
                    @__DIR__, "obligations", string(obligation), "$f.jl"))
            end
        end
    end

    @testset verbose=true "family regressions" begin
        @testset verbose=true "$f.jl" for f in family_testfiles
            include(joinpath(@__DIR__, "families", "$f.jl"))
        end
    end

    @testset verbose=true "extension regressions" begin
        @testset verbose=true "$f.jl" for f in extension_testfiles
            include(joinpath(@__DIR__, "extensions", "$f.jl"))
        end
    end
end
