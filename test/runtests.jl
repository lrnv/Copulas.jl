# Test-suite orchestrator. See test/README.md for the four proof obligations
# implemented by contracts, mathematical oracles, specialization comparisons,
# and exhaustive dispatch registries.
using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test

const rng = StableRNG(123)

testfiles = [
    "Aqua",
    "fixtures",
    "contracts/public_surface",
    "contracts/constructors",
    "contracts/copulas",
    "contracts/fitting",
    "contracts/sklar",
    "contracts/utilities",
    "components/generators",
    "components/tails",
    "components/distortions",
    "components/univariate_distributions",
    "components/public_compositions",
    "components/measure_inverses",
    "paths/mathematical_coherence",
    "paths/dispatch_paths",
    "paths/statistical_paths",
    "paths/fitting_paths",
]

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

extension_testfiles = ["expectation_maximization"]

@testset verbose=true "Copulas.jl testings"  begin
    @testset verbose=true "$f.jl" for f in testfiles
        @info "Launching test file $f.jl"
        elapsed = @elapsed include(joinpath(@__DIR__, "$f.jl"))
        @info "Completed test file $f.jl" elapsed
    end

    @testset verbose=true "families/$f.jl" for f in family_testfiles
        @info "Launching family regression file $f.jl"
        elapsed = @elapsed include(joinpath(@__DIR__, "families", "$f.jl"))
        @info "Completed family regression file $f.jl" elapsed
    end

    @testset verbose=true "extensions/$f.jl" for f in extension_testfiles
        @info "Launching extension regression file $f.jl"
        elapsed = @elapsed include(joinpath(@__DIR__, "extensions", "$f.jl"))
        @info "Completed extension regression file $f.jl" elapsed
    end
end
