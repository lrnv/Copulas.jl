using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test

const rng = StableRNG(123)

testfiles = [
    "Aqua",
    "fixtures",
    "contracts/distribution",
    "contracts/density",
    "contracts/subsetting",
    "contracts/conditioning",
    "contracts/rosenblatt",
    "contracts/dependence",
    "contracts/constructors",
    "contracts/copulas",
    "contracts/fitting",
    "contracts/sklar",
    "contracts/utilities",
    "components/generators",
    "components/tails",
    "paths/dispatch_paths",
]

# Legacy files remain enabled while their coverage is migrated to the new
# contract-based test architecture.
legacy_testfiles = [
    "ArchimedeanCopulas",
    "LiouvilleCopula",
    "NestedArchimedeanCopula",
    "ConditionalDistribution",
    "Constructors",
    "EllipticalCopulas",
    "ExpectationMaximizationExt",
    "FittingTest",
    "MiscelaneousCopulas",
    "NatafTest",
    "SklarDist",
    "Subsetting",
    "ExtremeValueArchitecture",
]

@testset verbose=true "Copulas.jl testings"  begin
    @testset verbose=true "$f.jl" for f in testfiles
        @info "Launching test file $f.jl"
        elapsed = @elapsed include(joinpath(@__DIR__, "$f.jl"))
        @info "Completed test file $f.jl" elapsed
    end

    @testset verbose=true "legacy/$f.jl" for f in legacy_testfiles
        @info "Launching legacy test file $f.jl"
        elapsed = @elapsed include(joinpath(@__DIR__, "old", "$f.jl"))
        @info "Completed legacy test file $f.jl" elapsed
    end
end
