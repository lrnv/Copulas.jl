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
    "GenericTests",
]

# You can override the definition of this GenericTestFilter if you want. 
GenericTestFilter(C) = true # the default value lets every copula go through. 

# An example: 
# GenericTestFilter(C) = C isa BC2Copula || C isa MOCopula || C isa CuadrasAugeCopula # || C isa GumbelCopula # You can filter on your model. 

@testset verbose=true "Copulas.jl testings"  begin
    @testset verbose=true "$f.jl" for f in testfiles
        @info "Launching test file $f.jl"
        include(joinpath(@__DIR__, "$f.jl"))
    end

    @testset verbose=true "legacy/$f.jl" for f in legacy_testfiles
        @info "Launching legacy test file $f.jl"
        include(joinpath(@__DIR__, "old", "$f.jl"))
    end
end
