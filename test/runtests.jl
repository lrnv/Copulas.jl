using Aqua, Copulas, Distributions, ForwardDiff, HCubature, 
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions, 
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs, 
    Statistics, StatsBase, Test

const rng = StableRNG(123)

# You can comment the lines to avoid running some tests while you develop:
testfiles = [
    "Aqua",
    "ArchimedeanCopulas",
    "ConditionalDistribution",
    "EllipticalCopulas",
    "FittingTest",
    "GenericTests",
    "MiscelaneousCopulas",
    "SklarDist",
]

@testset "Copulas.jl testings" verbose=true begin
    @testset for f in testfiles 
        @info "Launching test file $f.jl"
        include(joinpath(dirname(@__FILE__), "$f.jl"))
    end
end