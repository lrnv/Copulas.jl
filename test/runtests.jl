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
    "MiscelaneousCopulas",
    "SklarDist",
    "GenericTests",
]

# You can override the definition of this GenericTestFilter if you want. 
GenericTestFilter(C) = true # the default value lets every copula go through. 

# An example: 
# GenericTestFilter(C) = C isa JoeCopula || C isa GumbelCopula # You can filter on your model. 

@testset verbose=true "Copulas.jl testings"  begin
    @testset verbose=true "f = $f.jl" for f in testfiles  
        @info "Launching test file $f.jl"
        include(joinpath(dirname(@__FILE__), "$f.jl"))
    end
end