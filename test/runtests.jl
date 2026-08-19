using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test

const rng = StableRNG(123)

const TEST_GROUP = get(ENV, "COPULAS_TEST_GROUP", "all")

core_testfiles = [
    "Aqua",
    "ArchimedeanCopulas",
    "NestedArchimedeanCopula",
    "ConditionalDistribution",
    "EllipticalCopulas",
    "ExpectationMaximizationExt",
    "FittingTest",
    "MiscelaneousCopulas",
    "NatafTest",
    "SklarDist",
    "Subsetting",
]

testfiles = TEST_GROUP == "all" ? [core_testfiles; "GenericTests"] :
            TEST_GROUP == "core" ? core_testfiles :
            startswith(TEST_GROUP, "generic-") ? ["GenericTests"] :
            error("Unknown COPULAS_TEST_GROUP: $TEST_GROUP")

# You can override the definition of this GenericTestFilter if you want. 
GenericTestFilter(C) = true # the default value lets every copula go through. 

# An example: 
# GenericTestFilter(C) = C isa BC2Copula || C isa MOCopula || C isa CuadrasAugeCopula # || C isa GumbelCopula # You can filter on your model. 

@testset verbose=true "Copulas.jl testings"  begin
    @testset verbose=true "f = $f.jl" for f in testfiles  
        @info "Launching test file $f.jl"
        include(joinpath(dirname(@__FILE__), "$f.jl"))
    end
end
