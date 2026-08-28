# Test-suite orchestrator. See test/README.md for the four proof obligations
# implemented by contracts, mathematical oracles, specialization comparisons,
# and exhaustive dispatch registries.
using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test

const rng = StableRNG(123)

infrastructure_testfiles = ["Aqua", "fixtures"]

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

@testset verbose=true "Copulas.jl testings"  begin
    @testset verbose=true "infrastructure/$f.jl" for f in infrastructure_testfiles
        @info "Launching test file $f.jl"
        elapsed = @elapsed include(joinpath(@__DIR__, "$f.jl"))
        @info "Completed test file $f.jl" elapsed
    end

    for (obligation, files) in pairs(obligation_testfiles)
        @testset verbose=true "obligation: $obligation" begin
            for f in files
                @info "Launching obligation test file" obligation file=f
                elapsed = @elapsed include(joinpath(
                    @__DIR__, "obligations", string(obligation), "$f.jl"))
                @info "Completed obligation test file" obligation file=f elapsed
            end
        end
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
