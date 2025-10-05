

@testitem "Fitting smoke test" tags=[:Fitting] begin
    using Test
    using Random
    using Distributions
    using Copulas
    using StableRNGs
    rng = StableRNG(123)

    using Copulas: ClaytonGenerator, WilliamsonGenerator, GumbelGenerator, GalambosTail, MixedTail, ExtremeValueCopula, FrailtyGenerator # to avoid typing "Copulas." in front. 

    # Structured manifest of test cases
    # Each entry: (Type, dims::String)
    # - dims: string of digits among "2","3","4"; remove a digit to skip that dimension
    cases = [
        # No parameters
        (IndependentCopula,                              "234"),
        (MCopula,                                        "234"),
        (WCopula,                                        "2"),

        # Empirical/misc (all d)
        (BernsteinCopula,                                "234"),
        (BetaCopula,                                     "234"),
        (CheckerboardCopula,                             "234"),
        (EmpiricalCopula,                                "234"),
        (ArchimedeanCopula,                              "234"),
        (ExtremeValueCopula,                              "2"),


        # # Elliptical (all d)
        (GaussianCopula,                                "234"),
        # (TCopula,                                       "234"), # takes a loooooot of time.

        # Archimedean families wiht one parameters
        (AMHCopula,                                     "234"),
        (ClaytonCopula,                                 "234"),
        (FrankCopula,                                   "234"),
        (GumbelBarnettCopula,                           "234"),
        (GumbelCopula,                                  "234"),
        (InvGaussianCopula,                             "234"),
        (JoeCopula,                                     "234"),

        # Archimedeans families with two parameters.
        (BB1Copula,                                     "234"),
        (BB3Copula,                                     "234"),
        (BB6Copula,                                     "234"),
        (BB7Copula,                                     "234"),
        (BB8Copula,                                     "234"),
        (BB9Copula,                                     "234"),
        (BB10Copula,                                    "234"),
        
        # Bivariate-only miscellaneous
        (FGMCopula,                                     "2"),
        (PlackettCopula,                                "2"),
        (RafteryCopula,                                 "2"),

        # # Bivariate EV families
        (GalambosCopula,                                "2"),
        (HuslerReissCopula,                             "2"),
        (LogCopula,                                     "2"),
        (MixedCopula,                                   "2"),
        (CuadrasAugeCopula,                             "2"),
        (BC2Copula,                                     "2"),
        (tEVCopula,                                     "2"),
        (MOCopula,                                      "2"),
        (AsymLogCopula,                                 "2"),
        (AsymGalambosCopula,                            "2"),
        (AsymMixedCopula,                               "2"),

        # # Archimax (bivariate only)
        (ArchimaxCopula{2, GumbelGenerator, MixedTail}, "2"),
        (BB4Copula,                                     "2"),
        (BB5Copula,                                     "2"),
    ]

    for d in (2, 3, 4)
        U = rand(rng, d, 100)
        for (CT, dims) in cases
            occursin(string(d), dims) || continue
            avail = Copulas._available_fitting_methods(CT)
            if isempty(avail)
                @warn "Empty method list for $CT"
                continue
            end
            @testset "CT=$CT, d=$d" begin
                for m in avail
                    @testset "CT=$CT, d=$d, method=$m" begin
                        @info "CT=$CT, d=$d, method=$m..."
                        fitres = fit(CopulaModel, CT, U; method=m)
                        @test length(Copulas._copula_of(fitres)) == d
                        @test isa(fitres, CopulaModel)
                    end
                end
            end
        end
    end
end

@testitem "Fitting + vcov + StatsBase interfaces (reduced)" tags=[:vcov] begin
    using Test, Random, Distributions, Copulas, StableRNGs, LinearAlgebra, Statistics, StatsBase
    rng = StableRNG(2025)

    reps = [
            # Elliptical
            (GaussianCopula, 2, :mle),
            (GaussianCopula, 3, :mle),
            # (TCopula, 2, :mle), # maybe much time?

            # Archimedean one parameter
            (ClaytonCopula,  2, :mle),
            (GumbelCopula,   2, :itau),   # rank-based for godambe
            (FrankCopula,    2, :mle),
            (JoeCopula,      2, :itau),

            # Archimedean two params
            (BB1Copula,      2, :mle),
            (BB7Copula,      2, :mle),

            # Extreme Value
            (GalambosCopula, 2, :mle),
            (HuslerReissCopula, 2, :mle),
        ]

    function psd_ok(V; tol=1e-7)
        vals = eigvals(Symmetric(Matrix(V)))
        minimum(vals) >= -tol
    end

    n = 500 # maybe are many observations?
    for (CT, d, method) in reps
        C0 = Copulas._example(CT, d)
        true_θ = StatsBase.coef(C0)
        U  = rand(rng, C0, n)
        M  = fit(CopulaModel, CT, U; method=method, vcov=true, derived_measures=false)
        estimated_θ = StatsBase.coef(M)
        @test estimated_θ ≈ true_θ atol=0.5 #this tol is very big in some case for example gaussian because it's support is [-1,1]
        @test isa(StatsBase.vcov(M), AbstractMatrix)
        @test size(StatsBase.vcov(M)) == (StatsBase.dof(M), StatsBase.dof(M))
        @test psd_ok(StatsBase.vcov(M))
        # stderror/confint dimensions
        se = StatsBase.stderror(M)
        θ  = StatsBase.coef(M)
        @test length(se) == length(θ) == StatsBase.dof(M)
        lo, hi = StatsBase.confint(M; level=0.95)
        @test length(lo) == length(hi) == length(θ)
        # Information criteria...
        @test isfinite(StatsBase.aic(M))
        @test isfinite(StatsBase.bic(M))
    end
end
