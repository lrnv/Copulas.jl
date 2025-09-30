

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
        #(IndependentCopula,                              "234"),
        #(MCopula,                                        "234"),
        #(WCopula,                                        "2"),

        # Empirical/misc (all d)
        #(BernsteinCopula,                                "234"),
        #(BetaCopula,                                     "234"),
        #(CheckerboardCopula,                             "234"),
        #(EmpiricalCopula,                                "234"),
        #(ArchimedeanCopula,                              "234"),
        #(ExtremeValueCopula,                              "2"),


        # # Elliptical (all d)
        #(GaussianCopula,                                "234"),
        # (TCopula,                                       "234"), # takes a loooooot of time.

        # Archimedean families wiht one parameters
        #(AMHCopula,                                     "234"),
        #(ClaytonCopula,                                 "234"),
        #(FrankCopula,                                   "234"),
        #(GumbelBarnettCopula,                           "234"),
        #(GumbelCopula,                                  "234"),
        #(InvGaussianCopula,                             "234"),
        #(JoeCopula,                                     "234"),

        # Archimedeans families with two parameters.
        #(BB1Copula,                                     "234"),
        #(BB3Copula,                                     "234"),
        #(BB6Copula,                                     "234"),
        #(BB7Copula,                                     "234"),
        #(BB8Copula,                                     "234"),
        #(BB9Copula,                                     "234"),
        #(BB10Copula,                                    "234"),
        
        # Bivariate-only miscellaneous
        #(FGMCopula,                                     "2"),
        #(PlackettCopula,                                "2"),
        #(RafteryCopula,                                 "2"),

        # # Bivariate EV families
        #(GalambosCopula,                                "2"),
        #(HuslerReissCopula,                             "2"),
        #(LogCopula,                                     "2"),
        #(MixedCopula,                                   "2"),
        #(CuadrasAugeCopula,                             "2"),
        #(BC2Copula,                                     "2"),
        #(tEVCopula,                                     "2"),
        #(MOCopula,                                      "2"),
        #(AsymLogCopula,                                 "2"),
        #(AsymGalambosCopula,                            "2"),
        #(AsymMixedCopula,                               "2"),

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
                        @test length(Copulas.copula_of(fitres)) == d
                        @test isa(fitres, CopulaModel)
                    end
                end
            end
        end
    end
end

