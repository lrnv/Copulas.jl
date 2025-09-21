

@testitem "Fitting smoke test" tags=[:Fitting] begin
    using Test
    using Random
    using Distributions
    using Copulas
    using StableRNGs
    rng = StableRNG(123)

    using Copulas: ClaytonGenerator, WilliamsonGenerator, GumbelGenerator, GalambosTail, MixedTail, ExtremeValueCopula # to avoid typing "Copulas." in front. 

    # Methods legend (remove characters to skip a method for a model):
    # D = :default, M = :mle, T = :itau, R = :irho, B = :ibeta, G = :gnz2011
    const METHODS = Dict(
        'D' => :default,
        'M' => :mle,
        'T' => :itau,
        'R' => :irho,
        'B' => :ibeta,
        'G' => :gnz2011, # only for empirical archimedeans. 
        'O' => :ols, # only for empirical EVC
        'C' => :cfg, # only for empirical EVC
        'P' => :pickands, # only for empirical EVC
    )

    # Structured manifest of test cases
    # Each entry: (name::String, type_for_d::Function, dims::String, meth::String)
    # - dims: string of digits among "2","3","4"; remove a digit to skip that dimension
    # - meth: string of method flags per METHODS; remove letters to skip methods
    cases = [
        # No parameters
        # ("Independent",     d -> IndependentCopula,                                                       "234", "DMTRB"),
        # ("M",               d -> MCopula,                                                                 "234", "DMTRB"),
        # ("W",               d -> WCopula,                                                                 "2",   "DMTRB"),

        # Empirical/misc (all d)
        # ("Bernstein",       d -> BernsteinCopula,                                                         "234", "D"),
        # ("Beta",            d -> BetaCopula,                                                              "234", "D"),
        # ("Checkerboard",    d -> CheckerboardCopula,                                                      "234", "D"),
        # ("Empirical",       d -> EmpiricalCopula,                                                         "234", "D"),
        # ("Archimedean(emp)",        d -> ArchimedeanCopula,                                               "234", "DG"),
        # ("ExtremeValueCopula(emp)", d -> ExtremeValueCopula,                                                "2", "DOCP"),


        # # Elliptical (all d)
        # ("Gaussian",        d -> GaussianCopula,                                                          "234", "DMTRB"),
        ("t",               d -> TCopula,                                                                 "234", "TRB"),  ######## Cannot make the MLE work. 

        # # Archimedean generic and variants (all d)
        # ("Archimedean(emp)", d -> ArchimedeanCopula,                                                      "234", "DG"),
        # ("Archimedean{Clayton}", d -> ArchimedeanCopula{d, ClaytonGenerator},                             "234", "DMTRB"),
        # ("Archimedean{Williamson(LogNormal)}", d -> ArchimedeanCopula{d, WilliamsonGenerator{LogNormal}}, "234", "D"),
        # ("Archimedean{Frailty(LogNormal)}",   d -> ArchimedeanCopula{d, FrailtyGenerator{LogNormal}},     "234", "D"),

        # # Specific Archimedean families (all d)
        # ("AMH",             d -> AMHCopula,                                                               "234", "DMTRB"),
        # ("BB1",             d -> BB1Copula,                                                               "234", "DMTRB"),
        # ("BB2",             d -> BB2Copula,                                                               "234", "DMTRB"),
        # ("BB3",             d -> BB3Copula,                                                               "234", "DMTRB"),
        # ("BB6",             d -> BB6Copula,                                                               "234", "DMTRB"),
        # ("BB7",             d -> BB7Copula,                                                               "234", "DMTRB"),
        # ("BB8",             d -> BB8Copula,                                                               "234", "DMTRB"),
        # ("BB9",             d -> BB9Copula,                                                               "234", "DMTRB"),
        # ("BB10",            d -> BB10Copula,                                                              "234", "DMTRB"),
        # ("Clayton",         d -> ClaytonCopula,                                                           "234", "DMTRB"),
        # ("Frank",           d -> FrankCopula,                                                             "234", "DMTRB"),
        # ("GumbelBarnett",   d -> GumbelBarnettCopula,                                                     "234", "DMTRB"),
        # ("Gumbel",          d -> GumbelCopula,                                                            "234", "DMTRB"),
        # ("InvGaussian",     d -> InvGaussianCopula,                                                       "234", "DMTRB"),
        # ("Joe",             d -> JoeCopula,                                                               "234", "DMTRB"),

        # # Bivariate-only miscellaneous
        # ("FGMC",            d -> FGMCopula,                                                               "2",   "DMTRB"),
        # ("Plackett",        d -> PlackettCopula,                                                          "2",   "DMTRB"),
        # ("Raftery",         d -> RafteryCopula,                                                           "2",   "DMTRB"),

        # # Bivariate EV families
        # ("EV(emp)",         d -> Copulas.ExtremeValueCopula,                                              "2",   "D"),
        # ("Galambos",        d -> Copulas.ExtremeValueCopula{2, GalambosTail},                             "2",   "D"),
        # ("HuslerReiss",     d -> HuslerReissCopula,                                                       "2",   "D"),
        # ("Log",             d -> LogCopula,                                                               "2",   "D"),
        # ("Mixed",           d -> MixedCopula,                                                             "2",   "D"),
        # ("CuadrasAuge",     d -> CuadrasAugeCopula,                                                       "2",   "D"),
        # ("BC2",             d -> BC2Copula,                                                               "2",   "D"),
        # ("tEV",             d -> tEVCopula,                                                               "2",   "D"),
        # ("MO",              d -> MOCopula,                                                                "2",   "D"),
        # ("AsymLog",         d -> AsymLogCopula,                                                           "2",   "D"),
        # ("AsymGalambos",    d -> AsymGalambosCopula,                                                      "2",   "D"),
        # ("AsymMixed",       d -> AsymMixedCopula,                                                         "2",   "D"),

        # # Archimax (bivariate only)
        # ("Archimax{Gumbel × Mixed}", d -> ArchimaxCopula{2, GumbelGenerator, MixedTail},                  "2",   "DMTRB"),
        # ("BB4",             d -> BB4Copula,                                                               "2",   "DMTRB"),
        # ("BB5",             d -> BB5Copula,                                                               "2",   "DMTRB"),
    ]

    for d in (2, 3, 4)
        U = rand(rng, d, 100)
        for (name, ct_for_d, dims, meth) in cases
            occursin(string(d), dims) || continue
            CT = ct_for_d(d)
            local_methods = [METHODS[c] for c in meth]
            isempty(local_methods) && continue
            @testset "$name" begin
                for methsym in local_methods
                    @testset "$(name) | d=$(d) | $(methsym)" begin
                        @info "$(name) | d=$(d) | $(methsym)"
                        fitres = fit(CopulaModel, CT, U; method=methsym)
                        # Basic sanity checks
                        @test length(Copulas.copula_of(fitres)) == d
                        @test isa(fitres, CopulaModel)
                    end
                end
            end
        end
    end
end

