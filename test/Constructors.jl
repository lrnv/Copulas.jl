@testset "dimension-first constructors" begin
    @test @inferred(IndependentCopula{3}()) isa IndependentCopula{3}
    @test @inferred(MCopula{3}()) isa MCopula{3}
    @test @inferred(WCopula{2}()) isa WCopula{2}
    @test PlackettCopula{2}(2.0) isa PlackettCopula{2}
    @test_throws Exception WCopula{3}()
    @test_throws DimensionMismatch PlackettCopula{3}(2.0)

    Σ = [1.0 0.2; 0.2 1.0]
    @test ArchimedeanCopula{2}(Copulas.ClaytonGenerator(2.0)) isa Copulas.Copula{2}
    @test Copulas.ExtremeValueCopula{2}(Copulas.GalambosTail(1.0)) isa Copulas.Copula{2}
    @test Copulas.ExtremeValueCopula(2, Copulas.GalambosTail(1.0)) isa Copulas.Copula{2}
    @test ArchimaxCopula{2}(Copulas.ClaytonGenerator(2.0), Copulas.GalambosTail(1.0)) isa Copulas.Copula{2}
    @test TCopula{2}(4, copy(Σ)) isa TCopula{2}
    @test GaussianCopula{2,Matrix{Float64}}(2, copy(Σ)) isa GaussianCopula{2}
    @test TCopula{2,Int,Matrix{Float64}}(2, 4, copy(Σ)) isa TCopula{2}

    # These constructors can intentionally return a small union for exact
    # boundary cases, but every member has the statically selected dimension.
    @test GaussianCopula{3}(0.2) isa Copulas.Copula{3}
    @test FGMCopula{2}(0.5) isa Copulas.Copula{2}
    @test RafteryCopula{3}(0.5) isa Copulas.Copula{3}

    data = [0.1 0.4 0.8 0.6; 0.3 0.9 0.2 0.7]
    @test @inferred(EmpiricalCopula{2}(data)) isa EmpiricalCopula{2}
    @test @inferred(BetaCopula{2}(data)) isa BetaCopula{2}
    @test @inferred(CheckerboardCopula{2}(data; m=2)) isa CheckerboardCopula{2}
    @test @inferred(BernsteinCopula{2}(IndependentCopula{2}(); m=2)) isa BernsteinCopula{2}
    @test_throws DimensionMismatch EmpiricalCopula{3}(data)
    @test_throws DimensionMismatch GaussianCopula{3}([1.0 0.2; 0.2 1.0])

    base = ClaytonCopula{3}(2.0)
    @test Copulas.SubsetCopula{2}(base, (1, 3)) isa Copulas.Copula{2}
    @test SurvivalCopula{3}(base, (1, 3)) isa Copulas.Copula{3}
    nested = NestedArchimedeanCopula{4}(
        Copulas.ClaytonGenerator(1.0);
        leaves=[1, 2],
        children=[ClaytonCopula{2}(2.0)],
    )
    @test nested isa NestedArchimedeanCopula{4}
    @test_throws DimensionMismatch NestedArchimedeanCopula{3}(
        Copulas.ClaytonGenerator(1.0);
        leaves=[1, 2],
        children=[ClaytonCopula{2}(2.0)],
    )
end

@testset "named family constructors fix the dimension first" begin
    archimedean = (
        (AMHCopula, (0.5,)),
        (BB1Copula, (1.2, 1.5)),
        (BB2Copula, (1.2, 0.5)),
        (BB3Copula, (2.0, 1.5)),
        (BB6Copula, (1.2, 1.6)),
        (BB7Copula, (1.2, 1.6)),
        (BB8Copula, (1.2, 0.4)),
        (BB9Copula, (1.5, 2.4)),
        (BB10Copula, (1.5, 0.7)),
        (ClaytonCopula, (0.5,)),
        (FrankCopula, (1.0,)),
        (GumbelBarnettCopula, (0.5,)),
        (GumbelCopula, (1.5,)),
        (InvGaussianCopula, (0.5,)),
        (JoeCopula, (1.5,)),
    )
    for (family, args) in archimedean
        @test Core.apply_type(family, 2)(args...) isa Copulas.Copula{2}
    end
    @test ClaytonCopula{2}(2) isa ClaytonCopula{2}
    @test BB1Copula{2}(1, 2) isa BB1Copula{2}

    extreme_value = (
        (AsymGalambosCopula, (1.0, 0.4, 0.6)),
        (AsymLogCopula, (1.5, 0.4, 0.6)),
        (AsymMixedCopula, (0.3, 0.2)),
        (BC2Copula, (0.5, 0.3)),
        (CuadrasAugeCopula, (0.5,)),
        (GalambosCopula, (1.0,)),
        (HuslerReissCopula, (1.0,)),
        (LogCopula, (1.5,)),
        (MixedCopula, (0.5,)),
        (MOCopula, (0.2, 0.3, 0.4)),
        (tEVCopula, (4.0, 0.5)),
    )
    for (family, args) in extreme_value
        @test Core.apply_type(family, 2)(args...) isa Copulas.Copula{2}
        @test family(2, args...) isa Copulas.Copula{2}
    end
    @test GalambosCopula{2}(2) isa GalambosCopula{2}
    @test tEVCopula{2}(4, 0.5) isa tEVCopula{2}
    @test CuadrasAugeCopula{2}(0.0) isa IndependentCopula{2}
    @test CuadrasAugeCopula{2}(1.0) isa MCopula{2}

    @test GalambosCopula(2; θ=1.0) isa GalambosCopula{2}

    @test BB4Copula{2}(1.5, 1.0) isa Copulas.Copula{2}
    @test BB5Copula{2}(1.5, 1.0) isa Copulas.Copula{2}
end
