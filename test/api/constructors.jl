# Public-API proof: verifies every public copula family constructor, the
# type-stable `{d}` and runtime `(d, ...)` forms and inferred forms.
function test_constructor_case(case)
    Base.@nospecialize case
    typed_value = nothing
    @testset "$(case.name)" begin
        expr = typed_constructor_expr(case)
        typed_value = Core.eval(@__MODULE__, :(@inferred $expr))

        dynamic = build_dynamic(case)
        @test typeof(typed_value) === typeof(dynamic)
        @test params(typed_value) == params(dynamic)
    end
    return typed_value
end

@testset "documented dimension-inferred constructors" begin
    function same_model(inferred, canonical)
        Base.@nospecialize inferred
        Base.@nospecialize canonical
        @test typeof(inferred) === typeof(canonical)
        @test params(inferred) == params(canonical)
    end

    Σ3 = [1.0 0.3 0.2; 0.3 1.0 0.25; 0.2 0.25 1.0]
    Γ3 = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
    same_model(GaussianCopula(Σ3), GaussianCopula{3}(Σ3))
    same_model(TCopula(4.0, Σ3), TCopula{3}(4.0, Σ3))
    same_model(BetaCopula(_FIXTURE_DATA), BetaCopula{2}(_FIXTURE_DATA))
    same_model(EmpiricalCopula(_FIXTURE_DATA), EmpiricalCopula{2}(_FIXTURE_DATA))
    same_model(CheckerboardCopula(_FIXTURE_DATA; m=2), CheckerboardCopula{2}(_FIXTURE_DATA; m=2))
    same_model(BernsteinCopula(IndependentCopula{2}(); m=2), BernsteinCopula{2}(IndependentCopula{2}(); m=2))
    same_model(BernsteinCopula(2, _FIXTURE_DATA; m=2), BernsteinCopula{2}(_FIXTURE_DATA; m=2))
    same_model(PlackettCopula(2.0), PlackettCopula{2}(2.0))
    same_model(WCopula(), WCopula{2}())

    G = Copulas.ClaytonGenerator(1.0)
    α = (1.0, 2.0)
    same_model(LiouvilleCopula(G, α), LiouvilleCopula{2}(G, α))

    nested_kwargs = (; leaves=[1, 2], children=[ClaytonCopula{2}(2.0)])
    same_model(NestedArchimedeanCopula(G; nested_kwargs...), NestedArchimedeanCopula{4}(G; nested_kwargs...))

    base = ClaytonCopula{3}(1.5)
    same_model(SurvivalCopula(base, (1, 3)), SurvivalCopula{3}(base, (1, 3)))

    survival = SurvivalCopula{2}(ClaytonCopula{2}(1.5), (1,))
    same_model(SurvivalCopula{2}(ClaytonCopula{2}(1.5), (1,)), survival)

    base2 = ClaytonCopula{2}(1.5)
    full_survival = SurvivalCopula(base2)
    @test Copulas.basecopula(full_survival) === base2
    @test Copulas.flipmask(full_survival) == (true, true)
    @test Copulas.flips(full_survival) == (1, 2)

    rotations = (
        Rotated90Copula(base2) => (true, false),
        Rotated180Copula(base2) => (true, true),
        Rotated270Copula(base2) => (false, true),
    )
    for (rotation, mask) in rotations
        @test rotation isa Copulas.AbstractReflectedCopula{2}
        @test Copulas.basecopula(rotation) === base2
        @test Copulas.flipmask(rotation) == mask
        @test Copulas.flips(rotation) == Tuple(i for i in 1:2 if mask[i])
        reference = SurvivalCopula(base2, mask)
        point = [0.37, 0.68]
        @test cdf(rotation, point) ≈ cdf(reference, point)
        @test logpdf(rotation, point) ≈ logpdf(reference, point)
    end

    @test_throws MethodError Rotated90Copula(ClaytonCopula{3}(1.5))
    @test_throws MethodError Rotated180Copula(ClaytonCopula{3}(1.5))
    @test_throws MethodError Rotated270Copula(ClaytonCopula{3}(1.5))
    @test_throws DimensionMismatch Rotated90Copula(3, base2)
    @test_throws MethodError Rotated180Copula(2, ClaytonCopula{3}(1.5))
    @test_throws MethodError Rotated270Copula(3, ClaytonCopula{3}(1.5))

    B = [0.7 0.3; 0.2 0.8]
    spectral = Copulas.DiscreteSpectralTail(B)
    same_model(ExtremeValueCopula(2, spectral), ExtremeValueCopula{2}(spectral))

    @test_throws ArgumentError ExtremeValueCopula{3}(spectral)

    same_model(BC2Copula([0.3, 0.7, 0.5]), BC2Copula{3}([0.3, 0.7, 0.5]))
    same_model(MOCopula([0.2, 0.3, 0.4]), MOCopula{2}([0.2, 0.3, 0.4]))
    same_model(HuslerReissCopula(Γ3), HuslerReissCopula{3}(Γ3))
    same_model(EmpiricalEVCopula(_FIXTURE_DATA; method=:cfg,pseudo_values=false), EmpiricalEVCopula{2}(_FIXTURE_DATA; method=:cfg,pseudo_values=false))
    same_model(AsymGalambosCopula{3}(1.0, [0.4, 0.5, 0.6]), ExtremeValueCopula{3}(Copulas.AsymGalambosTail(1.0, [0.4, 0.5, 0.6])))
    same_model(tEVCopula{3}(4.0, Σ3),ExtremeValueCopula{3}(Copulas.tEVTail(4.0, Σ3)))
end

@testset "elliptical constructors own matrix parameters" begin
    for build in (Σ -> GaussianCopula(Σ), Σ -> TCopula(4.0, Σ))
        covariance = [4 2; 2 9]
        original = copy(covariance)
        C = build(covariance)
        @test covariance == original
        @test params(C).Σ ≈ [1.0 1/3; 1/3 1.0]

        covariance[1, 2] = covariance[2, 1] = 0
        @test params(C).Σ ≈ [1.0 1/3; 1/3 1.0]

        first_params = params(C)
        second_params = params(C)
        @test first_params.Σ == second_params.Σ
        @test first_params.Σ !== second_params.Σ
        first_params.Σ[1, 2] = first_params.Σ[2, 1] = 0
        @test params(C).Σ ≈ [1.0 1/3; 1/3 1.0]
    end
end

@testset "public constructors" begin
    constructed = map(test_constructor_case, ALL_COPULA_CASES)
    declared_symbols = Set(symbol for symbol in public_symbols()
        if Base.isexported(Copulas, symbol) &&
           getfield(Copulas, symbol) isa Type &&
           getfield(Copulas, symbol) <: Copulas.Copula)
    @test Set(case.symbol for case in ALL_COPULA_CASES) == declared_symbols
    for (case, C) in zip(ALL_COPULA_CASES, constructed)
        @test C isa case.family
    end
    @test_throws Exception WCopula{3}()
    @test_throws DimensionMismatch PlackettCopula{3}(2.0)
end
