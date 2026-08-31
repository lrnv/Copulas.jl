# Public-API proof: verifies every public copula family constructor, the
# type-stable `{d}` and runtime `(d, ...)` forms and inferred forms.
function test_constructor_case(case)
    typed_value = nothing
    @testset "$(case.name)" begin
        typed_value = if case.allowed_inference === nothing
            @inferred case.typed()
        else
            @inferred case.allowed_inference case.typed()
        end
        dynamic = case.dynamic()
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
    same_model(typeof(survival)(1.5), survival)

    B = [0.7 0.3; 0.2 0.8]
    spectral = DiscreteSpectralTail(B)
    same_model(ExtremeValueCopula(2, spectral), ExtremeValueCopula{2}(spectral))

    @test_throws ArgumentError ExtremeValueCopula{3}(spectral)

    same_model(BC2Copula([0.3, 0.7, 0.5]), BC2Copula{3}([0.3, 0.7, 0.5]))
    same_model(MOCopula([0.2, 0.3, 0.4]), MOCopula{2}([0.2, 0.3, 0.4]))
    same_model(HuslerReissCopula(Γ3), HuslerReissCopula{3}(Γ3))
    same_model(EmpiricalEVCopula(_FIXTURE_DATA; method=:cfg,pseudo_values=false), EmpiricalEVCopula{2}(_FIXTURE_DATA; method=:cfg,pseudo_values=false))
    same_model(AsymGalambosCopula{3}(1.0, [0.4, 0.5, 0.6]), ExtremeValueCopula{3}(Copulas.AsymGalambosTail(1.0, [0.4, 0.5, 0.6])))
    same_model(tEVCopula{3}(4.0, Σ3),ExtremeValueCopula{3}(Copulas.tEVTail(4.0, Σ3)))
end

@testset "public constructors" begin
    constructed = map(test_constructor_case, COPULA_CASES)
    declared_symbols = Set(symbol for symbol in public_symbols()
        if Base.isexported(Copulas, symbol) &&
           getfield(Copulas, symbol) isa Type &&
           getfield(Copulas, symbol) <: Copulas.Copula)
    @test Set(case.symbol for case in COPULA_CASES) == declared_symbols
    for (case, C) in zip(COPULA_CASES, constructed)
        @test C isa case.family
    end
    @test_throws Exception WCopula{3}()
    @test_throws DimensionMismatch PlackettCopula{3}(2.0)
end
