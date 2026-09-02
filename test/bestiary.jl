# Central registry of public copula regimes exercised by the operation
# contracts. The first entry for each family is an ordinary representative;
# additional entries cover dimensions, representations, parameter branches,
# and constructor-boundary regimes that select materially different code.


const BASE_COPULA_CASES = Any[
    copula_case(AMHCopula, 2, 0.5),
    copula_case(BB1Copula, 2, 1.2, 1.5),
    copula_case(BB2Copula, 2, 1.2, 0.5),
    copula_case(BB3Copula, 2, 2.0, 1.5),
    copula_case(BB6Copula, 2, 1.2, 1.6),
    copula_case(BB7Copula, 2, 1.2, 1.6),
    copula_case(BB8Copula, 2, 1.2, 0.4),
    copula_case(BB9Copula, 2, 1.5, 2.4),
    copula_case(BB10Copula, 2, 1.5, 0.7),
    copula_case(ClaytonCopula, 3, 1.5),
    copula_case(FrankCopula, 3, 2.0),
    copula_case(GumbelCopula, 3, 1.5),
    copula_case(GumbelBarnettCopula, 2, 0.5),
    copula_case(InvGaussianCopula, 2, 0.5),
    copula_case(JoeCopula, 2, 1.5),
    copula_case(AsymGalambosCopula, 2, 1.0, 0.4, 0.6),
    copula_case(AsymLogCopula, 2, 1.5, 0.4, 0.6),
    copula_case(AsymMixedCopula, 2, 0.3, 0.2),
    copula_case(BC2Copula, 2, 0.5, 0.3),
    copula_case(CuadrasAugeCopula, 2, 0.5),
    copula_case(GalambosCopula, 3, 1.0),
    copula_case(HuslerReissCopula, 3, 1.0),
    copula_case(LogCopula, 3, 1.5),
    copula_case(MixedCopula, 2, 0.5),
    copula_case(MOCopula, 2, 0.2, 0.3, 0.4),
    copula_case(TawnCopula, 3, 2.0, [0.6, 0.7, 0.8]),
    copula_case(tEVCopula, 2, 4.0, 0.5),
    copula_case(BB4Copula, 2, 1.5, 1.0),
    copula_case(BB5Copula, 2, 1.5, 1.0),
    copula_case(GaussianCopula, 3, 0.3; numerical_atol=1e-3),
    copula_case(TCopula, 2, 4.0, [1.0 0.3; 0.3 1.0]; numerical_atol=1e-3),
    copula_case(IndependentCopula, 3),
    copula_case(MCopula, 2),
    copula_case(WCopula, 2),
    copula_case(FGMCopula, 2, 0.5),
    copula_case(PlackettCopula, 2, 2.0),
    copula_case(RafteryCopula, 3, 0.5),
    copula_case(BernsteinCopula, 2, IndependentCopula{2}(); constructor_kwargs=(; m=2)),
    copula_case(BetaCopula, 2, _FIXTURE_DATA),
    copula_case(CheckerboardCopula, 2, _FIXTURE_DATA; constructor_kwargs=(; m=2)),
    copula_case(EmpiricalCopula, 2, _FIXTURE_DATA; margin_atol=inv(size(_FIXTURE_DATA, 2))),
    copula_case(EmpiricalEVCopula, 2, _FIXTURE_DATA; constructor_kwargs=(; method=:cfg, pseudo_values=false)),
    copula_case(ArchimedeanCopula, 2, Copulas.ClaytonGenerator(1.5)),
    copula_case(ExtremeValueCopula, 2, Copulas.GalambosTail(1.0)),
    copula_case(LiouvilleCopula, 2, Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
    copula_case(NestedArchimedeanCopula, 4, Copulas.ClaytonGenerator(1.0); constructor_kwargs=(; leaves=[1, 2], children=[ClaytonCopula{2}(2.0)])),
    copula_case(ArchimaxCopula, 2, Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0)),
    copula_case(SurvivalCopula, 3, ClaytonCopula{3}(1.5), (1, 3)),

    # Additional dispatch representatives.
    copula_case(FrankCopula, 2, -2.0; conditional_at=(1, 0.4)),
    copula_case(AMHCopula, 2, -0.5; conditional_at=(1, 0.4)),
    copula_case(PlackettCopula, 2, 0.5; conditional_at=(2, 0.7)),
    copula_case(GumbelCopula, 2, 1.001; conditional_at=(1, 0.25)),
    copula_case(GumbelCopula, 2, 8.0; conditional_at=(1, 0.7)),
    copula_case(LogCopula, 2, 1.001; conditional_at=(1, 0.25)),
    copula_case(InvGaussianCopula, 2, 0.01; conditional_at=(1, 0.4)),
    copula_case(BB9Copula, 2, 1.001, 0.8; conditional_at=(1, 0.4)),
    copula_case(GumbelBarnettCopula, 2, 0.01; conditional_at=(1, 0.3)),
    copula_case(GumbelBarnettCopula, 2, 0.8; conditional_at=(1, 0.7)),
    copula_case(EmpiricalEVCopula, 3, _FIXTURE_DATA3; constructor_kwargs=(; degree=1, pseudo_values=false)),
    copula_case(ArchimedeanCopula, 2, Copulas.FrailtyGenerator(Exponential())),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Dirac(1.0), 2.0)),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Dirac(1.0), 2.5)),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Pareto(1.0), 4)),
    copula_case(ArchimedeanCopula, 2, EmpiricalGenerator(_FIXTURE_DATA)),
    copula_case(ExtremeValueCopula, 2, DiscreteSpectralTail([0.7 0.3; 0.2 0.8])),
    copula_case(GumbelCopula, 2, 1.5),
    copula_case(GalambosCopula, 2, 1.0),
    copula_case(HuslerReissCopula, 2, 1.0),
    copula_case(HuslerReissCopula, 3, [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]),
    copula_case(LogCopula, 2, 1.5),
    copula_case(AsymGalambosCopula, 3, 1.0, [0.4, 0.5, 0.6]),
    copula_case(BC2Copula, 3, [0.3, 0.7, 0.5]),
    copula_case(CuadrasAugeCopula, 3, 0.5),
    copula_case(MOCopula, 3, [0.35, 0.55, 0.40, 0.25, 0.30, 0.45, 0.70]),
    # copula_case(tEVCopula, 3, 4.0, 0.2),
    # copula_case(tEVCopula, 3, 4.0, [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]),
    copula_case(GaussianCopula, 2, 0.3; numerical_atol=1e-3),
    # copula_case(TCopula, 3, 5.0, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]; numerical_atol=1e-3), # coute 7 minutes.
    copula_case(LiouvilleCopula, 3, Copulas.ClaytonGenerator(1.0), (0.8, 1.1, 1.3)),
    copula_case(FGMCopula, 3, [0.0, 0.0, 0.0, 0.4]),
    copula_case(IndependentCopula, 2),
    copula_case(MCopula, 3),
    copula_case(RafteryCopula, 2, 0.5),
    copula_case(SurvivalCopula, 2, ClaytonCopula{2}(1.5), (1,)),
]

# Boundary representatives are declared once in the reduction graph. Adding
# their source constructors here makes every one pass the same public contracts
# and routing proofs as the manually declared representatives.
function reduction_case(edge)
    Base.@nospecialize edge

    expected_family = edge.expected_family
    prototype = nothing

    for case in BASE_COPULA_CASES
        if expected_family <: case.family
            prototype = case
            break
        end
    end

    isnothing(prototype) && error("no prototype for $(edge.name)")

    build = edge.source
    return merge(prototype, (; name=edge.name, typed=build, dynamic=build, build))
end

const REDUCTION_COPULA_CASES = Any[reduction_case(edge) for edge in CONSTRUCTOR_REDUCTIONS]

const ALL_COPULA_CASES = vcat(BASE_COPULA_CASES, REDUCTION_COPULA_CASES)
