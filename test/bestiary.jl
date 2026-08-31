# The single central bestiary. The first entry for each public family is its
# canonical contract/constructor representative; later entries exercise extra
# dimensions, representations, or value-dependent routes only.

allow_I(d)   = IndependentCopula{d}
allow_IW(d)  = Union{IndependentCopula{d}, WCopula{d}}
allow_IM(d)  = Union{IndependentCopula{d}, MCopula{d}}
allow_IWM(d) = Union{IndependentCopula{d}, WCopula{d}, MCopula{d}}

allow_Clayton(d) = Union{ClaytonCopula{d}, allow_IWM(d)}
allow_Joe(d)     = Union{JoeCopula{d}, allow_IM(d)}
allow_Gumbel(d)  = Union{GumbelCopula{d}, allow_IM(d)}
allow_AMH(d)     = Union{AMHCopula{d}, allow_I(d)}

allow_A(d) = Union{allow_IWM(d),ArchimedeanCopula{d}}
allow_E(d) = Union{IndependentCopula{d}, MCopula{d}, ExtremeValueCopula{d}}
allow_FGM(d) = Union{allow_IWM(d), FGMCopula{d}}
allow_AsymMixed(d) = Union{IndependentCopula{d}, MixedCopula{d}, AsymMixedCopula{d}}

const ALL_COPULA_CASES = (
    copula_case(AMHCopula, 2, 0.5; allowed_inference=allow_I(2)),
    copula_case(BB1Copula, 2, 1.2, 1.5; allowed_inference=allow_A(2)),
    copula_case(BB2Copula, 2, 1.2, 0.5),
    copula_case(BB3Copula, 2, 2.0, 1.5),
    copula_case(BB6Copula, 2, 1.2, 1.6; allowed_inference=allow_A(2)),
    copula_case(BB7Copula, 2, 1.2, 1.6; allowed_inference=allow_A(2)),
    copula_case(BB8Copula, 2, 1.2, 0.4; allowed_inference=allow_A(2)),
    copula_case(BB9Copula, 2, 1.5, 2.4),
    copula_case(BB10Copula, 2, 1.5, 0.7; allowed_inference=allow_AMH(2)),
    copula_case(ClaytonCopula, 3, 1.5; allowed_inference=allow_IWM(3)),
    copula_case(FrankCopula, 3, 2.0; allowed_inference=allow_IWM(3)),
    copula_case(GumbelCopula, 3, 1.5; allowed_inference=allow_IM(3)),
    copula_case(GumbelBarnettCopula, 2, 0.5; allowed_inference=allow_I(2)),
    copula_case(InvGaussianCopula, 2, 0.5; allowed_inference=allow_I(2)),
    copula_case(JoeCopula, 2, 1.5; allowed_inference=allow_IM(2)),
    copula_case(AsymGalambosCopula, 2, 1.0, 0.4, 0.6; allowed_inference=allow_E(2)),
    copula_case(AsymLogCopula, 2, 1.5, 0.4, 0.6; allowed_inference=allow_E(2)),
    copula_case(AsymMixedCopula, 2, 0.3, 0.2; allowed_inference=allow_AsymMixed(2)),
    copula_case(BC2Copula, 2, 0.5, 0.3; allowed_inference=BC2Copula{2}),
    copula_case(CuadrasAugeCopula, 2, 0.5; allowed_inference=allow_IM(2)),
    copula_case(GalambosCopula, 3, 1.0; allowed_inference=allow_IM(3)),
    copula_case(HuslerReissCopula, 3, 1.0; allowed_inference=allow_IM(3)),
    copula_case(LogCopula, 3, 1.5; allowed_inference=allow_IM(3)),
    copula_case(MixedCopula, 2, 0.5; allowed_inference=Union{IndependentCopula{2}, MixedCopula{2}}),
    copula_case(MOCopula, 2, 0.2, 0.3, 0.4; allowed_inference=MOCopula{2}),
    copula_case(TawnCopula, 3, 2.0, [0.6, 0.7, 0.8]; allowed_inference=allow_E(3)),
    copula_case(tEVCopula, 2, 4.0, 0.5; allowed_inference=MCopula{2}),
    copula_case(BB4Copula, 2, 1.5, 1.0; allowed_inference=Union{ArchimaxCopula{2}, ClaytonCopula{2}}),
    copula_case(BB5Copula, 2, 1.5, 1.0; allowed_inference=Union{ArchimaxCopula{2}, GumbelCopula{2}}),
    copula_case(GaussianCopula, 3, 0.3; allowed_inference=IndependentCopula, numerical_atol=1e-3),
    copula_case(TCopula, 2, 4.0, [1.0 0.3; 0.3 1.0]),
    copula_case(IndependentCopula, 3),
    copula_case(MCopula, 2),
    copula_case(WCopula, 2),
    copula_case(FGMCopula, 2, 0.5; allowed_inference=allow_FGM(2)),
    copula_case(PlackettCopula, 2, 2.0; allowed_inference=allow_IWM(2)),
    copula_case(RafteryCopula, 3, 0.5; allowed_inference=allow_IM(3)),
    copula_case(BernsteinCopula, 2, IndependentCopula{2}(); constructor_kwargs=(; m=2)),
    copula_case(BetaCopula, 2, _FIXTURE_DATA),
    copula_case(CheckerboardCopula, 2, _FIXTURE_DATA; constructor_kwargs=(; m=2)),
    copula_case(EmpiricalCopula, 2, _FIXTURE_DATA; margin_atol=inv(size(_FIXTURE_DATA, 2))),
    copula_case(EmpiricalEVCopula, 2, _FIXTURE_DATA; constructor_kwargs=(; method=:cfg, pseudo_values=false), allowed_inference=EmpiricalEVCopula),
    copula_case(ArchimedeanCopula, 2, Copulas.ClaytonGenerator(1.5)),
    copula_case(ExtremeValueCopula, 2, Copulas.GalambosTail(1.0)),
    copula_case(LiouvilleCopula, 2, Copulas.ClaytonGenerator(1.0), (1.0, 2.0); allowed_inference=ArchimedeanCopula),
    copula_case(NestedArchimedeanCopula, 4, Copulas.ClaytonGenerator(1.0); constructor_kwargs=(; leaves=[1, 2], children=[ClaytonCopula{2}(2.0)]), allowed_inference=Union{NestedArchimedeanCopula,ArchimedeanCopula}),
    copula_case(ArchimaxCopula, 2, Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0)),
    copula_case(SurvivalCopula, 3, ClaytonCopula{3}(1.5), (1, 3); allowed_inference=Union{SurvivalCopula{3,ClaytonCopula{3,Float64}}, ClaytonCopula{3,Float64}}),

    # Additional dispatch representatives.
    copula_case(FrankCopula, 2, -2.0; allowed_inference=allow_IWM(2), conditional_at=(1, 0.4)),
    copula_case(AMHCopula, 2, -0.5; allowed_inference=allow_I(2), conditional_at=(1, 0.4)),
    copula_case(PlackettCopula, 2, 0.5; allowed_inference=allow_IWM(2), conditional_at=(2, 0.7)),
    copula_case(GumbelCopula, 2, 1.001; allowed_inference=allow_IM(2), conditional_at=(1, 0.25)),
    copula_case(GumbelCopula, 2, 8.0; allowed_inference=allow_IM(2), conditional_at=(1, 0.7)),
    copula_case(LogCopula, 2, 1.001; allowed_inference=allow_IM(2), conditional_at=(1, 0.25)),
    copula_case(InvGaussianCopula, 2, 0.01; allowed_inference=allow_I(2), conditional_at=(1, 0.4)),
    copula_case(BB9Copula, 2, 1.001, 0.8; conditional_at=(1, 0.4)),
    copula_case(GumbelBarnettCopula, 2, 0.01; allowed_inference=allow_I(2), conditional_at=(1, 0.3)),
    copula_case(GumbelBarnettCopula, 2, 0.8; allowed_inference=allow_I(2), conditional_at=(1, 0.7)),
    copula_case(EmpiricalEVCopula, 3, _FIXTURE_DATA3; constructor_kwargs=(; degree=1, pseudo_values=false)),
    copula_case(ArchimedeanCopula, 2, Copulas.FrailtyGenerator(Exponential())),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Dirac(1.0), 2.0)),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Dirac(1.0), 2.5)),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Pareto(1.0), 4)),
    copula_case(ArchimedeanCopula, 2, EmpiricalGenerator(_FIXTURE_DATA)),
    copula_case(ExtremeValueCopula, 2, DiscreteSpectralTail([0.7 0.3; 0.2 0.8])),
    copula_case(GumbelCopula, 2, 1.5; allowed_inference=allow_IM(2)),
    copula_case(GalambosCopula, 2, 1.0; allowed_inference=allow_IM(2)),
    copula_case(HuslerReissCopula, 2, 1.0; allowed_inference=allow_IM(2)),
    copula_case(HuslerReissCopula, 3, [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]; allowed_inference=MCopula{3}),
    copula_case(LogCopula, 2, 1.5; allowed_inference=allow_IM(2)),
    copula_case(AsymGalambosCopula, 3, 1.0, [0.4, 0.5, 0.6]; allowed_inference=allow_E(3)),
    copula_case(BC2Copula, 3, [0.3, 0.7, 0.5]),
    copula_case(CuadrasAugeCopula, 3, 0.5; allowed_inference=allow_IM(3)),
    copula_case(MOCopula, 3, [0.35, 0.55, 0.40, 0.25, 0.30, 0.45, 0.70]),
    copula_case(tEVCopula, 3, 4.0, 0.2; allowed_inference=MCopula{3}),
    copula_case(tEVCopula, 3, 4.0, [1.0 0.2 0.2; 0.2 1.0 0.2; 0.2 0.2 1.0]; allowed_inference=MCopula{3}),
    copula_case(GaussianCopula, 2, 0.3; allowed_inference=IndependentCopula{2}, numerical_atol=1e-3),
    copula_case(TCopula, 3, 5.0, [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
    copula_case(LiouvilleCopula, 3, Copulas.ClaytonGenerator(1.0), (0.8, 1.1, 1.3)),
    copula_case(FGMCopula, 3, [0.0, 0.0, 0.0, 0.4]; allowed_inference=allow_I(3)),
    copula_case(IndependentCopula, 2),
    copula_case(MCopula, 3),
    copula_case(RafteryCopula, 2, 0.5; allowed_inference=allow_IM(2)),
    copula_case(SurvivalCopula, 2, ClaytonCopula{2}(1.5), (1,)),
)
