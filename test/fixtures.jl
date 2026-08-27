# Shared test data and registries: declares the minimal representative models
# consumed by contracts and path tests; it contains no assertions itself.
"""A public copula fixture and the mathematical contract it must satisfy."""
copula_case(name, build; kind=:continuous, rosenblatt=true,
            numerical_atol=1e-8, margin_atol=1e-6) =
    (; name, build, kind, rosenblatt, numerical_atol, margin_atol)

const _FIXTURE_DATA = [
    0.12 0.31 0.54 0.73 0.89 0.42
    0.81 0.22 0.63 0.47 0.15 0.68
]
const _FIXTURE_DATA3 = vcat(
    _FIXTURE_DATA,
    reshape([0.24, 0.76, 0.45, 0.91, 0.33, 0.58], 1, :),
)

# One ordinary interior point per public family is intentional. Numerical
# limits and alternate algorithms belong to path and family regressions, not
# to the public contract matrix.
const COPULA_CASES = (
    copula_case("AMH", () -> AMHCopula{2}(0.5)),
    copula_case("BB1", () -> BB1Copula{2}(1.2, 1.5)),
    copula_case("BB2", () -> BB2Copula{2}(1.2, 0.5)),
    copula_case("BB3", () -> BB3Copula{2}(2.0, 1.5)),
    copula_case("BB6", () -> BB6Copula{2}(1.2, 1.6)),
    copula_case("BB7", () -> BB7Copula{2}(1.2, 1.6)),
    copula_case("BB8", () -> BB8Copula{2}(1.2, 0.4)),
    copula_case("BB9", () -> BB9Copula{2}(1.5, 2.4)),
    copula_case("BB10", () -> BB10Copula{2}(1.5, 0.7)),
    copula_case("Clayton", () -> ClaytonCopula{3}(1.5)),
    copula_case("Frank", () -> FrankCopula{3}(2.0)),
    copula_case("Gumbel", () -> GumbelCopula{3}(1.5)),
    copula_case("Gumbel--Barnett", () -> GumbelBarnettCopula{2}(0.5)),
    copula_case("inverse Gaussian", () -> InvGaussianCopula{2}(0.5)),
    copula_case("Joe", () -> JoeCopula{2}(1.5)),
    copula_case("generic Archimedean", () -> ArchimedeanCopula{2}(Copulas.ClaytonGenerator(1.5))),
    copula_case("nested Archimedean", () -> NestedArchimedeanCopula{4}(
        Copulas.ClaytonGenerator(1.0); leaves=[1, 2],
        children=[ClaytonCopula{2}(2.0)])),
    copula_case("Liouville", () -> LiouvilleCopula{2}(
        Copulas.ClaytonGenerator(1.0), (1.0, 2.0))),
    copula_case("Archimax", () -> ArchimaxCopula{2}(
        Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0))),
    copula_case("BB4", () -> BB4Copula{2}(1.5, 1.0)),
    copula_case("BB5", () -> BB5Copula{2}(1.5, 1.0)),
    copula_case("asymmetric Galambos", () -> AsymGalambosCopula{2}(1.0, 0.4, 0.6)),
    copula_case("asymmetric logistic", () -> AsymLogCopula{2}(1.5, 0.4, 0.6)),
    copula_case("asymmetric mixed", () -> AsymMixedCopula{2}(0.3, 0.2)),
    copula_case("BC2", () -> BC2Copula{2}(0.5, 0.3); kind=:mixed, rosenblatt=false),
    copula_case("Cuadras--Auge", () -> CuadrasAugeCopula{2}(0.5); kind=:mixed, rosenblatt=false),
    copula_case("Galambos", () -> GalambosCopula{3}(1.0)),
    copula_case("Husler--Reiss", () -> HuslerReissCopula{3}(1.0)),
    copula_case("logistic EV", () -> LogCopula{3}(1.5)),
    copula_case("mixed EV", () -> MixedCopula{2}(0.5)),
    copula_case("Marshall--Olkin", () -> MOCopula{2}(0.2, 0.3, 0.4); kind=:mixed, rosenblatt=false),
    copula_case("Tawn", () -> TawnCopula{3}(2.0, [0.6, 0.7, 0.8])),
    copula_case("t-EV", () -> tEVCopula{2}(4.0, 0.5)),
    copula_case("empirical EV", () -> EmpiricalEVCopula{2}(_FIXTURE_DATA; method=:cfg, pseudo_values=false)),
    copula_case("empirical EV multivariate", () -> EmpiricalEVCopula{3}(
        _FIXTURE_DATA3; degree=1, pseudo_values=false);
        kind=:singular, rosenblatt=false),
    copula_case("generic EV", () -> ExtremeValueCopula{2}(Copulas.GalambosTail(1.0))),
    copula_case("discrete spectral", () -> ExtremeValueCopula{2}(
        DiscreteSpectralTail([0.7 0.3; 0.2 0.8]));
        kind=:singular, rosenblatt=false),
    # Gaussian probabilities use numerical multivariate-normal integration.
    copula_case("Gaussian", () -> GaussianCopula{3}(0.3); numerical_atol=1e-3),
    copula_case("Student", () -> TCopula{2}(4.0, [1.0 0.3; 0.3 1.0])),
    copula_case("Bernstein", () -> BernsteinCopula{2}(IndependentCopula{2}(); m=2)),
    copula_case("beta", () -> BetaCopula{2}(_FIXTURE_DATA)),
    copula_case("checkerboard", () -> CheckerboardCopula{2}(_FIXTURE_DATA; m=2)),
    # An empirical copula has discrete-uniform margins with jumps of size 1/n.
    copula_case("empirical", () -> EmpiricalCopula{2}(_FIXTURE_DATA);
        kind=:singular, rosenblatt=false, margin_atol=inv(size(_FIXTURE_DATA, 2))),
    copula_case("FGM", () -> FGMCopula{2}(0.5)),
    copula_case("independence", () -> IndependentCopula{3}()),
    copula_case("upper Frechet bound", () -> MCopula{2}(); kind=:singular, rosenblatt=false),
    copula_case("lower Frechet bound", () -> WCopula{2}(); kind=:singular, rosenblatt=false),
    copula_case("Plackett", () -> PlackettCopula{2}(2.0)),
    copula_case("Raftery", () -> RafteryCopula{3}(0.5); kind=:mixed, rosenblatt=false),
    copula_case("survival", () -> SurvivalCopula{3}(ClaytonCopula{3}(1.5), (1, 3))),
)

constructor_case(name, typed, dynamic;
                 allowed_inference=nothing, reconstruct=true) =
    (; name, typed, dynamic, allowed_inference, reconstruct)

const CONSTRUCTOR_CASES = (
    constructor_case("AMH", () -> AMHCopula{2}(0.5), () -> AMHCopula(2, 0.5)),
    constructor_case("BB1", () -> BB1Copula{2}(1.2, 1.5), () -> BB1Copula(2, 1.2, 1.5)),
    constructor_case("BB2", () -> BB2Copula{2}(1.2, 0.5), () -> BB2Copula(2, 1.2, 0.5)),
    constructor_case("BB3", () -> BB3Copula{2}(2.0, 1.5), () -> BB3Copula(2, 2.0, 1.5)),
    constructor_case("BB6", () -> BB6Copula{2}(1.2, 1.6), () -> BB6Copula(2, 1.2, 1.6)),
    constructor_case("BB7", () -> BB7Copula{2}(1.2, 1.6), () -> BB7Copula(2, 1.2, 1.6)),
    constructor_case("BB8", () -> BB8Copula{2}(1.2, 0.4), () -> BB8Copula(2, 1.2, 0.4)),
    constructor_case("BB9", () -> BB9Copula{2}(1.5, 2.4), () -> BB9Copula(2, 1.5, 2.4)),
    constructor_case("BB10", () -> BB10Copula{2}(1.5, 0.7), () -> BB10Copula(2, 1.5, 0.7)),
    constructor_case("Clayton", () -> ClaytonCopula{3}(1.5), () -> ClaytonCopula(3, 1.5)),
    constructor_case("Frank", () -> FrankCopula{3}(2.0), () -> FrankCopula(3, 2.0)),
    constructor_case("Gumbel", () -> GumbelCopula{3}(1.5), () -> GumbelCopula(3, 1.5)),
    constructor_case("Gumbel--Barnett", () -> GumbelBarnettCopula{2}(0.5), () -> GumbelBarnettCopula(2, 0.5)),
    constructor_case("inverse Gaussian", () -> InvGaussianCopula{2}(0.5), () -> InvGaussianCopula(2, 0.5)),
    constructor_case("Joe", () -> JoeCopula{2}(1.5), () -> JoeCopula(2, 1.5)),
    # Its value-dependent boundary simplifications intentionally infer a small
    # union rather than one concrete family.
    constructor_case("asymmetric Galambos",
        () -> AsymGalambosCopula{2}(1.0, 0.4, 0.6),
        () -> AsymGalambosCopula(2, 1.0, 0.4, 0.6);
        allowed_inference=Union{
            IndependentCopula,
            MCopula,
            ExtremeValueCopula{2},
        }),
    constructor_case("asymmetric logistic", () -> AsymLogCopula{2}(1.5, 0.4, 0.6), () -> AsymLogCopula(2, 1.5, 0.4, 0.6)),
    constructor_case("asymmetric mixed",
        () -> AsymMixedCopula{2}(0.3, 0.2),
        () -> AsymMixedCopula(2, 0.3, 0.2);
        allowed_inference=Union{
            IndependentCopula{2}, MixedCopula{2}, AsymMixedCopula{2},
        }),
    constructor_case("BC2",
        () -> BC2Copula{2}(0.5, 0.3),
        () -> BC2Copula(2, 0.5, 0.3);
        allowed_inference=BC2Copula{2}),
    constructor_case("Cuadras--Auge", () -> CuadrasAugeCopula{2}(0.5), () -> CuadrasAugeCopula(2, 0.5)),
    constructor_case("Galambos", () -> GalambosCopula{3}(1.0), () -> GalambosCopula(3, 1.0)),
    constructor_case("Husler--Reiss", () -> HuslerReissCopula{3}(1.0), () -> HuslerReissCopula(3, 1.0)),
    constructor_case("logistic EV", () -> LogCopula{3}(1.5), () -> LogCopula(3, 1.5)),
    constructor_case("mixed EV", () -> MixedCopula{2}(0.5), () -> MixedCopula(2, 0.5)),
    constructor_case("Marshall--Olkin", () -> MOCopula{2}(0.2, 0.3, 0.4), () -> MOCopula(2, 0.2, 0.3, 0.4)),
    constructor_case("Tawn", () -> TawnCopula{3}(2.0, [0.6, 0.7, 0.8]), () -> TawnCopula(3, 2.0, [0.6, 0.7, 0.8])),
    constructor_case("t-EV", () -> tEVCopula{2}(4.0, 0.5), () -> tEVCopula(2, 4.0, 0.5)),
    constructor_case("BB4", () -> BB4Copula{2}(1.5, 1.0), () -> BB4Copula(2, 1.5, 1.0)),
    constructor_case("BB5", () -> BB5Copula{2}(1.5, 1.0), () -> BB5Copula(2, 1.5, 1.0)),
    # The scalar-correlation constructor intentionally infers a small union because
    # its independence boundary returns IndependentCopula.
    constructor_case("Gaussian", () -> GaussianCopula{3}(0.3),
        () -> GaussianCopula(3, 0.3); allowed_inference=IndependentCopula),
    constructor_case("Student", () -> TCopula{2}(4.0, [1.0 0.3; 0.3 1.0]), () -> TCopula(2, 4.0, [1.0 0.3; 0.3 1.0])),
    constructor_case("independence", () -> IndependentCopula{3}(), () -> IndependentCopula(3)),
    constructor_case("upper Frechet", () -> MCopula{3}(), () -> MCopula(3)),
    constructor_case("lower Frechet", () -> WCopula{2}(), () -> WCopula(2)),
    constructor_case("FGM", () -> FGMCopula{2}(0.5), () -> FGMCopula(2, 0.5)),
    constructor_case("Plackett", () -> PlackettCopula{2}(2.0), () -> PlackettCopula(2, 2.0)),
    constructor_case("Raftery", () -> RafteryCopula{3}(0.5), () -> RafteryCopula(3, 0.5)),
    constructor_case("Bernstein", () -> BernsteinCopula{2}(IndependentCopula{2}(); m=2), () -> BernsteinCopula(2, IndependentCopula{2}(); m=2)),
    constructor_case("beta", () -> BetaCopula{2}(_FIXTURE_DATA), () -> BetaCopula(2, _FIXTURE_DATA)),
    constructor_case("checkerboard", () -> CheckerboardCopula{2}(_FIXTURE_DATA; m=2), () -> CheckerboardCopula(2, _FIXTURE_DATA; m=2)),
    constructor_case("empirical", () -> EmpiricalCopula{2}(_FIXTURE_DATA), () -> EmpiricalCopula(2, _FIXTURE_DATA)),
    constructor_case("empirical EV", () -> EmpiricalEVCopula{2}(_FIXTURE_DATA; method=:cfg, pseudo_values=false), () -> EmpiricalEVCopula(2, _FIXTURE_DATA; method=:cfg, pseudo_values=false)),
    constructor_case("empirical EV multivariate",
        () -> EmpiricalEVCopula{3}(_FIXTURE_DATA3; degree=1, pseudo_values=false),
        () -> EmpiricalEVCopula(3, _FIXTURE_DATA3; degree=1, pseudo_values=false)),
    constructor_case("generic Archimedean",
        () -> ArchimedeanCopula{2}(Copulas.ClaytonGenerator(1.5)),
        () -> ArchimedeanCopula(2, Copulas.ClaytonGenerator(1.5))),
    constructor_case("generic extreme value",
        () -> ExtremeValueCopula{2}(Copulas.GalambosTail(1.0)),
        () -> ExtremeValueCopula(2, Copulas.GalambosTail(1.0))),
    # The all-one Dirichlet boundary is exactly Archimedean.
    constructor_case("Liouville",
        () -> LiouvilleCopula{2}(Copulas.ClaytonGenerator(1.0), (1.0, 2.0)),
        () -> LiouvilleCopula(2, Copulas.ClaytonGenerator(1.0), (1.0, 2.0));
        allowed_inference=ArchimedeanCopula),
    # An empty children collection produces the flat Archimedean fast path.
    constructor_case("nested Archimedean",
        () -> NestedArchimedeanCopula{4}(Copulas.ClaytonGenerator(1.0);
            leaves=[1, 2], children=[ClaytonCopula{2}(2.0)]),
        () -> NestedArchimedeanCopula(4, Copulas.ClaytonGenerator(1.0);
            leaves=[1, 2], children=[ClaytonCopula{2}(2.0)]);
        allowed_inference=ArchimedeanCopula),
    constructor_case("Archimax",
        () -> ArchimaxCopula{2}(Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0)),
        () -> ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0))),
    constructor_case("survival",
        () -> SurvivalCopula{3}(ClaytonCopula{3}(1.5), (1, 3)),
        () -> SurvivalCopula(3, ClaytonCopula{3}(1.5), (1, 3))),
)

fitting_case(name, build; method=:default, model=false, kwargs=NamedTuple()) =
    (; name, build, method, model, kwargs)

const FITTING_CASES = (
    fitting_case("AMH", () -> AMHCopula{2}(0.5)),
    fitting_case("BB1", () -> BB1Copula{2}(1.2, 1.5)),
    fitting_case("BB2", () -> BB2Copula{2}(1.2, 0.5)),
    fitting_case("BB3", () -> BB3Copula{2}(2.0, 1.5)),
    fitting_case("BB6", () -> BB6Copula{2}(1.2, 1.6)),
    fitting_case("BB7", () -> BB7Copula{2}(1.2, 1.6)),
    fitting_case("BB8", () -> BB8Copula{2}(1.2, 0.4)),
    fitting_case("BB9", () -> BB9Copula{2}(1.5, 2.4)),
    fitting_case("BB10", () -> BB10Copula{2}(1.5, 0.7)),
    fitting_case("Clayton", () -> ClaytonCopula{2}(1.5); method=:itau, model=true),
    fitting_case("Frank", () -> FrankCopula{2}(2.0); method=:itau),
    fitting_case("Gumbel", () -> GumbelCopula{2}(1.5); method=:itau),
    fitting_case("Gumbel--Barnett", () -> GumbelBarnettCopula{2}(0.5); method=:itau),
    fitting_case("inverse Gaussian", () -> InvGaussianCopula{2}(0.5); method=:itau),
    fitting_case("Joe", () -> JoeCopula{2}(1.5); method=:itau),
    fitting_case("Archimax", () -> ArchimaxCopula{2}(
        Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0))),
    fitting_case("BB4", () -> BB4Copula{2}(1.5, 1.0)),
    fitting_case("BB5", () -> BB5Copula{2}(1.5, 1.0)),
    fitting_case("asymmetric Galambos", () -> AsymGalambosCopula{2}(1.0, 0.4, 0.6)),
    fitting_case("asymmetric logistic", () -> AsymLogCopula{2}(1.5, 0.4, 0.6)),
    fitting_case("asymmetric mixed", () -> AsymMixedCopula{2}(0.3, 0.2)),
    fitting_case("BC2", () -> BC2Copula{2}(0.5, 0.3)),
    fitting_case("Cuadras--Auge", () -> CuadrasAugeCopula{2}(0.5); method=:itau),
    fitting_case("Galambos", () -> GalambosCopula{2}(1.0); method=:itau),
    fitting_case("Husler--Reiss", () -> HuslerReissCopula{2}(1.0); method=:itau),
    fitting_case("logistic EV", () -> LogCopula{2}(1.5); method=:itau),
    fitting_case("mixed EV", () -> MixedCopula{2}(0.5); method=:itau),
    fitting_case("Marshall--Olkin", () -> MOCopula{2}(0.2, 0.3, 0.4)),
    fitting_case("Tawn", () -> TawnCopula{3}(2.0, [0.6, 0.7, 0.8])),
    fitting_case("t-EV", () -> tEVCopula{2}(4.0, 0.5)),
    fitting_case("empirical EV", () -> EmpiricalEVCopula{2}(
        _FIXTURE_DATA; method=:cfg, pseudo_values=false); method=:cfg),
    fitting_case("empirical EV multivariate", () -> EmpiricalEVCopula{3}(
        _FIXTURE_DATA3; degree=1, pseudo_values=false); method=:cfg,
        kwargs=(degree=1,)),
    fitting_case("Gaussian", () -> GaussianCopula{2}(0.3); method=:itau, model=true),
    fitting_case("Student", () -> TCopula{2}(4.0, [1.0 0.3; 0.3 1.0])),
    fitting_case("Bernstein", () -> BernsteinCopula{2}(
        IndependentCopula{2}(); m=2); method=:bernstein, kwargs=(m=2,)),
    fitting_case("beta", () -> BetaCopula{2}(_FIXTURE_DATA); method=:beta),
    fitting_case("checkerboard", () -> CheckerboardCopula{2}(
        _FIXTURE_DATA; m=2); method=:exact, kwargs=(m=2,)),
    fitting_case("empirical", () -> EmpiricalCopula{2}(
        _FIXTURE_DATA); method=:deheuvels),
    fitting_case("FGM", () -> FGMCopula{2}(0.5); method=:itau),
    fitting_case("independence", () -> IndependentCopula{2}(); method=:mle),
    fitting_case("upper Frechet", () -> MCopula{2}(); method=:mle),
    fitting_case("lower Frechet", () -> WCopula{2}(); method=:mle),
    fitting_case("Plackett", () -> PlackettCopula{2}(2.0); method=:itau),
    fitting_case("Raftery", () -> RafteryCopula{2}(0.5); method=:itau),
    fitting_case("survival", () -> SurvivalCopula{2}(
        ClaytonCopula{2}(1.5), (1,)); method=:itau),
)

const PATH_CASES = (
    generic_cdf=FGMCopula{2}(0.4),
    archimedean_frailty=FrankCopula{3}(2.0),
    matrix_sampler=ClaytonCopula{5}(1.5),
    biv_ev_distortion=GalambosCopula{2}(1.0),
    generic_condition=RafteryCopula{2}(0.5),
    singular_condition=MCopula{2}(),
    numerical_ev=HuslerReissCopula{3}(1.0),
    fractional_williamson=LiouvilleCopula{2}(
        Copulas.ClaytonGenerator(1.0), (0.75, 1.25)),
)
