"""A public copula fixture and the mathematical contract it must satisfy."""
copula_case(name, build; kind=:continuous, rosenblatt=true) =
    (; name, build, kind, rosenblatt)

const _FIXTURE_DATA = [
    0.12 0.31 0.54 0.73 0.89 0.42
    0.81 0.22 0.63 0.47 0.15 0.68
]

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
    copula_case("empirical EV", () -> EmpiricalEVCopula{2}(_FIXTURE_DATA; degree=1, pseudo_values=false)),
    copula_case("generic EV", () -> ExtremeValueCopula{2}(Copulas.GalambosTail(1.0))),
    copula_case("Gaussian", () -> GaussianCopula{3}(0.3)),
    copula_case("Student", () -> TCopula{2}(4.0, [1.0 0.3; 0.3 1.0])),
    copula_case("Bernstein", () -> BernsteinCopula{2}(IndependentCopula{2}(); m=2)),
    copula_case("beta", () -> BetaCopula{2}(_FIXTURE_DATA)),
    copula_case("checkerboard", () -> CheckerboardCopula{2}(_FIXTURE_DATA; m=2)),
    copula_case("empirical", () -> EmpiricalCopula{2}(_FIXTURE_DATA); kind=:singular, rosenblatt=false),
    copula_case("FGM", () -> FGMCopula{2}(0.5)),
    copula_case("independence", () -> IndependentCopula{3}()),
    copula_case("upper Frechet bound", () -> MCopula{2}(); kind=:singular, rosenblatt=false),
    copula_case("lower Frechet bound", () -> WCopula{2}(); kind=:singular, rosenblatt=false),
    copula_case("Plackett", () -> PlackettCopula{2}(2.0)),
    copula_case("Raftery", () -> RafteryCopula{3}(0.5); kind=:mixed, rosenblatt=false),
    copula_case("survival", () -> SurvivalCopula{3}(ClaytonCopula{3}(1.5), (1, 3))),
)

constructor_case(name, typed, dynamic) = (; name, typed, dynamic)

const CONSTRUCTOR_CASES = (
    constructor_case("AMH", () -> AMHCopula{2}(0.5), () -> AMHCopula(2, 0.5)),
    constructor_case("Clayton", () -> ClaytonCopula{3}(1.5), () -> ClaytonCopula(3, 1.5)),
    constructor_case("Frank", () -> FrankCopula{3}(2.0), () -> FrankCopula(3, 2.0)),
    constructor_case("Gumbel", () -> GumbelCopula{3}(1.5), () -> GumbelCopula(3, 1.5)),
    constructor_case("Galambos", () -> GalambosCopula{3}(1.0), () -> GalambosCopula(3, 1.0)),
    constructor_case("Husler--Reiss", () -> HuslerReissCopula{3}(1.0), () -> HuslerReissCopula(3, 1.0)),
    constructor_case("logistic EV", () -> LogCopula{3}(1.5), () -> LogCopula(3, 1.5)),
    constructor_case("Gaussian", () -> GaussianCopula{3}(0.3), () -> GaussianCopula(3, 0.3)),
    constructor_case("independence", () -> IndependentCopula{3}(), () -> IndependentCopula(3)),
)

const FITTING_CASES = (
    (; name="Clayton inversion of tau", family=ClaytonCopula, method=:itau),
    (; name="Gaussian inversion of tau", family=GaussianCopula, method=:itau),
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
