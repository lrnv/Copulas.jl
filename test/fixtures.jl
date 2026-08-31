# Shared test data and registries: declares the minimal representative models
# consumed by contracts and path tests; it contains no assertions itself.
"""A public copula fixture and the mathematical contract it must satisfy."""
copula_case(name, build; numerical_atol=1e-8, margin_atol=1e-6) =
    (; name, build, numerical_atol, margin_atol)

is_absolutely_continuous(C) =
    Copulas.copula_measure_style(C) isa Copulas.AbsolutelyContinuousMeasure

const _FIXTURE_DATA = [
    0.12 0.31 0.54 0.73 0.89 0.42
    0.81 0.22 0.63 0.47 0.15 0.68
]
const _FIXTURE_DATA3 = vcat(
    _FIXTURE_DATA,
    reshape([0.24, 0.76, 0.45, 0.91, 0.33, 0.58], 1, :),
)

# One ordinary interior point per public family is intentional. Numerical
# limits and alternate algorithms belong to focused obligation tests, not
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
    copula_case("discrete-radial Archimedean", () -> ArchimedeanCopula{2}(
        WilliamsonGenerator([1.0, 2.0], [0.4, 0.6], 2))),
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
    copula_case("BC2", () -> BC2Copula{2}(0.5, 0.3)),
    copula_case("Cuadras--Auge", () -> CuadrasAugeCopula{2}(0.5)),
    copula_case("Galambos", () -> GalambosCopula{3}(1.0)),
    copula_case("Husler--Reiss", () -> HuslerReissCopula{3}(1.0)),
    copula_case("logistic EV", () -> LogCopula{3}(1.5)),
    copula_case("mixed EV", () -> MixedCopula{2}(0.5)),
    copula_case("Marshall--Olkin", () -> MOCopula{2}(0.2, 0.3, 0.4)),
    copula_case("Tawn", () -> TawnCopula{3}(2.0, [0.6, 0.7, 0.8])),
    copula_case("t-EV", () -> tEVCopula{2}(4.0, 0.5)),
    copula_case("empirical EV", () -> EmpiricalEVCopula{2}(_FIXTURE_DATA; method=:cfg, pseudo_values=false)),
    copula_case("empirical EV multivariate", () -> EmpiricalEVCopula{3}(
        _FIXTURE_DATA3; degree=1, pseudo_values=false)),
    copula_case("generic EV", () -> ExtremeValueCopula{2}(Copulas.GalambosTail(1.0))),
    copula_case("discrete spectral", () -> ExtremeValueCopula{2}(
        DiscreteSpectralTail([0.7 0.3; 0.2 0.8]))),
    # Gaussian probabilities use numerical multivariate-normal integration.
    copula_case("Gaussian", () -> GaussianCopula{3}(0.3); numerical_atol=1e-3),
    copula_case("Student", () -> TCopula{2}(4.0, [1.0 0.3; 0.3 1.0])),
    copula_case("Bernstein", () -> BernsteinCopula{2}(IndependentCopula{2}(); m=2)),
    copula_case("beta", () -> BetaCopula{2}(_FIXTURE_DATA)),
    copula_case("checkerboard", () -> CheckerboardCopula{2}(_FIXTURE_DATA; m=2)),
    # An empirical copula has discrete-uniform margins with jumps of size 1/n.
    copula_case("empirical", () -> EmpiricalCopula{2}(_FIXTURE_DATA);
        margin_atol=inv(size(_FIXTURE_DATA, 2))),
    copula_case("FGM", () -> FGMCopula{2}(0.5)),
    copula_case("independence", () -> IndependentCopula{3}()),
    copula_case("upper Frechet bound", () -> MCopula{2}()),
    copula_case("lower Frechet bound", () -> WCopula{2}()),
    copula_case("Plackett", () -> PlackettCopula{2}(2.0)),
    copula_case("Raftery", () -> RafteryCopula{3}(0.5)),
    copula_case("survival", () -> SurvivalCopula{3}(ClaytonCopula{3}(1.5), (1, 3))),
)

# Additional dimensional representations that select methods not reachable
# from the one-instance-per-family public contract above. They are consumed by
# routing and proof tests only, avoiding repetition of the full API contract.
const ROUTING_EXTRA_CASES = (
    copula_case("Gumbel bivariate", () -> GumbelCopula{2}(1.5)),
    copula_case("Galambos bivariate", () -> GalambosCopula{2}(1.0)),
    copula_case("Husler--Reiss bivariate", () -> HuslerReissCopula{2}(1.0)),
    copula_case("logistic EV bivariate", () -> LogCopula{2}(1.5)),
    copula_case("asymmetric Galambos multivariate",
        () -> AsymGalambosCopula{3}(1.0, [0.4, 0.5, 0.6])),
    copula_case("BC2 multivariate",
        () -> BC2Copula{3}([0.3, 0.7, 0.5])),
    copula_case("Cuadras--Auge multivariate",
        () -> CuadrasAugeCopula{3}(0.5)),
    copula_case("Marshall--Olkin multivariate", () -> MOCopula{3}(
        [0.35, 0.55, 0.40, 0.25, 0.30, 0.45, 0.70])),
    copula_case("t-EV multivariate", () -> tEVCopula{3}(4.0, 0.2)),
    copula_case("Gaussian bivariate", () -> GaussianCopula{2}(0.3);
                numerical_atol=1e-3),
    copula_case("Student multivariate", () -> TCopula{3}(5.0,
        [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0])),
    copula_case("Liouville multivariate", () -> LiouvilleCopula{3}(
        Copulas.ClaytonGenerator(1.0), (0.8, 1.1, 1.3))),
    copula_case("FGM multivariate", () -> FGMCopula{3}([0.0, 0.0, 0.0, 0.4])),
    copula_case("independence bivariate", () -> IndependentCopula{2}()),
    copula_case("upper Frechet multivariate", () -> MCopula{3}()),
    copula_case("Raftery bivariate", () -> RafteryCopula{2}(0.5)),
    copula_case("survival bivariate", () -> SurvivalCopula{2}(
        ClaytonCopula{2}(1.5), (1,))),
)

const ROUTING_COPULA_CASES = (COPULA_CASES..., ROUTING_EXTRA_CASES...)

# Deterministic model fixtures are constructed once and shared by the proof
# layers.  RNGs, sample buffers, conditionals, and fitted results remain local
# to each test, so this cache removes only identical constructor work and does
# not introduce order-dependent state.
const COPULA_FIXTURES = Tuple((case=case, copula=case.build()) for case in COPULA_CASES)
const ROUTING_COPULA_FIXTURES = (
    COPULA_FIXTURES...,
    ((case=case, copula=case.build()) for case in ROUTING_EXTRA_CASES)...,
)

const SCALAR_DEPENDENCE_MEASURES = (
    Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ, Copulas.ι,
    Copulas.λₗ, Copulas.λᵤ,
)
const PAIRWISE_DEPENDENCE_MEASURES = (
    (StatsBase.corkendall, 1),
    (StatsBase.corspearman, 1),
    (Copulas.corblomqvist, 1),
    (Copulas.corgini, 1),
    (Copulas.corentropy, 0),
    (Copulas.corlowertail, 1),
    (Copulas.coruppertail, 1),
)

# Proof ledger shared by the four obligation layers. A route is entered only
# after the test providing its oracle/equivalence has succeeded. The routing
# layer, which runs last, compares this ledger with every method selected by the
# public fixtures.
const PROVEN_DISPATCH_ROUTES = Dict{Symbol,Dict{Any,Set{Symbol}}}()
const PROVEN_DEPENDENCE_ROUTES = Dict(
    measure => Set{Any}() for measure in SCALAR_DEPENDENCE_MEASURES)

function _which(f, args...)
    Base.@nospecialize f args
    return which(f, Tuple{typeof.(args)...})
end

function dispatch_path(operation, C)
    Base.@nospecialize operation C
    d = length(C)
    u = fill(0.6, d)
    if operation === :cdf
        return _which(Copulas._cdf, C, u)
    elseif operation === :logpdf
        is_absolutely_continuous(C) || return nothing
        return _which(Distributions._logpdf, C, u)
    elseif operation === :sampling
        return _which(Distributions._rand!, StableRNG(51), C, zeros(d, 1))
    elseif operation === :conditioning
        js = Tuple(1:(d - 1))
        values = ntuple(_ -> 0.4, d - 1)
        return _which(Copulas.DistortionFromCop, C, js, values, d)
    elseif operation === :conditional_joint
        d > 2 || return nothing
        js = (1,)
        values = (0.4,)
        is = Tuple(2:d)
        return _which(Copulas._conditional_components, C, js, values, is)
    elseif operation === :rosenblatt
        return _which(Copulas.rosenblatt, C, reshape(u, :, 1))
    elseif operation === :inverse_rosenblatt
        is_absolutely_continuous(C) || return nothing
        return _which(Copulas.inverse_rosenblatt, C, reshape(u, :, 1))
    elseif operation === :subsetting
        dims = d == 2 ? (2, 1) : (1, d)
        return _which(Copulas.subsetdims, C, dims)
    elseif operation === :measure
        return _which(Copulas.measure, C, zeros(d), ones(d))
    end
    error("unknown dispatch operation $operation")
end

function dispatch_route_key(operation, C)
    Base.@nospecialize operation C
    method = dispatch_path(operation, C)
    isnothing(method) && return nothing
    return (method, length(C) == 2 ? :bivariate : :multivariate)
end

function prove_dispatch_route!(operation, C, source::Symbol)
    Base.@nospecialize operation C
    key = dispatch_route_key(operation, C)
    isnothing(key) && return nothing
    sources = get!(get!(PROVEN_DISPATCH_ROUTES, operation, Dict{Any,Set{Symbol}}()),
                   key, Set{Symbol}())
    push!(sources, source)
    return key
end

function dependence_route_key(measure, C)
    Base.@nospecialize measure C
    return (which(measure, Tuple{typeof(C)}),
            length(C) == 2 ? :bivariate : :multivariate)
end
function prove_dependence_route!(measure, C)
    Base.@nospecialize measure C
    return push!(PROVEN_DEPENDENCE_ROUTES[measure],
                 dependence_route_key(measure, C))
end

constructor_case(name, typed, dynamic; allowed_inference=nothing) =
    (; name, typed, dynamic, allowed_inference)

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
    constructor_case("Marshall--Olkin", () -> MOCopula{2}(0.2, 0.3, 0.4), () -> MOCopula(2, 0.2, 0.3, 0.4); allowed_inference=MOCopula{2}),
    constructor_case("Tawn", () -> TawnCopula{3}(2.0, [0.6, 0.7, 0.8]), () -> TawnCopula(3, 2.0, [0.6, 0.7, 0.8]); allowed_inference=Union{IndependentCopula,MCopula,ExtremeValueCopula{3}}),
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
    constructor_case("FGM", () -> FGMCopula{2}(0.5), () -> FGMCopula(2, 0.5); allowed_inference=Union{IndependentCopula{2},MCopula{2},WCopula{2},FGMCopula{2}}),
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
        allowed_inference=Union{NestedArchimedeanCopula,ArchimedeanCopula}),
    constructor_case("Archimax",
        () -> ArchimaxCopula{2}(Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0)),
        () -> ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5), Copulas.GalambosTail(1.0))),
    constructor_case("survival",
        () -> SurvivalCopula{3}(ClaytonCopula{3}(1.5), (1, 3)),
        () -> SurvivalCopula(3, ClaytonCopula{3}(1.5), (1, 3))),
)

# Exact public binding exercised by each constructor case.  This intentionally
# preserves aliases and repeated dimensional representations: comparing only
# concrete return types would let two distinct public spellings collapse.
const CONSTRUCTOR_SYMBOLS = (
    :AMHCopula, :BB1Copula, :BB2Copula, :BB3Copula, :BB6Copula,
    :BB7Copula, :BB8Copula, :BB9Copula, :BB10Copula,
    :ClaytonCopula, :FrankCopula, :GumbelCopula, :GumbelBarnettCopula,
    :InvGaussianCopula, :JoeCopula, :AsymGalambosCopula, :AsymLogCopula,
    :AsymMixedCopula, :BC2Copula, :CuadrasAugeCopula, :GalambosCopula,
    :HuslerReissCopula, :LogCopula, :MixedCopula, :MOCopula, :TawnCopula,
    :tEVCopula, :BB4Copula, :BB5Copula, :GaussianCopula, :TCopula,
    :IndependentCopula, :MCopula, :WCopula, :FGMCopula, :PlackettCopula,
    :RafteryCopula, :BernsteinCopula, :BetaCopula, :CheckerboardCopula,
    :EmpiricalCopula, :EmpiricalEVCopula, :EmpiricalEVCopula,
    :ArchimedeanCopula, :ExtremeValueCopula, :LiouvilleCopula,
    :NestedArchimedeanCopula, :ArchimaxCopula, :SurvivalCopula,
)

fitting_case(name, build; method=:default, model=false, kwargs=NamedTuple()) =
    (; name, build, method, model, kwargs)

fitting_statistic(::Val{:itau}, object) = SCALAR_DEPENDENCE_MEASURES[1](object)
fitting_statistic(::Val{:irho}, object) = SCALAR_DEPENDENCE_MEASURES[2](object)
fitting_statistic(::Val{:ibeta}, object) = SCALAR_DEPENDENCE_MEASURES[3](object)
fitting_statistic(::Val{:iupper}, object) = SCALAR_DEPENDENCE_MEASURES[7](object)
fitting_statistic(::Val, _) = nothing

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

const FITTING_FIXTURES = Tuple((case=case, copula=case.build())
                               for case in FITTING_CASES)

# A fitting route is the complete internal composition, not merely `_fit`.
# Generic fitting additionally depends on the example, parameter transform,
# and reconstruction methods selected for the concrete family.
const PROVEN_FITTING_ROUTES = Set{Any}()
function fitting_route_key(C, U, method)
    Base.@nospecialize C U method
    CT, d = typeof(C), length(C)
    components = Any[
        which(Copulas._available_fitting_methods, Tuple{Type{CT},Int}),
        which(Copulas._find_method, Tuple{Type{CT},Int,Symbol}),
        which(Copulas._fit, Tuple{Type{CT},typeof(U),Val{method}}),
    ]
    applicable(Copulas._example, CT, d) &&
        push!(components, which(Copulas._example, Tuple{Type{CT},Int}))
    bounded = params(C)
    topology = (keys(bounded), map(values(bounded)) do value
        value isa AbstractArray ? (typeof(value), size(value)) : typeof(value)
    end)
    component_type = C isa ArchimedeanCopula ? typeof(C.G) :
                     C isa ExtremeValueCopula ? typeof(C.tail) : nothing
    bounds = !isnothing(component_type) &&
             applicable(Copulas._θ_bounds, component_type, d) ?
             (which(Copulas._θ_bounds, Tuple{Type{component_type},Int}),
              Copulas._θ_bounds(component_type, d)) : nothing
    # Empirical EV fits reconstruct their non-parametric tail directly from
    # the observations; the generic EV forwarding method is technically
    # applicable but its parametric tail transform is not part of that route.
    # Multivariate FGM uses its dedicated constrained MLE directly; its
    # bivariate-only scalar transform is applicable by signature but rejects d>2.
    if !(C isa EmpiricalEVCopula) &&
       !(C isa FGMCopula && d != 2) && !isempty(bounded) &&
       applicable(Copulas._unbound_params, CT, d, bounded)
        unbound = Copulas._unbound_params(CT, d, bounded)
        push!(components,
              which(Copulas._unbound_params,
                    Tuple{Type{CT},Int,typeof(bounded)}))
        applicable(Copulas._rebound_params, CT, d, unbound) &&
            push!(components,
                  which(Copulas._rebound_params,
                        Tuple{Type{CT},Int,typeof(unbound)}))
        applicable(Copulas._fit_copula, CT, d, bounded, C) &&
            push!(components,
                  which(Copulas._fit_copula,
                        Tuple{Type{CT},Int,typeof(bounded),typeof(C)}))
    end
    # `which` already distinguishes genuinely dimension-specific dispatches:
    # adding the dimension itself would instead execute the same generic
    # algorithm once per representation. Parameter topology and bounds retain
    # the non-dispatch differences that affect generic reconstruction.
    return (Tuple(components), method, topology, bounds)
end
function prove_fitting_route!(C, U, method)
    Base.@nospecialize C U method
    return push!(PROVEN_FITTING_ROUTES, fitting_route_key(C, U, method))
end
