# Shared test data and registries: declares the minimal representative models
# consumed by contracts and path tests; it contains no assertions itself.
public_symbols() = filter(!=(:Copulas), names(Copulas; all=false, imported=false))

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
# Additional dimensional representations that select methods not reachable
# from the one-instance-per-family public contract above. They are consumed by
# routing and proof tests only, avoiding repetition of the full API contract.
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

function _public_copula_symbol(family)
    symbols = [symbol for symbol in public_symbols()
               if getfield(Copulas, symbol) === family]
    return only(symbols)
end

function copula_case(family, d::Int, args...; constructor_kwargs=NamedTuple(),
                     allowed_inference=nothing, numerical_atol=1e-8,
                     margin_atol=1e-6)
    symbol = _public_copula_symbol(family)
    name = replace(string(symbol), r"Copula$" => "")
    typed_family = Core.apply_type(family, d)
    typed = () -> typed_family(args...; constructor_kwargs...)
    dynamic = () -> family(d, args...; constructor_kwargs...)
    return (; family, symbol, name, d, args, constructor_kwargs, typed,
            dynamic, build=typed, allowed_inference, numerical_atol,
            margin_atol)
end

# The single central bestiary. The first entry for each public family is its
# canonical contract/constructor representative; later entries exercise extra
# dimensions, representations, or value-dependent routes only.
const ALL_COPULA_CASES = (
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
    copula_case(AsymGalambosCopula, 2, 1.0, 0.4, 0.6;
        allowed_inference=Union{IndependentCopula,MCopula,ExtremeValueCopula{2}}),
    copula_case(AsymLogCopula, 2, 1.5, 0.4, 0.6),
    copula_case(AsymMixedCopula, 2, 0.3, 0.2;
        allowed_inference=Union{IndependentCopula{2},MixedCopula{2},AsymMixedCopula{2}}),
    copula_case(BC2Copula, 2, 0.5, 0.3; allowed_inference=BC2Copula{2}),
    copula_case(CuadrasAugeCopula, 2, 0.5),
    copula_case(GalambosCopula, 3, 1.0),
    copula_case(HuslerReissCopula, 3, 1.0),
    copula_case(LogCopula, 3, 1.5),
    copula_case(MixedCopula, 2, 0.5),
    copula_case(MOCopula, 2, 0.2, 0.3, 0.4; allowed_inference=MOCopula{2}),
    copula_case(TawnCopula, 3, 2.0, [0.6, 0.7, 0.8];
        allowed_inference=Union{IndependentCopula,MCopula,ExtremeValueCopula{3}}),
    copula_case(tEVCopula, 2, 4.0, 0.5),
    copula_case(BB4Copula, 2, 1.5, 1.0),
    copula_case(BB5Copula, 2, 1.5, 1.0),
    copula_case(GaussianCopula, 3, 0.3;
        allowed_inference=IndependentCopula, numerical_atol=1e-3),
    copula_case(TCopula, 2, 4.0, [1.0 0.3; 0.3 1.0]),
    copula_case(IndependentCopula, 3),
    copula_case(MCopula, 2),
    copula_case(WCopula, 2),
    copula_case(FGMCopula, 2, 0.5;
        allowed_inference=Union{IndependentCopula{2},MCopula{2},WCopula{2},FGMCopula{2}}),
    copula_case(PlackettCopula, 2, 2.0),
    copula_case(RafteryCopula, 3, 0.5),
    copula_case(BernsteinCopula, 2, IndependentCopula{2}();
        constructor_kwargs=(; m=2)),
    copula_case(BetaCopula, 2, _FIXTURE_DATA),
    copula_case(CheckerboardCopula, 2, _FIXTURE_DATA;
        constructor_kwargs=(; m=2)),
    copula_case(EmpiricalCopula, 2, _FIXTURE_DATA;
        margin_atol=inv(size(_FIXTURE_DATA, 2))),
    copula_case(EmpiricalEVCopula, 2, _FIXTURE_DATA;
        constructor_kwargs=(; method=:cfg, pseudo_values=false)),
    copula_case(ArchimedeanCopula, 2, Copulas.ClaytonGenerator(1.5)),
    copula_case(ExtremeValueCopula, 2, Copulas.GalambosTail(1.0)),
    copula_case(LiouvilleCopula, 2, Copulas.ClaytonGenerator(1.0), (1.0, 2.0);
        allowed_inference=ArchimedeanCopula),
    copula_case(NestedArchimedeanCopula, 4, Copulas.ClaytonGenerator(1.0);
        constructor_kwargs=(; leaves=[1, 2], children=[ClaytonCopula{2}(2.0)]),
        allowed_inference=Union{NestedArchimedeanCopula,ArchimedeanCopula}),
    copula_case(ArchimaxCopula, 2, Copulas.ClaytonGenerator(1.5),
        Copulas.GalambosTail(1.0)),
    copula_case(SurvivalCopula, 3, ClaytonCopula{3}(1.5), (1, 3)),

    # Additional dispatch representatives.
    copula_case(EmpiricalEVCopula, 3, _FIXTURE_DATA3;
        constructor_kwargs=(; degree=1, pseudo_values=false)),
    copula_case(ArchimedeanCopula, 2, Copulas.FrailtyGenerator(Exponential())),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Dirac(1.0), 2.0)),
    copula_case(ArchimedeanCopula, 2, WilliamsonGenerator(Dirac(1.0), 2.5)),
    copula_case(ArchimedeanCopula, 2, EmpiricalGenerator(_FIXTURE_DATA)),
    copula_case(GumbelCopula, 2, 1.5),
    copula_case(GalambosCopula, 2, 1.0),
    copula_case(HuslerReissCopula, 2, 1.0),
    copula_case(HuslerReissCopula, 3,
        [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]),
    copula_case(LogCopula, 2, 1.5),
    copula_case(AsymGalambosCopula, 3, 1.0, [0.4, 0.5, 0.6]),
    copula_case(BC2Copula, 3, [0.3, 0.7, 0.5]),
    copula_case(CuadrasAugeCopula, 3, 0.5),
    copula_case(MOCopula, 3,
        [0.35, 0.55, 0.40, 0.25, 0.30, 0.45, 0.70]),
    copula_case(tEVCopula, 3, 4.0, 0.2),
    copula_case(GaussianCopula, 2, 0.3; numerical_atol=1e-3),
    copula_case(TCopula, 3, 5.0,
        [1.0 0.4 0.2; 0.4 1.0 0.3; 0.2 0.3 1.0]),
    copula_case(LiouvilleCopula, 3, Copulas.ClaytonGenerator(1.0),
        (0.8, 1.1, 1.3)),
    copula_case(FGMCopula, 3, [0.0, 0.0, 0.0, 0.4]),
    copula_case(IndependentCopula, 2),
    copula_case(MCopula, 3),
    copula_case(RafteryCopula, 2, 0.5),
    copula_case(SurvivalCopula, 2, ClaytonCopula{2}(1.5), (1,)),
)

const COPULA_CASES = Tuple(unique(case -> case.symbol, ALL_COPULA_CASES))
const CONSTRUCTOR_CASES = COPULA_CASES
const ROUTING_COPULA_CASES = ALL_COPULA_CASES
const COPULA_FIXTURES = Tuple((case=case, copula=case.build())
                              for case in COPULA_CASES)
const ROUTING_COPULA_FIXTURES = Tuple((case=case, copula=case.build())
                                      for case in ROUTING_COPULA_CASES)

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
