# Test-suite orchestrator. Public contracts exercise every bestiary regime;
# expensive numerical identities are deduplicated locally by implementation
# mechanism inside the corresponding operation tests.

using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test

const rng = StableRNG(123)

######## Helpers for the actual tests
function is_absolutely_continuous(C)
    Base.@nospecialize C
    return Copulas.copula_measure_style(C) isa Copulas.AbsolutelyContinuousMeasure
end

const _FIXTURE_DATA = [
    0.12 0.31 0.54 0.73 0.89 0.42
    0.81 0.22 0.63 0.47 0.15 0.68
]
const _FIXTURE_DATA3 = vcat(
    _FIXTURE_DATA,
    reshape([0.24, 0.76, 0.45, 0.91, 0.33, 0.58], 1, :),
)

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
    elseif operation === :conditioning
        js = Tuple(1:(d - 1))
        values = ntuple(_ -> 0.4, d - 1)
        return _which(Copulas.distortion, C, js, values, d)
    elseif operation === :conditional_joint
        d > 2 || return nothing
        js = (1,)
        values = (0.4,)
        return _which(Copulas.conditional_copula, C, js, values)
    end
    error("unknown dispatch operation $operation")
end
function dispatch_route_key(operation, C)
    Base.@nospecialize operation C
    method = dispatch_path(operation, C)
    isnothing(method) && return nothing
    return (method, length(C) == 2 ? :bivariate : :multivariate)
end

# Shared test data and registries: declares the minimal representative models
# consumed by contracts and path tests; it contains no assertions itself.
public_symbols() = filter(!=(:Copulas), names(Copulas; all=false, imported=false))
const _PUBLIC_SYMBOLS = public_symbols()
function _public_copula_symbol(family)
    Base.@nospecialize family

    found = nothing
    for symbol in _PUBLIC_SYMBOLS
        getfield(Copulas, symbol) === family || continue
        isnothing(found) || error("multiple public symbols for $family")
        found = symbol
    end

    isnothing(found) && error("no public symbol for $family")
    return found
end

function constructor_spec(family, d::Int, args...; kwargs...)
    Base.@nospecialize family args
    return (family = family, d = d, args = args, constructor_kwargs = (; kwargs...))
end
function constructor_type(spec)
    Base.@nospecialize spec
    return Core.apply_type(spec.family, spec.d)
end
function build_typed(spec)
    Base.@nospecialize spec
    family = constructor_type(spec)
    return family(spec.args...; spec.constructor_kwargs...)
end
function build_dynamic(spec)
    Base.@nospecialize spec
    return spec.family(spec.d, spec.args...; spec.constructor_kwargs...)
end
function typed_constructor_expr(spec)
    Base.@nospecialize spec
    call = Any[QuoteNode(constructor_type(spec))]
    if !isempty(spec.constructor_kwargs)
        parameters = Expr(:parameters)
        for (key, value) in pairs(spec.constructor_kwargs)
            push!(parameters.args, Expr(:kw, key, QuoteNode(value)))
        end
        push!(call, parameters)
    end
    for arg in spec.args
        push!(call, QuoteNode(arg))
    end
    return Expr(:call, call...)
end
function copula_case(
    family,
    d::Int,
    args...;
    constructor_kwargs=NamedTuple(),
    numerical_atol=1e-8,
    margin_atol=1e-6,
    conditional_at=nothing,
)
    Base.@nospecialize family args
    symbol = _public_copula_symbol(family)
    name = replace(string(symbol), r"Copula$" => "")
    return (
        family = family,
        symbol = symbol,
        name = name,
        d = d,
        args = args,
        constructor_kwargs = constructor_kwargs,
        numerical_atol = numerical_atol,
        margin_atol = margin_atol,
        conditional_at = conditional_at,
    )
end

include("reduction_bestiary.jl")
include("bestiary.jl")

function build_copula_fixture(case)
    Base.@nospecialize case
    return (case = case, copula = build_typed(case))
end
const COPULA_FIXTURES = map(build_copula_fixture, ALL_COPULA_CASES)

# Public component cases derive from the same central copula bestiary. They
# live here because several earlier mathematical-oracle files consume them
# before the component operation suites themselves are included.
function generator_case_key(G)
    Base.@nospecialize G
    return (typeof(G), G isa WilliamsonGenerator ? isinteger(G.order) : nothing)
end
const GENERATOR_CASES = unique(generator_case_key,
    [
        fixture.copula.G
        for fixture in COPULA_FIXTURES
        if fixture.copula isa ArchimedeanCopula
    ],
)

function tail_case_key(entry)
    Base.@nospecialize entry
    tail, d = entry
    return (typeof(tail), d)
end
const TAIL_CASES = unique(tail_case_key,
    [
        (fixture.copula.tail, length(fixture.copula))
        for fixture in COPULA_FIXTURES
        if fixture.copula isa ExtremeValueCopula
    ],
)

# Conditioning and Rosenblatt both instantiate highly parametric conditional
# distributions. Exercising every concrete bestiary type recompiles the same
# implementation routes dozens of times, while the family mathematics is
# already covered by the distribution/correctness suites. Select one fixture
# per actual conditioning kernel before including these operation proofs.
function _conditioning_compile_route_key(fixture)
    Base.@nospecialize fixture
    C = fixture.copula
    d = length(C)
    js = Tuple(1:(d - 1))
    values = ntuple(_ -> 0.4, d - 1)
    is = (d,)
    uis = (0.5,)

    distortion_method = _which(Copulas.distortion, C, js, values, d)
    partial_method = _which(Copulas._partial_cdf, C, is, js, uis, values)
    joint_method = d > 2 ? dispatch_path(:conditional_joint, C) : nothing
    dimension_class = d == 2 ? :bivariate : d == 3 ? :trivariate : :higher

    # EV tails share the outer ExtremeValueCopula conditioning route but have
    # distinct STDF-partial kernels. Keep one representative of every tail type;
    # in particular this retains Hüsler-Reiss and extremal-t coverage.
    tail_type = C isa ExtremeValueCopula ? Base.typename(typeof(C.tail)).wrapper : nothing
    return (distortion_method, partial_method, joint_method, dimension_class, tail_type)
end

function _rosenblatt_compile_route_key(fixture)
    Base.@nospecialize fixture
    C = fixture.copula
    is_absolutely_continuous(C) || return (:non_ac,)
    d = length(C)
    U = fill(0.5, d, 1)
    forward = _which(Copulas.rosenblatt, C, U)
    inverse = _which(Copulas.inverse_rosenblatt, C, U)
    return (forward, inverse, _conditioning_compile_route_key(fixture))
end

function _operation_fixture_subset(f)
    if f == "operations/conditioning.jl"
        return unique(_conditioning_compile_route_key, COPULA_FIXTURES)
    elseif f == "operations/rosenblatt.jl"
        return unique(_rosenblatt_compile_route_key, COPULA_FIXTURES)
    end
    return nothing
end

function _include_testfile(f)
    subset = _operation_fixture_subset(f)
    subset === nothing && return Base.include(@__MODULE__, joinpath(@__DIR__, f))

    original = copy(COPULA_FIXTURES)
    empty!(COPULA_FIXTURES)
    append!(COPULA_FIXTURES, subset)
    @info "Deduplicated operation fixtures" file=f selected=length(subset) total=length(original)
    try
        return Base.include(@__MODULE__, joinpath(@__DIR__, f))
    finally
        empty!(COPULA_FIXTURES)
        append!(COPULA_FIXTURES, original)
    end
end

# Paramorph 0.0.3 migration checkpoint: run the complete suite and let each
# operation proof deduplicate only the concrete compilation routes it exercises.
testfiles = (
    "Aqua.jl",
    "api/constructors.jl",
    "api/constructor_validation.jl",
    "api/liebscher_constructor_validation.jl",
    "api/public_exception_taxonomy.jl",
    "correctness/reduction_graph.jl",
    "api/public_compositions.jl",
    "api/copulas.jl",
    "api/generators.jl",
    "api/generator_numeric_fallback.jl",
    "api/tails.jl",
    "api/univariate_distributions.jl",
    "api/sklar.jl",
    "api/sklar_discrete_mixed.jl",
    "api/utilities.jl",
    "correctness/numerical.jl",
    "correctness/williamson.jl",
    "correctness/mathematical.jl",
    "correctness/archimedean.jl",
    "correctness/elliptical.jl",
    "correctness/bivariate_families.jl",
    "correctness/extreme_value.jl",
    "correctness/extreme_value_quantiles.jl",
    "correctness/extreme_value_equivalence.jl",
    "correctness/liouville.jl",
    "correctness/liebscher.jl",
    "correctness/nested_archimedean.jl",
    "correctness/nested_archimedean_equivalence.jl",
    "correctness/family_specialization_equivalence.jl",
    "correctness/statistical.jl",
    "correctness/dependence_inverses.jl",
    "correctness/behavioural_branches.jl",
    "operations/distribution.jl",
    "operations/measure.jl",
    "operations/sampling.jl",
    "operations/subsetting.jl",
    "operations/conditioning.jl",
    "operations/conditioning_numeric_types.jl",
    "operations/rosenblatt.jl",
    "operations/dependence.jl",
    "operations/dependence_dimension_validation.jl",
    "operations/fitting.jl",
    "operations/liebscher_fitting.jl",
    "operations/parameter_coefficients.jl",
    "operations/sklar_fitting_validation.jl",
    "operations/nested_fitting_validation.jl",
    "operations/inference.jl",
    "operations/fitting_selection.jl",
    "operations/hypothesis_testing.jl",
    "operations/nataf.jl",
    "extensions/expectation_maximization.jl",
    "extensions/partitioned_distributions.jl"
)

@testset verbose=true "Copulas.jl" begin
    nfiles = length(testfiles)
    for (i, f) in enumerate(testfiles)
        @info "Running tests [$i/$nfiles]" file=f
        @testset "$f" begin
            _include_testfile(f)
        end
    end
end