# Test-suite orchestrator. See the developer guide's "Testing architecture"
# section for the proof obligations. Tests run from foundational public-API
# checks through mathematical and operation proofs; routing closure runs only
# after those proofs have populated its ledgers, and optional extensions run
# last in isolation.
using Aqua, Copulas, DelimitedFiles, Distributions, ForwardDiff, HCubature,
    HypothesisTests, InteractiveUtils, LinearAlgebra, LogExpFunctions,
    MvNormalCDF, QuadGK, Random, Roots, SpecialFunctions, StableRNGs,
    Statistics, StatsBase, Test, TOML

const rng = StableRNG(123)
const _TEST_RUN_STARTED = time()
const _TEST_PROGRESS_LAST = Ref(_TEST_RUN_STARTED)
const _TEST_PROGRESS_CURRENT = Ref{Union{Nothing,String}}(nothing)
const _TEST_TIMINGS = Dict{String,Any}()
function test_progress(parts...)
    # Logging must not generate a new method instance for every combination of
    # family names, symbols, dimensions, and fitting methods passed by tests.
    Base.@nospecialize parts
    next_path = join(string.(parts), " / ")
    now = time()
    if isnothing(_TEST_PROGRESS_CURRENT[])
        @info "Starting test: path=$next_path, total=$(round(now - _TEST_RUN_STARTED; digits=2))"
    else
        @info "Test progress: completed=$(_TEST_PROGRESS_CURRENT[]), elapsed=$(round(now - _TEST_PROGRESS_LAST[]; digits=2)), next=$next_path, total=$(round(now - _TEST_RUN_STARTED; digits=2))"
    end
    _TEST_PROGRESS_CURRENT[] = next_path
    _TEST_PROGRESS_LAST[] = now
end

function finish_test_progress()
    isnothing(_TEST_PROGRESS_CURRENT[]) && return
    now = time()
    @info "Test progress completed=$(_TEST_PROGRESS_CURRENT[]), elapsed=$(round(now - _TEST_PROGRESS_LAST[]; digits=2)), total=$(round(now - _TEST_RUN_STARTED; digits=2))"
    _TEST_PROGRESS_CURRENT[] = nothing
    _TEST_PROGRESS_LAST[] = now
end

function timed_include(label, path)
    started = time()
    try
        timing = @timed include(path)
        _TEST_TIMINGS[string(label)] = Dict(
            "elapsed_seconds" => timing.time,
            "compile_seconds" => timing.compile_time,
            "recompile_seconds" => timing.recompile_time,
        )
        return timing.value
    finally
        get!(_TEST_TIMINGS, string(label), Dict(
            "elapsed_seconds" => time() - started,
            "compile_seconds" => -1.0,
            "recompile_seconds" => -1.0,
        ))
    end
end

function write_test_timings()
    path = get(ENV, "COPULAS_TEST_TIMINGS", "")
    isempty(path) && return
    mkpath(dirname(abspath(path)))
    report = Dict(
        "total_seconds" => time() - _TEST_RUN_STARTED,
        "files" => _TEST_TIMINGS,
    )
    open(path, "w") do io
        TOML.print(io, report; sorted=true)
    end
end
atexit(write_test_timings)

# Shared test data and registries: declares the minimal representative models
# consumed by contracts and path tests; it contains no assertions itself.
public_symbols() = filter(!=(:Copulas), names(Copulas; all=false, imported=false))

is_absolutely_continuous(C) = Copulas.copula_measure_style(C) isa Copulas.AbsolutelyContinuousMeasure

_dependence_is_defined(measure, C::Copulas.Copula) = _dependence_is_defined(measure, Copulas.copula_measure_style(C))
_dependence_is_defined(::Union{typeof(Copulas.ι),typeof(Copulas.corentropy)}, ::Copulas.NonAbsolutelyContinuousMeasure) = false
_dependence_is_defined(::Any, ::Copulas.CopulaMeasureStyle) = true

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
const SCALAR_DEPENDENCE_MEASURES = (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ, Copulas.ι, Copulas.λₗ, Copulas.λᵤ)
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
const PROVEN_DEPENDENCE_ROUTES = Dict(measure => Set{Any}() for measure in SCALAR_DEPENDENCE_MEASURES)

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
        return _which(Copulas.DistortionFromCop, C, js, values, d)
    elseif operation === :conditional_joint
        d > 2 || return nothing
        js = (1,)
        values = (0.4,)
        is = Tuple(2:d)
        return _which(Copulas._conditional_components, C, js, values, is)
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
                     numerical_atol=1e-8, margin_atol=1e-6,
                     conditional_at=nothing)
    symbol = _public_copula_symbol(family)
    name = replace(string(symbol), r"Copula$" => "")
    typed_family = Core.apply_type(family, d)
    typed = () -> typed_family(args...; constructor_kwargs...)
    dynamic = () -> family(d, args...; constructor_kwargs...)
    call = Any[QuoteNode(typed_family)]
    isempty(constructor_kwargs) || push!(call, Expr(:parameters,
        (Expr(:kw, key, QuoteNode(value))
         for (key, value) in pairs(constructor_kwargs))...))
    append!(call, QuoteNode.(args))
    typed_expr = Expr(:call, call...)
    return (; family, symbol, name, d, args, constructor_kwargs, typed,
            typed_expr, dynamic, build=typed, numerical_atol,
            margin_atol, conditional_at)
end

include("correctness/reduction_graph.jl")
include("bestiary.jl")

const COPULA_CASES = Tuple(unique(case -> case.symbol, ALL_COPULA_CASES))
const COPULA_FIXTURES = Tuple((case=case, copula=case.build())
                              for case in ALL_COPULA_CASES)
const ROUTING_COPULA_FIXTURES = Tuple((case=case, copula=case.build())
                                      for case in ALL_COPULA_CASES)

# Public component cases derive from the same central copula bestiary. They
# live here because several earlier mathematical-oracle files consume them
# before the component operation suites themselves are included.
generator_case_key(G) = (typeof(G), G isa WilliamsonGenerator ? isinteger(G.order) : nothing)
const GENERATOR_CASES = Tuple(unique(generator_case_key,
    [fixture.copula.G for fixture in ROUTING_COPULA_FIXTURES
     if fixture.copula isa ArchimedeanCopula]))

tail_case_key((tail, d)) = (typeof(tail), d, typeof(params(tail)))
const TAIL_CASES = Tuple(unique(tail_case_key,
    [(fixture.copula.tail, length(fixture.copula))
     for fixture in ROUTING_COPULA_FIXTURES
     if fixture.copula isa ExtremeValueCopula]))


# Run cheap foundational checks first, then mathematical and operation proofs.
# Routing must remain after every test that records proven dispatch routes.
# Exact `.jl` paths are used throughout; `timed_include` does not alter them.

testfiles = (
    "Aqua.jl",
    "api/constructors.jl",
    "api/constructor_validation.jl",
    "correctness/reduction_graph_tests.jl",
    "api/public_compositions.jl",
    "api/copulas.jl",
    "api/generators.jl",
    "api/tails.jl",
    "api/univariate_distributions.jl",
    "api/sklar.jl",
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
    "correctness/nested_archimedean.jl",
    "correctness/nested_archimedean_equivalence.jl",
    "correctness/family_specialization_equivalence.jl",
    "correctness/statistical.jl",
    "operations/distribution.jl",
    "operations/measure.jl",
    "operations/sampling.jl",
    "operations/subsetting.jl",
    "operations/conditioning.jl",
    "operations/rosenblatt.jl",
    "operations/dependence.jl",
    "operations/fitting.jl",
    "operations/nataf.jl",
    "routing/branches.jl",
    "routing/dispatch.jl",
    "extensions/expectation_maximization.jl"
)

@info "Starting main tests."
try
    @testset verbose=true "Copulas.jl" begin
        @testset verbose=true "$f" for f in testfiles
            test_progress(f)
            timed_include(f, f)
        end
    end
finally
    finish_test_progress()
end
