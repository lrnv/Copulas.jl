# Test-suite orchestrator. Public contracts exercise every bestiary regime;
# expensive numerical identities are deduplicated locally by implementation
# mechanism inside the corresponding operation tests.

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

function record_timing!(label, timing)
    _TEST_TIMINGS[string(label)] = Dict(
        "elapsed_seconds" => timing.time,
        "compile_seconds" => timing.compile_time,
        "recompile_seconds" => timing.recompile_time,
    )

    return timing.value
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

function is_absolutely_continuous(C)
    Base.@nospecialize C
    return Copulas.copula_measure_style(C) isa
           Copulas.AbsolutelyContinuousMeasure
end

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

    return (
        family = family,
        d = d,
        args = args,
        constructor_kwargs = (; kwargs...),
    )
end

function constructor_type(spec)
    Base.@nospecialize spec
    return Core.apply_type(spec.family, spec.d)
end

function build_typed(spec)
    Base.@nospecialize spec

    family = constructor_type(spec)
    return family(
        spec.args...;
        spec.constructor_kwargs...,
    )
end

function build_dynamic(spec)
    Base.@nospecialize spec

    return spec.family(
        spec.d,
        spec.args...;
        spec.constructor_kwargs...,
    )
end

function typed_constructor_expr(spec)
    Base.@nospecialize spec

    call = Any[QuoteNode(constructor_type(spec))]

    if !isempty(spec.constructor_kwargs)
        parameters = Expr(:parameters)
        for (key, value) in pairs(spec.constructor_kwargs)
            push!(
                parameters.args,
                Expr(:kw, key, QuoteNode(value)),
            )
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

timed_include("setup/reduction_graph.jl", "correctness/reduction_graph.jl")
timed_include("setup/bestiary.jl", "bestiary.jl")


function build_copula_fixture(case)
    Base.@nospecialize case
    return (case = case, copula = build_typed(case))
end
const _COPULA_FIXTURES_TIMING = @timed map(build_copula_fixture, ALL_COPULA_CASES)
const COPULA_FIXTURES = _COPULA_FIXTURES_TIMING.value
record_timing!("setup/copula_fixtures", _COPULA_FIXTURES_TIMING)

# Public component cases derive from the same central copula bestiary. They
# live here because several earlier mathematical-oracle files consume them
# before the component operation suites themselves are included.
function generator_case_key(G)
    Base.@nospecialize G
    return (typeof(G), G isa WilliamsonGenerator ? isinteger(G.order) : nothing)
end
const _GENERATOR_CASES_TIMING = @timed unique(generator_case_key,
    [
        fixture.copula.G
        for fixture in COPULA_FIXTURES
        if fixture.copula isa ArchimedeanCopula
    ],
)

const GENERATOR_CASES = _GENERATOR_CASES_TIMING.value
record_timing!("setup/generator_cases", _GENERATOR_CASES_TIMING)

function tail_case_key(entry)
    Base.@nospecialize entry
    tail, d = entry
    return (typeof(tail), d, typeof(params(tail)))
end
const _TAIL_CASES_TIMING = @timed unique(tail_case_key,
    [
        (fixture.copula.tail, length(fixture.copula))
        for fixture in COPULA_FIXTURES
        if fixture.copula isa ExtremeValueCopula
    ],
)
const TAIL_CASES = _TAIL_CASES_TIMING.value
record_timing!("setup/tail_cases", _TAIL_CASES_TIMING)


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
    "correctness/behavioural_branches.jl",
    "operations/distribution.jl",
    "operations/measure.jl",
    "operations/sampling.jl",
    "operations/subsetting.jl",
    "operations/conditioning.jl",
    "operations/rosenblatt.jl",
    "operations/dependence.jl",
    "operations/fitting.jl",
    "operations/nataf.jl",
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
