# Public-API contract: applies the universal distribution, sampling, subsetting,
# conditioning, Rosenblatt, and dependence-measure behavior to every family.
struct CopulaContractContext{TU,TM}
    u::TU
    U::TM
end

function copula_contract_context(C, seed)
    d = length(C)
    u = collect(range(0.31, 0.69; length=d))
    U = rand(StableRNG(seed), C, 4)
    return CopulaContractContext{typeof(u),typeof(U)}(u, U)
end

function test_distribution_contract(C, ctx, numerical_atol, margin_atol)
    d = length(C)
    @test d >= 2
    @test eltype(C) <: Real
    @test params(C) isa NamedTuple
    c = cdf(C, ctx.u)
    @test 0 <= c <= 1
    @test max(sum(ctx.u) - d + 1, 0) - 1e-8 <= c <= minimum(ctx.u) + 1e-8
    lower = 0.8 .* ctx.u
    upper = ctx.u .+ 0.2 .* (1 .- ctx.u)
    @test cdf(C, lower) <= c <= cdf(C, upper)
    @test logcdf(C, ctx.u) ≈ log(c) atol=numerical_atol
    @test cdf(C, zeros(d)) == 0
    @test cdf(C, ones(d)) == 1
    @test cdf(C, fill(-0.1, d)) == 0
    @test cdf(C, fill(1.1, d)) == 1
    for i in 1:d
        margin = ones(d)
        margin[i] = 0.37
        @test cdf(C, margin) ≈ 0.37 atol=margin_atol
        extended_margin = fill(1.1, d)
        extended_margin[i] = 0.37
        @test cdf(C, extended_margin) ≈ 0.37 atol=margin_atol
    end
    matrix_u = reshape(ctx.u, :, 1)
    @test cdf(C, matrix_u) ≈ [c] atol=numerical_atol
    @test logcdf(C, matrix_u) ≈ log.([c]) atol=1e-3
    @test Copulas.measure(C, zeros(d), ones(d)) ≈ 1 atol=1e-3
    @test Copulas.measure(C, fill(0.2, d), fill(0.6, d)) >= 0
    @test size(ctx.U) == (d, 4)
    @test eltype(ctx.U) == eltype(C)
    @test all(x -> 0 <= x <= 1, ctx.U)
    x = rand(StableRNG(41), C)
    @test length(x) == d
    @test eltype(x) == eltype(C)
    @test all(y -> 0 <= y <= 1, x)
    @test_throws ArgumentError cdf(C, zeros(d + 1))
    @test_throws ArgumentError cdf(C, zeros(d + 1, 1))
end

function test_density_contract(C, ctx, kind)
    kind === :continuous || return
    p = pdf(C, ctx.u)
    lp = logpdf(C, ctx.u)
    @test p >= 0
    @test pdf(C, fill(1e-5, length(C))) >= 0
    @test pdf(C, fill(0.5, length(C))) >= 0
    @test pdf(C, fill(1 - 1e-5, length(C))) >= 0
    @test iszero(p) ? lp == -Inf : lp ≈ log(p)
    matrix_pdf = pdf(C, reshape(ctx.u, :, 1))
    @test matrix_pdf == [p]
    @test logpdf(C, reshape(ctx.u, :, 1)) ≈ log.(matrix_pdf)
    @test all(isfinite, matrix_pdf)
    @test loglikelihood(C, ctx.U) isa Real
    @test_throws DimensionMismatch logpdf(C, zeros(length(C) + 1))
    @test_throws ArgumentError logpdf(C, zeros(length(C) + 1, 1))
end

function test_subsetting_contract(C, ctx, numerical_atol)
    d = length(C)
    dims = d == 2 ? (2, 1) : (1, d)
    S = subsetdims(C, dims)
    @test length(S) == length(dims)
    point = ctx.u[collect(dims)]
    full_point = ones(d)
    full_point[collect(dims)] = point
    @test cdf(S, point) ≈ cdf(C, full_point) atol=max(1e-5, numerical_atol)
    @test length(subsetdims(S, (1,))) == 1
    @test_throws Exception subsetdims(C, (1, 1))
    @test_throws Exception subsetdims(C, (0,))
end

function test_conditioning_contract(C, ctx, kind)
    d = length(C)
    if d == 2
        scalar = condition(C, 1, ctx.u[1])
        tupled = condition(C, (1,), (ctx.u[1],))
        @test scalar isa Distributions.UnivariateDistribution
        @test cdf(scalar, ctx.u[2]) ≈ cdf(tupled, ctx.u[2])
    end
    if d > 2
        joint = condition(C, 1, ctx.u[1])
        @test length(joint) == d - 1
        @test 0 <= cdf(joint, ctx.u[2:end]) <= 1
    end
    if d > 3
        js2 = Tuple(1:(d - 2))
        joint2 = condition(C, js2, Tuple(ctx.u[1:(d - 2)]))
        @test length(joint2) == 2
        @test 0 <= cdf(joint2, ctx.u[(d - 1):d]) <= 1
    end

    js = Tuple(1:(d - 1))
    values = Tuple(ctx.u[1:(d - 1)])
    D = condition(C, js, values)
    @test D isa Distributions.UnivariateDistribution
    @test minimum(D) == 0
    @test maximum(D) == 1
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    @test issorted(vals)
    @test logcdf(D, 0.5) ≈ log(cdf(D, 0.5))
    if kind === :continuous
        densities = pdf.(Ref(D), (0.25, 0.5, 0.75))
        @test all(x -> x >= 0, densities)
        density = pdf(D, 0.5)
        @test iszero(density) ? logpdf(D, 0.5) == -Inf :
              logpdf(D, 0.5) ≈ log(density)
    end
    @test all(x -> 0 <= x <= 1, rand(StableRNG(73), D, 3))
    q = quantile(D, 0.5)
    @test 0 <= q <= 1
    # Continuous conditionals invert their CDF. For mixed/singular models the
    # public quantile convention is only required to return a valid support
    # point; atom semantics are checked in `correctness/mathematical.jl`.
    kind === :continuous && @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
end

function test_rosenblatt_contract(C, ctx, invertible)
    R = rosenblatt(C, ctx.U)
    @test size(R) == size(ctx.U)
    @test all(x -> 0 <= x <= 1, R)
    invertible || return
    @test inverse_rosenblatt(C, R) ≈ ctx.U atol=2e-5 rtol=2e-5
    @test rosenblatt(C, ctx.u) ≈ vec(rosenblatt(C, reshape(ctx.u, :, 1)))
    @test inverse_rosenblatt(C, rosenblatt(C, ctx.u)) ≈ ctx.u atol=2e-5 rtol=2e-5
end

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

_dependence_is_defined(::typeof(Copulas.ι), kind) = kind === :continuous
_dependence_is_defined(::typeof(Copulas.corentropy), kind) = kind === :continuous
_dependence_is_defined(::Any, ::Any) = true
_dependence_dispatch_key(measure, C) =
    (which(measure, Tuple{typeof(C)}), length(C) == 2 ? :bivariate : :multivariate)

function test_dependence_contract(C, kind)
    # Distribution, density, sampling and subsetting primitives are exercised
    # above for every family.  The expensive generic measures only compose
    # those primitives, so the per-family API contract needs to guarantee that
    # dispatch exists; each distinct implementation is executed once below.
    for measure in SCALAR_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, kind) || continue
        @test applicable(measure, C)
    end
    for (measure, _) in PAIRWISE_DEPENDENCE_MEASURES
        _dependence_is_defined(measure, kind) || continue
        @test applicable(measure, C)
    end
end

function test_scalar_dependence_result(measure, C)
    value = measure(C)
    @test value isa Real
    @test !isnan(value)
    if measure !== Copulas.ι
        @test -1 <= value <= 1
    end
end

function test_pairwise_dependence_result(measure, diagonal, C)
    d = length(C)
    matrix = measure(C)
    @test size(matrix) == (d, d)
    @test matrix ≈ transpose(matrix)
    @test diag(matrix) == fill(diagonal, d)
    @test all(x -> x isa Real && !isnan(x), matrix)
end

function test_copula_contract(case, seed)
    @testset "$(case.name)" begin
        @info "Testing public copula contract" copula=case.name
        C = case.build()
        ctx = copula_contract_context(C, seed)
        @info "Testing copula operation group" copula=case.name group=:distribution
        test_distribution_contract(C, ctx, case.numerical_atol, case.margin_atol)
        @info "Testing copula operation group" copula=case.name group=:density
        test_density_contract(C, ctx, case.kind)
        @info "Testing copula operation group" copula=case.name group=:subsetting
        test_subsetting_contract(C, ctx, case.numerical_atol)
        @info "Testing copula operation group" copula=case.name group=:conditioning
        test_conditioning_contract(C, ctx, case.kind)
        @info "Testing copula operation group" copula=case.name group=:rosenblatt
        test_rosenblatt_contract(C, ctx, case.rosenblatt)
        @info "Testing copula operation group" copula=case.name group=:dependence
        test_dependence_contract(C, case.kind)
        @info "Completed public copula contract" copula=case.name
    end
end

@testset "public copula registry is exhaustive" begin
    public_families = Set(getfield(Copulas, symbol) for symbol in PUBLIC_SYMBOLS
        if getfield(Copulas, symbol) isa Type &&
           symbol !== :Copula &&
           getfield(Copulas, symbol) <: Copulas.Copula)
    represented = Set(typeof(case.build()) for case in COPULA_CASES)
    @test all(F -> any(T -> T <: F, represented), public_families)
    @test all(T -> any(F -> T <: F, public_families), represented)
end

@testset "public copula contract" begin
    for (i, case) in pairs(COPULA_CASES)
        test_copula_contract(case, 10_000 + i)
    end
end

@testset "one execution per dependence-measure dispatch" begin
    models = Tuple((case=case, copula=case.build()) for case in COPULA_CASES)

    for measure in SCALAR_DEPENDENCE_MEASURES
        seen = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, case.kind) || continue
            method, dimension_path = _dependence_dispatch_key(measure, copula)
            (method, dimension_path) in seen && continue
            push!(seen, (method, dimension_path))
            @info "Testing scalar dependence dispatch" measure=nameof(measure) copula=case.name method
            test_scalar_dependence_result(measure, copula)
        end
    end

    for (measure, diagonal) in PAIRWISE_DEPENDENCE_MEASURES
        seen = Set{Any}()
        for (; case, copula) in models
            _dependence_is_defined(measure, case.kind) || continue
            method, dimension_path = _dependence_dispatch_key(measure, copula)
            (method, dimension_path) in seen && continue
            push!(seen, (method, dimension_path))
            @info "Testing pairwise dependence dispatch" measure=nameof(measure) copula=case.name method
            test_pairwise_dependence_result(measure, diagonal, copula)
        end
    end
end
