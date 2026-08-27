# Public-API contract: applies the universal distribution, sampling, subsetting,
# conditioning, Rosenblatt, and dependence-measure behavior to every family.
struct CopulaContractContext{TU,TM}
    u::TU
    U::TM
end

function CopulaContractContext(C, seed)
    d = length(C)
    u = collect(range(0.31, 0.69; length=d))
    U = rand(StableRNG(seed), C, 4)
    return CopulaContractContext{typeof(u),typeof(U)}(u, U)
end

function test_distribution_contract(C, ctx)
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
    @test logcdf(C, ctx.u) ≈ log(c)
    @test cdf(C, zeros(d)) == 0
    @test cdf(C, ones(d)) == 1
    @test cdf(C, fill(-0.1, d)) == 0
    @test cdf(C, fill(1.1, d)) == 1
    for i in 1:d
        margin = ones(d)
        margin[i] = 0.37
        @test cdf(C, margin) ≈ 0.37 atol=1e-6
        extended_margin = fill(1.1, d)
        extended_margin[i] = 0.37
        @test cdf(C, extended_margin) ≈ 0.37 atol=1e-6
    end
    matrix_u = reshape(ctx.u, :, 1)
    @test cdf(C, matrix_u) == [c]
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

function test_subsetting_contract(C, ctx)
    d = length(C)
    dims = d == 2 ? (2, 1) : (1, d)
    S = subsetdims(C, dims)
    @test length(S) == length(dims)
    point = ctx.u[collect(dims)]
    full_point = ones(d)
    full_point[collect(dims)] = point
    @test cdf(S, point) ≈ cdf(C, full_point) atol=1e-5
    @test length(subsetdims(S, (1,))) == 1
    @test_throws Exception subsetdims(C, (1, 1))
    @test_throws Exception subsetdims(C, (0,))
end

function test_conditioning_contract(C, ctx, kind)
    d = length(C)
    if d == 2
        scalar = condition(C, 1, ctx.u[1])
        tupled = condition(C, (1,), (ctx.u[1],))
        @test scalar isa Copulas.Distortion
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
    @test D isa Copulas.Distortion
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
    @test cdf(D, q) >= 0.5 - sqrt(eps(Float64))
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

function test_dependence_contract(C, kind)
    d = length(C)
    scalar_measures = kind === :continuous ?
        (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ, Copulas.ι,
         Copulas.λₗ, Copulas.λᵤ) :
        (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ,
         Copulas.λₗ, Copulas.λᵤ)
    for f in scalar_measures
        value = f(C)
        @test value isa Real
        @test !isnan(value)
        if f !== Copulas.ι
            @test -1 <= value <= 1
        end
    end
    K = StatsBase.corkendall(C)
    S = StatsBase.corspearman(C)
    @test size(K) == size(S) == (d, d)
    @test K ≈ transpose(K)
    @test S ≈ transpose(S)
    @test diag(K) == diag(S) == ones(d)
    pair = subsetdims(C, (1, 2))
    @test K[1, 2] ≈ Copulas.τ(pair)
    @test S[1, 2] ≈ Copulas.ρ(pair)

    pairwise_measures = (
        (Copulas.corblomqvist, Copulas.β, 1),
        (Copulas.corgini, Copulas.γ, 1),
        (Copulas.corlowertail, Copulas.λₗ, 1),
        (Copulas.coruppertail, Copulas.λᵤ, 1),
    )
    if kind === :continuous
        pairwise_measures = (pairwise_measures..., (Copulas.corentropy, Copulas.ι, 0))
    end
    for (pairwise, scalar, diagonal) in pairwise_measures
        M = pairwise(C)
        @test size(M) == (d, d)
        @test M ≈ transpose(M)
        @test diag(M) == fill(diagonal, d)
        @test M[1, 2] ≈ scalar(pair)
    end
end

function test_copula_contract(case, seed)
    @testset "$(case.name)" begin
        C = case.build()
        ctx = CopulaContractContext(C, seed)
        test_distribution_contract(C, ctx)
        test_density_contract(C, ctx, case.kind)
        test_subsetting_contract(C, ctx)
        test_conditioning_contract(C, ctx, case.kind)
        test_rosenblatt_contract(C, ctx, case.rosenblatt)
        test_dependence_contract(C, case.kind)
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
