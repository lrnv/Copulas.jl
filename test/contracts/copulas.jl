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
    @test logcdf(C, ctx.u) ≈ log(c)
    @test cdf(C, zeros(d)) == 0
    @test cdf(C, ones(d)) == 1
    for i in 1:d
        margin = ones(d)
        margin[i] = 0.37
        @test cdf(C, margin) ≈ 0.37 atol=1e-6
    end
    @test cdf(C, reshape(ctx.u, :, 1)) == [c]
    @test Copulas.measure(C, zeros(d), ones(d)) ≈ 1
    @test Copulas.measure(C, fill(0.2, d), fill(0.6, d)) >= 0
    @test size(ctx.U) == (d, 4)
    @test all(x -> 0 <= x <= 1, ctx.U)
    x = rand(StableRNG(41), C)
    @test length(x) == d
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
    @test all(isfinite, matrix_pdf)
    @test loglikelihood(C, ctx.U) isa Real
end

function test_subsetting_contract(C, ctx)
    d = length(C)
    dims = d == 2 ? (2, 1) : (1, d)
    S = subsetdims(C, dims)
    @test length(S) == length(dims)
    point = ctx.u[collect(dims)]
    full_point = ones(d)
    full_point[collect(dims)] = point
    @test cdf(S, point) ≈ cdf(C, full_point)
    @test length(subsetdims(S, (1,))) == 1
    @test_throws Exception subsetdims(C, (1, 1))
    @test_throws Exception subsetdims(C, (0,))
end

function test_conditioning_contract(C, ctx)
    d = length(C)
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
    @test minimum(D) == 0
    @test maximum(D) == 1
    vals = cdf.(Ref(D), (0.25, 0.5, 0.75))
    densities = pdf.(Ref(D), (0.25, 0.5, 0.75))
    @test issorted(vals)
    @test all(x -> x >= 0, densities)
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
    length(C) == 2 || return
    scalar_measures = kind === :continuous ?
        (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ, Copulas.ι,
         Copulas.λₗ, Copulas.λᵤ) :
        (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ,
         Copulas.λₗ, Copulas.λᵤ)
    for f in scalar_measures
        value = f(C)
        @test value isa Real
        @test !isnan(value)
    end
    K = StatsBase.corkendall(C)
    S = StatsBase.corspearman(C)
    @test size(K) == size(S) == (2, 2)
    @test K ≈ transpose(K)
    @test S ≈ transpose(S)
    @test diag(K) == diag(S) == ones(2)
    @test K[1, 2] ≈ Copulas.τ(C)
    @test S[1, 2] ≈ Copulas.ρ(C)

    pairwise_measures = (
        (Copulas.corblomqvist, Copulas.β),
        (Copulas.corgini, Copulas.γ),
        (Copulas.corlowertail, Copulas.λₗ),
        (Copulas.coruppertail, Copulas.λᵤ),
    )
    if kind === :continuous
        pairwise_measures = (pairwise_measures..., (Copulas.corentropy, Copulas.ι))
    end
    for (pairwise, scalar) in pairwise_measures
        M = pairwise(C)
        @test size(M) == (2, 2)
        @test M ≈ transpose(M)
        @test M[1, 2] ≈ scalar(C)
    end
end

function test_copula_contract(case, seed)
    @testset "$(case.name)" begin
        C = case.build()
        ctx = CopulaContractContext(C, seed)
        test_distribution_contract(C, ctx)
        test_density_contract(C, ctx, case.kind)
        test_subsetting_contract(C, ctx)
        test_conditioning_contract(C, ctx)
        test_rosenblatt_contract(C, ctx, case.rosenblatt)
        test_dependence_contract(C, case.kind)
    end
end

@testset "public copula contract" begin
    for (i, case) in pairs(COPULA_CASES)
        test_copula_contract(case, 10_000 + i)
    end
end
