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
