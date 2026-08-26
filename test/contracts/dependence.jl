function test_dependence_contract(C)
    length(C) == 2 || return
    for f in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ,
              Copulas.λₗ, Copulas.λᵤ)
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
end
