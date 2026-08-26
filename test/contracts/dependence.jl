function test_dependence_contract(C)
    length(C) == 2 || return
    for f in (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.γ,
              Copulas.λₗ, Copulas.λᵤ)
        value = f(C)
        @test value isa Real
        @test !isnan(value)
    end
    @test StatsBase.corkendall(C) == Copulas.τ(C)
    @test StatsBase.corspearman(C) == Copulas.ρ(C)
end
