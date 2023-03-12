
@testitem "test bijection" begin
    using Random 
    Random.seed!(123)
    taus = [0.0, 0.1, 0.5, 0.9, 1.0]

    for T in (
        # AMHCopula,
        ClaytonCopula,
        # FrankCopula,
        GumbelCopula,
        # IndependentCopula,
        # JoeCopula
    )
        for τ in taus
            @test Copulas.τ(T(2,Copulas.τ⁻¹(T,τ))) ≈ τ
        end
    end
end
