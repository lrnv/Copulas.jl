@testset "public dependence-measure inverses" begin
    for C in (CuadrasAugeCopula{2}(0.4), GalambosCopula{2}(1.0),
              LogCopula{2}(1.5), MixedCopula{2}(0.4))
        CT = typeof(C)
        for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                                   (Copulas.ρ, Copulas.ρ⁻¹),
                                   (Copulas.β, Copulas.β⁻¹),
                                   (Copulas.λᵤ, Copulas.λᵤ⁻¹))
            value = measure(C)
            rebuilt = CT(inverse(CT, value))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end


@testset "generator dependence-measure inverses" begin
    for G in (Copulas.ClaytonGenerator(1.0), Copulas.GumbelGenerator(1.5),
              Copulas.FrankGenerator(2.0), Copulas.JoeGenerator(1.5))
        GT = typeof(G)
        for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                                   (Copulas.ρ, Copulas.ρ⁻¹))
            value = measure(G)
            rebuilt = GT(inverse(GT, value))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end
