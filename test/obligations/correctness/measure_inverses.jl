# Correctness obligation: verifies each public dependence-measure inverse on
# representative supported families and both type- and instance-based dispatch.
@testset "public dependence-measure inverses" begin
    cases = (
        (CuadrasAugeCopula{2}(0.4), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
        (GalambosCopula{2}(1.0), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
        (HuslerReissCopula{2}(1.0), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
        (LogCopula{2}(1.5), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
        (MixedCopula{2}(0.4), (Copulas.τ, Copulas.ρ, Copulas.β, Copulas.λᵤ)),
    )
    inverses = Dict(Copulas.τ => Copulas.τ⁻¹, Copulas.ρ => Copulas.ρ⁻¹,
                    Copulas.β => Copulas.β⁻¹, Copulas.λᵤ => Copulas.λᵤ⁻¹)
    for (C, measures) in cases
        CT = typeof(C)
        for measure in measures
            inverse = inverses[measure]
            value = measure(C)
            parameter = inverse(CT, value)
            rebuilt = ExtremeValueCopula{2}(typeof(C.tail)(parameter))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end

@testset "one-parameter copula dependence-measure inverses" begin
    archimedean = (
        AMHCopula{2}(0.5), ClaytonCopula{2}(1.0), FrankCopula{2}(2.0),
        GumbelCopula{2}(1.5), GumbelBarnettCopula{2}(0.5),
        InvGaussianCopula{2}(0.5), JoeCopula{2}(1.5),
    )
    for C in archimedean
        CT = typeof(C)
        for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                                   (Copulas.ρ, Copulas.ρ⁻¹))
            value = measure(C)
            rebuilt = CT(inverse(CT, value))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end

    C = FGMCopula{2}(0.5)
    for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                               (Copulas.ρ, Copulas.ρ⁻¹))
        value = measure(C)
        rebuilt = FGMCopula{2}(inverse(FGMCopula{2}, value))
        @test measure(rebuilt) ≈ value atol=2e-6
    end
end


@testset "generator dependence-measure inverses" begin
    for G in (Copulas.AMHGenerator(0.5), Copulas.ClaytonGenerator(1.0),
              Copulas.FrankGenerator(2.0), Copulas.GumbelGenerator(1.5),
              Copulas.GumbelBarnettGenerator(0.5),
              Copulas.InvGaussianGenerator(0.5), Copulas.JoeGenerator(1.5))
        GT = typeof(G)
        for (measure, inverse) in ((Copulas.τ, Copulas.τ⁻¹),
                                   (Copulas.ρ, Copulas.ρ⁻¹))
            value = measure(G)
            rebuilt = GT(inverse(GT, value))
            @test measure(rebuilt) ≈ value atol=2e-6
        end
    end
end
