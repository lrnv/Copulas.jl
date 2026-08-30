# Equivalence obligation for alternative public parameterizations of the same
# extreme-value model. Full multivariate numerical oracles live in correctness.
function test_ev_equivalence(left, right, point; atol, rtol)
    @test cdf(left, point) ≈ cdf(right, point) atol=atol rtol=rtol
    @test logpdf(left, point) ≈ logpdf(right, point) atol=10atol rtol=10rtol
end

@testset "historical discrete-spectral parameterizations" begin
    cases = (
        (Copulas.MOTail(0.30, 0.50, 0.70),
         Copulas.MOTail(2, [0.50, 0.30, 0.70]), 4e-14),
        (Copulas.BC2Tail(0.30, 0.70), Copulas.BC2Tail([0.30, 0.70]), 3e-14),
    )
    for (historical, structured, tol) in cases
        @test typeof(historical) == typeof(structured)
        @test historical isa Copulas.DiscreteSpectralPickandsTail
        x = [0.37, 1.29]
        @test Copulas.ℓ(structured, x) ≈
              sum(x) * Copulas.A(historical, x[1] / sum(x)) atol=tol rtol=tol
        @test cdf(ExtremeValueCopula{2}(historical), [0.34, 0.76]) ≈
              cdf(ExtremeValueCopula{2}(structured), [0.34, 0.76]) atol=tol rtol=tol
    end
    @test params(cases[1][1]) == (λ₁=0.30, λ₂=0.50, λ₃=0.70)
    @test params(cases[2][1]) == (a=0.30, b=0.70)
end

@testset "equivalent extremal-t parameterizations" begin
    for (d, ν, ρ) in ((3, 1.3, 0.25), (4, 2.2, 0.4))
        R = fill(ρ, d, d)
        R[diagind(R)] .= 1
        scalar = ExtremeValueCopula{d}(Copulas.tEVTail(ν, ρ))
        matrix = ExtremeValueCopula{d}(Copulas.tEVTail(ν, R))
        test_ev_equivalence(scalar, matrix,
            collect(range(0.29, 0.83; length=d)); atol=3e-7, rtol=3e-7)
    end
end

@testset "Tawn reductions" begin
    α, θ1, θ2 = 2.1, 0.67, 0.38
    historical = ExtremeValueCopula{2}(Copulas.AsymLogTail(α, θ1, θ2))
    tawn = ExtremeValueCopula{2}(Copulas.TawnTail(α, [θ2, θ1]))
    test_ev_equivalence(tawn, historical, [0.34, 0.76];
                        atol=3e-12, rtol=3e-12)

    for d in (3, 4)
        symmetric = ExtremeValueCopula{d}(Copulas.TawnTail(1.7, ones(d)))
        logistic = ExtremeValueCopula{d}(Copulas.LogTail(1.7))
        test_ev_equivalence(symmetric, logistic,
            collect(range(0.29, 0.82; length=d)); atol=5e-12, rtol=5e-12)
    end
end

@testset "asymmetric Galambos reductions" begin
    α, θ1, θ2 = 1.4, 0.67, 0.38
    historical = ExtremeValueCopula{2}(Copulas.AsymGalambosTail(α, θ1, θ2))
    structured = ExtremeValueCopula{2}(Copulas.AsymGalambosTail(
        2, [α], [[1 - θ1], [1 - θ2], [θ1, θ2]]))
    test_ev_equivalence(structured, historical, [0.34, 0.76];
                        atol=3e-10, rtol=3e-10)

    for d in (3, 4)
        asymmetric = ExtremeValueCopula{d}(Copulas.AsymGalambosTail(1.1, ones(d)))
        symmetric = ExtremeValueCopula{d}(Copulas.GalambosTail(1.1))
        test_ev_equivalence(asymmetric, symmetric,
            collect(range(0.29, 0.82; length=d)); atol=3e-9, rtol=3e-9)
    end
end
