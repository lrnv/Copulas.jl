# Exact low-dimensional regression battery for the public discrete/mixed SklarDist API.
# These tests use closed-form probabilities whenever possible so atom semantics do not
# depend only on simulation or numerical normalization checks.

@testset "SklarDist discrete and mixed exact laws" begin
    @testset "independent Bernoulli rectangle masses" begin
        p1, p2 = 0.25, 0.60
        S = SklarDist(IndependentCopula{2}(), (Bernoulli(p1), Bernoulli(p2)))
        expected = [
            (1 - p1) * (1 - p2)  (1 - p1) * p2
            p1 * (1 - p2)        p1 * p2
        ]
        actual = [pdf(S, [i, j]) for i in 0:1, j in 0:1]
        @test actual ≈ expected atol=1e-14
        @test sum(actual) ≈ 1 atol=1e-14
        @test pdf(S, [0.5, 0]) == 0
    end

    @testset "FGM Bernoulli rectangle masses" begin
        S = SklarDist(FGMCopula{2}(1.0), (Bernoulli(0.5), Bernoulli(0.5)))
        expected = [0.3125 0.1875; 0.1875 0.3125]
        actual = [pdf(S, [i, j]) for i in 0:1, j in 0:1]
        @test actual ≈ expected atol=1e-12
        @test sum(actual) ≈ 1 atol=1e-12
    end

    @testset "mixed Normal-Bernoulli likelihood" begin
        θ = 0.8
        p = 0.35
        x = 0.2
        u = cdf(Normal(), x)
        fx = pdf(Normal(), x)
        S = SklarDist(FGMCopula{2}(θ), (Normal(), Bernoulli(p)))

        # For FGM, ∂C(u,v)/∂u = v * (1 + θ(1-v)(1-2u)).
        # Differencing this derivative across the Bernoulli latent interval gives
        # the exact mixed density/mass with respect to dx × counting measure.
        expected0 = fx * (1 - p) * (1 + θ * p * (1 - 2u))
        expected1 = fx * p * (1 - θ * (1 - p) * (1 - 2u))
        @test pdf(S, [x, 0]) ≈ expected0 atol=1e-12 rtol=1e-12
        @test pdf(S, [x, 1]) ≈ expected1 atol=1e-12 rtol=1e-12
        @test pdf(S, [x, 0]) + pdf(S, [x, 1]) ≈ fx atol=1e-12 rtol=1e-12
    end

    @testset "conditioning on an atom" begin
        S = SklarDist(FGMCopula{2}(1.0), (Bernoulli(0.5), Bernoulli(0.5)))

        # P(X₂=0 | X₁=0) = 0.3125 / 0.5 = 0.625.
        D0 = condition(S, 1, 0)
        @test pdf(D0, 0) ≈ 0.625 atol=1e-12
        @test pdf(D0, 1) ≈ 0.375 atol=1e-12
        @test cdf(D0, 0) ≈ 0.625 atol=1e-12
        @test pdf(D0, 0) + pdf(D0, 1) ≈ 1 atol=1e-12

        # The other atom reverses the conditional association.
        D1 = condition(S, 1, 1)
        @test pdf(D1, 0) ≈ 0.375 atol=1e-12
        @test pdf(D1, 1) ≈ 0.625 atol=1e-12
    end

    @testset "finite-support likelihood normalization" begin
        margins = (Bernoulli(0.2), Bernoulli(0.4), Bernoulli(0.7))
        S = SklarDist(IndependentCopula{3}(), margins)
        masses = [pdf(S, [i, j, k]) for i in 0:1, j in 0:1, k in 0:1]
        @test sum(masses) ≈ 1 atol=1e-14
        for i in 0:1, j in 0:1, k in 0:1
            @test pdf(S, [i, j, k]) ≈
                  pdf(margins[1], i) * pdf(margins[2], j) * pdf(margins[3], k) atol=1e-14
        end
    end

    @testset "atom-free route agrees with continuous density formula" begin
        C = FGMCopula{2}(0.6)
        S = SklarDist(C, (Normal(), Exponential()))
        x = [0.3, 0.8]
        u = [cdf(S.m[1], x[1]), cdf(S.m[2], x[2])]
        expected = pdf(S.m[1], x[1]) * pdf(S.m[2], x[2]) * pdf(C, u)
        @test pdf(S, x) ≈ expected atol=1e-13 rtol=1e-13
        @test exp(Copulas._sklar_logpdf_atoms(S, x)) ≈ expected atol=1e-13 rtol=1e-13
    end
end
