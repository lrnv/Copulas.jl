@testset "Nataf correction" begin

    @testset "Gaussian margins reproduce the target exactly" begin
        R = [1.0 0.6 -0.2; 0.6 1.0 0.3; -0.2 0.3 1.0]
        R₀ = Nataf((Normal(), Normal(2, 3), Normal(-1, 0.5)), R)
        @test R₀ == R
    end

    @testset "zero targets stay exactly zero, structure is preserved" begin
        R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3)), [1.0 0.0; 0.0 1.0])
        @test R₀ == [1.0 0.0; 0.0 1.0]
        R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3), Beta(2, 5)), [1.0 0.5 0.0; 0.5 1.0 -0.3; 0.0 -0.3 1.0])
        @test LinearAlgebra.issymmetric(R₀)
        @test all(LinearAlgebra.diag(R₀) .== 1)
        @test R₀[1, 3] == 0
    end

    @testset "bivariate lognormal against the closed form" begin
        # For LogNormal margins, the Pearson correlation induced by a Gaussian
        # copula with parameter ρ₀ is known in closed form:
        #   r(ρ₀) = (exp(ρ₀s₁s₂) - 1) / √((exp(s₁²) - 1)(exp(s₂²) - 1)),
        # so the exact correction is ρ₀ = log(1 + r√(⋯)) / (s₁s₂).
        for (s₁, s₂, r) in ((0.8, 0.8, 0.7), (0.5, 1.2, 0.4), (1.0, 1.0, -0.2))
            ρ₀_exact = log(1 + r * sqrt(expm1(s₁^2) * expm1(s₂^2))) / (s₁ * s₂)
            ρ₀ = Nataf((LogNormal(0, s₁), LogNormal(0, s₂)), r)
            @test ρ₀ ≈ ρ₀_exact atol = 1e-6
        end
        s, r = 0.8, 0.6
        expected = r * sqrt(expm1(s^2)) / s
        @test Nataf((Normal(1, 2), LogNormal(0, s)), r) ≈ expected
        @test Nataf((LogNormal(0, s), Normal(1, 2)), r) ≈ expected
    end

    @testset "scalar and matrix methods agree" begin
        m = (LogNormal(0, 0.8), Gamma(2, 3))
        @test Nataf(m, 0.6) == Nataf(m, [1.0 0.6; 0.6 1.0])[1, 2]
    end

    @testset "end-to-end: sampled Pearson correlation matches the target" begin
        m  = (LogNormal(0, 0.8), Gamma(1, 2), Beta(1, 2))
        R  = [1.0 0.7 0.3; 0.7 1.0 0.5; 0.3 0.5 1.0]
        D  = SklarDist(GaussianCopula(Nataf(m, R)), m)
        R̂  = Statistics.cor(rand(rng, D, 10^5)')
        @test R̂ ≈ R atol = 0.02
        # while the uncorrected copula misses the lognormal pair by far more:
        R̃ = Statistics.cor(rand(rng, SklarDist(GaussianCopula(R), m), 10^5)')
        @test abs(R̃[1, 2] - R[1, 2]) > 0.03
    end

    @testset "invalid inputs throw" begin
        m = (LogNormal(0, 0.8), Gamma(2, 3))
        @test_throws ArgumentError Nataf((Pareto(1.0), Normal()), 0.5)          # infinite mean
        @test_throws ArgumentError Nataf((LogNormal(0, 2), LogNormal(0, 2)), -0.5) # unattainable target
        @test_throws ArgumentError Nataf(m, [1.0 0.5; 0.4 1.0])                 # not symmetric
        @test_throws ArgumentError Nataf(m, [0.9 0.5; 0.5 1.0])                 # bad diagonal
        @test_throws ArgumentError Nataf(m, [1.0 0.5 0.1; 0.5 1.0 0.1; 0.1 0.1 1.0]) # size mismatch
        @test_throws ArgumentError Nataf((LogNormal(),), 0.5)                   # scalar target needs 2 margins
        @test_throws ArgumentError Nataf(m, 1.5)                                # target outside [-1, 1]
        @test_throws ArgumentError Nataf(m, 0.5; nodes=1)                       # not enough nodes
    end

    @testset "type-generic: BigFloat inputs give BigFloat results" begin
        # closed-form path, at full precision:
        s = big"0.8"
        ρ₀ = Nataf((LogNormal(big"0.0", s), LogNormal(big"0.0", s)), big"0.7")
        @test ρ₀ isa BigFloat
        @test ρ₀ ≈ log1p(big"0.7" * expm1(s^2)) / s^2 atol = big"1e-60"
        @test Nataf((Normal(big"0.0", big"1.0"), Normal(big"2.0", big"3.0")), big"0.6") == big"0.6"
        # generic quadrature path (few nodes to keep the BigFloat bisection cheap):
        m = (LogNormal(big"0.0", big"0.8"), Exponential(big"1.0"))
        ρ₀ = Nataf(m, big"0.5"; nodes = 8)
        @test ρ₀ isa BigFloat
        @test Float64(ρ₀) ≈ Nataf((LogNormal(0.0, 0.8), Exponential(1.0)), 0.5; nodes = 8) atol = 1e-12
    end

    @testset "attainable extremes map to ±1" begin
        # A target exactly on the attainable boundary (comonotone Gaussian margins) maps to ρ₀ = ±1.
        @test Nataf((Normal(), Normal()), 1.0) == 1.0
        @test Nataf((Normal(), Normal()), -1.0) == -1.0
    end
end
