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
    end

    @testset "closed-form fast paths agree with the quadrature fallback" begin
        # The exact-path pairs (Normal/LogNormal) never reach the quadrature, so
        # drive the internal quadrature machinery directly for comparison.
        quad_pair(Fᵢ, Fⱼ, r; nodes = 48) = begin
            z, w = Copulas._gauss_hermite(nodes)
            gᵢ = Copulas._nataf_standardized(Fᵢ, 1, z, w)
            gⱼ = Copulas._nataf_standardized(Fⱼ, 2, z, w)
            Copulas._nataf_pair(Float64(r), gᵢ.(z), gⱼ, z, w, 1, 2)
        end
        for (Fᵢ, Fⱼ, r) in ((Normal(1, 2), LogNormal(0, 0.8), 0.6),
                            (Normal(), LogNormal(1, 0.5), -0.3),
                            (LogNormal(0, 0.8), LogNormal(1, 0.5), 0.4))
            @test Nataf((Fᵢ, Fⱼ), r) ≈ quad_pair(Fᵢ, Fⱼ, r) atol = 1e-8
        end
        # the reversed-order dispatch hits the same closed form:
        @test Nataf((LogNormal(0, 0.8), Normal(1, 2)), 0.6) == Nataf((Normal(1, 2), LogNormal(0, 0.8)), 0.6)
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

    @testset "attainable extremes map to ±1" begin
        # A target exactly on the attainable boundary (comonotone Gaussian margins) maps to ρ₀ = ±1.
        @test Nataf((Normal(), Normal()), 1.0) == 1.0
        @test Nataf((Normal(), Normal()), -1.0) == -1.0
    end
end
