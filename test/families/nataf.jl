# Family-regression layer: exact and numerical Nataf-correction
# identities and input-validation regressions.
@testset "Nataf correction" begin

    @testset "zero targets stay exactly zero, structure is preserved" begin
        R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3)), [1.0 0.0; 0.0 1.0])
        @test R₀ == [1.0 0.0; 0.0 1.0]
        R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3), Beta(2, 5)), [1.0 0.5 0.0; 0.5 1.0 -0.3; 0.0 -0.3 1.0])
        @test LinearAlgebra.issymmetric(R₀)
        @test all(LinearAlgebra.diag(R₀) .== 1)
        @test R₀[1, 3] == 0
    end

    @testset "closed-form attainable range" begin
        @test_throws ArgumentError Nataf((Uniform(), Normal()), 0.99)
    end

    @testset "end-to-end: sampled Pearson correlation matches the target" begin
        m  = (LogNormal(0, 0.8), Gamma(1, 2), Beta(1, 2))
        R  = [1.0 0.7 0.3; 0.7 1.0 0.5; 0.3 0.5 1.0]
        D  = SklarDist(GaussianCopula{length(m)}(Nataf(m, R)), m)
        R̂  = Statistics.cor(rand(rng, D, 10^5)')
        @test R̂ ≈ R atol = 0.02
        # while the uncorrected copula misses the lognormal pair by far more:
        R̃ = Statistics.cor(rand(rng, SklarDist(GaussianCopula{length(m)}(R), m), 10^5)')
        @test abs(R̃[1, 2] - R[1, 2]) > 0.03
    end

    @testset "invalid inputs throw" begin
        m = (LogNormal(0, 0.8), Gamma(2, 3))
        @test_throws ArgumentError Nataf((Pareto(1.0), Normal()), 0.5)          # infinite mean
        @test_throws ArgumentError Nataf((LogNormal(0, 2), LogNormal(0, 2)), -0.5) # unattainable target
        @test_throws ArgumentError Nataf(m, [1.0 0.5; 0.4 1.0])                 # not symmetric
        @test_throws ArgumentError Nataf(m, [0.9 0.5; 0.5 1.0])                 # bad diagonal
        @test_throws ArgumentError Nataf(m, [1.0 0.5 0.1; 0.5 1.0 0.1; 0.1 0.1 1.0]) # size mismatch
        # margins are validated even when their targets are all zero:
        @test_throws ArgumentError Nataf((Pareto(1.0), Normal()), [1.0 0.0; 0.0 1.0])
        # degenerate (Dirac) margins are rejected on the closed-form paths too:
        @test_throws ArgumentError Nataf((Normal(0, 0), Normal()), 0.5)
        @test_throws ArgumentError Nataf((LogNormal(0, 0), LogNormal(0, 1)), 0.3)
        # a bare distribution instead of a collection gets a clean error:
        @test_throws ArgumentError Nataf(Normal(), [1.0 0.5; 0.5 1.0])
    end

    @testset "type-generic: BigFloat inputs give BigFloat results" begin
        # closed-form path, at full precision:
        s = big"0.8"
        ρ₀ = Nataf((LogNormal(big"0.0", s), LogNormal(big"0.0", s)), big"0.7")
        @test ρ₀ isa BigFloat
        @test ρ₀ ≈ log1p(big"0.7" * expm1(s^2)) / s^2 atol = big"1e-60"
        @test Nataf((Normal(big"0.0", big"1.0"), Normal(big"2.0", big"3.0")), big"0.6") == big"0.6"
        # generic quadrature path (few nodes to keep the BigFloat root search cheap):
        m = (LogNormal(big"0.0", big"0.8"), Exponential(big"1.0"))
        ρ₀ = Nataf(m, big"0.5"; nodes = 8)
        @test ρ₀ isa BigFloat
        @test Float64(ρ₀) ≈ Nataf((LogNormal(0.0, 0.8), Exponential(1.0)), 0.5; nodes = 8) atol = 1e-12

        # The quadrature nodes originate in Float64, so their attainable-bound
        # tolerance must not incorrectly shrink to eps(BigFloat).
        hi = Copulas._nataf_problem(m..., big"0.5", 8).hi
        @test Nataf(m, hi + big"1e-9"; nodes = 8) == prevfloat(one(BigFloat))
    end

    @testset "attainable extremes snap just inside ±1" begin
        # A target exactly on the attainable boundary (comonotone Gaussian margins)
        # maps just inside ±1, so the GaussianCopula pipeline stays usable.
        ρ₊ = Nataf((Normal(), Normal()), 1.0)
        ρ₋ = Nataf((Normal(), Normal()), -1.0)
        @test ρ₊ == prevfloat(1.0)
        @test ρ₋ == nextfloat(-1.0)
        @test GaussianCopula{2}(ρ₊) isa GaussianCopula
        @test GaussianCopula{2}([1.0 ρ₋; ρ₋ 1.0]) isa GaussianCopula
    end

    @testset "generic-path attainability tolerance matches quadrature accuracy" begin
        # The quadrature bounds are only ~1e-8 accurate, so a target within
        # quadrature noise of the computed bound is attainable and snaps to the
        # boundary instead of being rejected; clearly outside still throws.
        m = (Gamma(2.0, 3.0), Beta(2.0, 5.0))
        hi = Copulas._nataf_problem(m..., 0.5, 32).hi
        @test Nataf(m, hi + 1e-9) == prevfloat(1.0)
        @test Nataf(m, hi - 1e-9) == prevfloat(1.0)
        @test_throws ArgumentError Nataf(m, hi + 1e-6)
    end
end
