# Nataf-operation contract: matrix structure, attainable ranges, and documented
# validation errors for `Nataf`.
@testset "Nataf structure and validation" begin
    R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3)), [1.0 0.0; 0.0 1.0])
    @test R₀ == [1.0 0.0; 0.0 1.0]

    R₀ = Nataf((LogNormal(0, 0.8), Gamma(2, 3), Beta(2, 5)),
        [1.0 0.5 0.0; 0.5 1.0 -0.3; 0.0 -0.3 1.0])
    @test LinearAlgebra.issymmetric(R₀)
    @test all(LinearAlgebra.diag(R₀) .== 1)
    @test R₀[1, 3] == 0

    @test_throws ArgumentError Nataf((Uniform(), Normal()), 0.99)

    margins = (LogNormal(0, 0.8), Gamma(2, 3))
    @test_throws ArgumentError Nataf((Pareto(1.0), Normal()), 0.5)
    @test_throws ArgumentError Nataf(
        (LogNormal(0, 2), LogNormal(0, 2)), -0.5)
    @test_throws ArgumentError Nataf(margins, [1.0 0.5; 0.4 1.0])
    @test_throws ArgumentError Nataf(margins, [0.9 0.5; 0.5 1.0])
    @test_throws ArgumentError Nataf(margins,
        [1.0 0.5 0.1; 0.5 1.0 0.1; 0.1 0.1 1.0])
    @test_throws ArgumentError Nataf(
        (Pareto(1.0), Normal()), [1.0 0.0; 0.0 1.0])
    @test_throws ArgumentError Nataf((Normal(0, 0), Normal()), 0.5)
    @test_throws ArgumentError Nataf(
        (LogNormal(0, 0), LogNormal(0, 1)), 0.3)
    @test_throws ArgumentError Nataf(Normal(), [1.0 0.5; 0.5 1.0])
end


# Nataf-operation correctness: exact, statistical, type-generic, and boundary
# regressions for the public Nataf correction.

@testset "all documented Nataf dispatches have an oracle" begin
    r, s = 0.2, 0.8
    lognormal_scale = sqrt(expm1(s^2))
    uniform_lognormal = sqrt(2) / s * quantile(
        Normal(), 1 / 2 + r * lognormal_scale / (2sqrt(3)))
    exact_cases = (
        (Normal(), Normal(2, 3), r),
        (LogNormal(0, s), LogNormal(1, s), log1p(r * expm1(s^2)) / s^2),
        (Normal(), LogNormal(0, s), r * lognormal_scale / s),
        (LogNormal(0, s), Normal(), r * lognormal_scale / s),
        (Uniform(), Uniform(-2, 3), 2sinpi(r / 6)),
        (Uniform(), Normal(), r * sqrt(π / 3)),
        (Normal(), Uniform(), r * sqrt(π / 3)),
        (Uniform(), LogNormal(0, s), uniform_lognormal),
        (LogNormal(0, s), Uniform(), uniform_lognormal),
    )
    checked = Set{Method}()
    for (Fᵢ, Fⱼ, expected) in exact_cases
        @test Nataf((Fᵢ, Fⱼ), r) ≈ expected
        push!(checked, which(Copulas._nataf_problem,
            Tuple{typeof(Fᵢ),typeof(Fⱼ),Float64,Int}))
    end

    Fᵢ, Fⱼ = Gamma(2.0, 1.0), Beta(2.0, 3.0)
    generic = Nataf((Fᵢ, Fⱼ), r; nodes=8)
    @test generic ≈ Nataf((Fⱼ, Fᵢ), r; nodes=8) atol=1e-7
    @test -1 < generic < 1
    push!(checked, which(Copulas._nataf_problem,
        Tuple{typeof(Fᵢ),typeof(Fⱼ),Float64,Int}))

    documented_pairs = (
        (Normal(), Normal()), (Normal(), LogNormal(0, s)),
        (Normal(), Uniform()), (LogNormal(0, s), Normal()),
        (LogNormal(0, s), LogNormal(0, s)), (LogNormal(0, s), Uniform()),
        (Uniform(), Normal()), (Uniform(), LogNormal(0, s)),
        (Uniform(), Uniform()), (Fᵢ, Fⱼ),
    )
    selected = Set(which(Copulas._nataf_problem,
        Tuple{typeof(a),typeof(b),Float64,Int}) for (a, b) in documented_pairs)
    @test selected == checked
end
@testset "Nataf correction" begin

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
