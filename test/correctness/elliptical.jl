# Correctness obligation: Gaussian and Student representation, type promotion,
# and an external Sklar reference value.
@testset "TCopula degrees of freedom are data, not a type value" begin
    Σ = [1.0 0.25; 0.25 1.0]
    C2 = TCopula{2}(2, copy(Σ))
    C20 = TCopula{2}(20, copy(Σ))

    @test typeof(C2) === typeof(C20)
    @test params(C2).ν == 2
    @test params(C20).ν == 20
    @test Copulas.U(C2) == TDist(2)
    @test Copulas.U(C20) == TDist(20)
end

@testset "Bivariate Student rank dependence" begin
    C = TCopula{2}(4.0, [1.0 0.5; 0.5 1.0])
    @test Copulas.τ(C) ≈ 1 / 3 atol=2e-15
    # Heinen and Valdesogo (2020), evaluated from their Theorem 2.
    @test Copulas.ρ(C) ≈ 0.4690201700259207 atol=2e-12
    @test Copulas.ρ(TCopula{2}(4.0, [1.0 0.0; 0.0 1.0])) == 0
    @test Copulas.ρ(TCopula{2}(Inf, [1.0 0.5; 0.5 1.0])) ≈ 6asin(0.25) / π
end

@testset "Multivariate Student CDF" begin
    C2 = TCopula{2}(4.0, [1.0 0.5; 0.5 1.0])
    @test cdf(C2, [0.5, 0.5]) ≈ 1 / 4 + asin(0.5) / (2π) atol=5e-4

    C3 = TCopula{3}(5.0, Matrix{Float64}(I, 3, 3))
    @test cdf(C3, fill(0.5, 3)) ≈ 1 / 8 atol=5e-4
    @test cdf(C3, [0.0, 0.5, 0.5]) == 0
    @test cdf(C3, ones(3)) == 1

    # The internal randomized normal integration is deliberately reproducible.
    p = cdf(C3, [0.1, 0.7, 0.95])
    @test cdf(C3, [0.1, 0.7, 0.95]) == p
    @test 0 < p < 0.1

    for r in (-0.999, 0.999)
        Cnear = TCopula{2}(4.0, [1.0 r; r 1.0])
        @test cdf(Cnear, [0.5, 0.5]) ≈ 1 / 4 + asin(r) / (2π) atol=5e-4
    end

    C32 = TCopula{2}(4.0f0, Float32[1 0.5; 0.5 1])
    @test cdf(C32, Float32[0.5, 0.5]) isa Float32
    @test cdf(C32, Float32[0.5, 0.5]) ≈ Float32(1 / 3) atol=5f-4
end

@testset "Fix value Gaussian Copula & SklarDist" begin
    # source: https://discourse.julialang.org/t/cdf-of-a-copula-from-copulas-jl/85786/20
    Random.seed!(123)
    C1 = GaussianCopula{2}([1 0.5; 0.5 1])
    D1 = SklarDist(C1, (Normal(0,1),Normal(0,2)))
    @test cdf(D1, [-0.1, 0.1]) ≈ 0.3219002977336174 rtol=1e-3
end

@testset "Elliptical logpdf promotes input and parameter types" begin
    C32 = GaussianCopula{2}(Float32[1 0.25; 0.25 1])
    C64 = GaussianCopula{2}([1.0 0.25; 0.25 1.0])

    sample32 = rand(rng, C32, 2)
    @test eltype(sample32) === Float32
    @test all(0f0 .<= sample32 .<= 1f0)

    @test logpdf(C32, [0.4, 0.6]) isa Float64
    @test logpdf(C64, Float32[0.4, 0.6]) isa Float64
    @test logpdf(C64, Float32[0.4, 0.6]) ≈ logpdf(C64, [0.4, 0.6]) rtol=1e-6
end

@testset "TCopula Gaussian limit at ν = Inf" begin
    Σ = [1.0  0.6  0.3; 0.6  1.0  0.5; 0.3  0.5  1.0]
    T∞ = TCopula(Inf, copy(Σ))
    G = GaussianCopula(copy(Σ))
    u = [0.2, 0.4, 0.7]
    @test pdf(T∞, u) ≈ pdf(G, u) rtol=1e-13
    # Both CDF evaluations are numerical.
    @test cdf(T∞, u) ≈ cdf(G, u) atol=1e-12
    U = rand(StableRNG(479), G, 20)
    RT = rosenblatt(T∞, U)
    RG = rosenblatt(G, U)
    @test RT ≈ RG atol=1e-13
    @test inverse_rosenblatt(T∞, RT) ≈ U atol=1e-12
    @test loglikelihood(T∞, U) ≈ loglikelihood(G, U) atol=1e-12
end
