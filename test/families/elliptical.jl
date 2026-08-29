# Family-regression layer: Gaussian and Student copula reference,
# fitting, marginal, and numerical regressions.
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

@testset "Fix value Gaussian Copula & SklarDist" begin
    # source: https://discourse.julialang.org/t/cdf-of-a-copula-from-copulas-jl/85786/20
    Random.seed!(123)
    C1 = GaussianCopula{2}([1 0.5; 0.5 1])
    D1 = SklarDist(C1, (Normal(0,1),Normal(0,2)))
    @test cdf(D1, [-0.1, 0.1]) ≈ 0.3219002977336174 rtol=1e-3
end

@testset "GaussianCopula equicorrelation constructor" begin
    Cρ = GaussianCopula{2}(0.5)
    @test Cρ isa GaussianCopula{2}
    # PD lower bound check (just above boundary for d=3: lower = -0.5)
    Cneg = GaussianCopula{3}(-0.49)
    @test Cneg isa GaussianCopula{3}
    # Boundary should throw
    @test_throws ArgumentError GaussianCopula{3}(-0.5)
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
