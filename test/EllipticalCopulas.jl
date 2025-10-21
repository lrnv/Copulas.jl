@testset "GaussianCopula" begin
    # [GenericTests integration]: Maybe. The broken fit on mixed marginals is out-of-scope for generic copula properties; keep here.
    Random.seed!(rng,123)
    C = GaussianCopula([1 -0.1; -0.1 1])
    M1 = Beta(2,3)
    M2 = LogNormal(2,3)
    D = SklarDist(C,(M1,M2))
    X = rand(rng,D,10)
    loglikelihood(D,X)
    @test true
end

@testset "Fix value Gaussian Copula & SklarDist" begin
    # [GenericTests integration]: Yes. This is a regression value test for cdf(SklarDist(...)); can be moved to a generic Sklar fixture tests.

    # source: https://discourse.julialang.org/t/cdf-of-a-copula-from-copulas-jl/85786/20
    Random.seed!(123)
    C1 = GaussianCopula([1 0.5; 0.5 1])
    D1 = SklarDist(C1, (Normal(0,1),Normal(0,2)))
    @test cdf(D1, [-0.1, 0.1]) ≈ 0.3219002977336174 rtol=1e-3
end

@testset "GaussianCopula equicorrelation constructor" begin
    Cρ = GaussianCopula(2, 0.5)
    @test Cρ isa GaussianCopula{2}
    # Theoretical Kendall tau for bivariate Gaussian: τ = 2/π asin(ρ)
    @test isapprox(Copulas.τ(Cρ), 2*asin(0.5)/π; rtol=1e-12)
    # Zero correlation gives independent copula
    C0 = GaussianCopula(2, 0.0)
    @test C0 == IndependentCopula(2)
    # PD lower bound check (just above boundary for d=3: lower = -0.5)
    Cneg = GaussianCopula(3, -0.49)
    @test Cneg isa GaussianCopula{3}
    # Boundary should throw
    @test_throws ArgumentError GaussianCopula(3, -0.5)
end