@testitem "GaussianCopula" begin
    using Distributions
    using Random
    using StableRNGs
    rng = StableRNG(123)
    C = GaussianCopula([1 -0.1; -0.1 1])
    M1 = Beta(2, 3)
    M2 = LogNormal(2, 3)
    D = SklarDist(C, (M1, M2))
    X = rand(rng, D, 10)
    loglikelihood(D, X)
    @test_broken fit(SklarDist{TCopula,Tuple{Beta,LogNormal}}, X) # should give a very high \nu for the student copula.
end

@testitem "Fix value Gaussian Copula & SklarDist" begin
    using Distributions
    using Random

    # source: https://discourse.julialang.org/t/cdf-of-a-copula-from-copulas-jl/85786/20
    Random.seed!(123)
    C1 = GaussianCopula([1 0.5; 0.5 1])
    D1 = SklarDist(C1, (Normal(0, 1), Normal(0, 2)))
    @test cdf(D1, [-0.1, 0.1]) ≈ 0.3219002977336174 rtol = 1e-3
end

@testitem "Rosenblatt" begin
    using StatsBase
    C = GaussianCopula([1 0.7071; 0.7071 1])

    u = rand(C, 10^6)

    U = rosenblatt(C, u)
    @test corkendall(U[1, :], U[2, :]) ≈ 0 atol = 0.01

    @test inverse_rosenblatt(C, U) ≈ u
end
