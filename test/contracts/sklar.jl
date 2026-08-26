@testset "SklarDist public contract" begin
    C = GaussianCopula{2}(0.3)
    D = SklarDist(C, (Normal(), Exponential()))
    x = [0.1, 1.2]
    @test length(D) == 2
    @test params(D) isa NamedTuple
    @test StatsBase.dof(D) == StatsBase.dof(C) + sum(StatsBase.dof, D.m)
    @test 0 <= cdf(D, x) <= 1
    @test logcdf(D, x) ≈ log(cdf(D, x))
    @test pdf(D, x) >= 0
    @test logpdf(D, x) ≈ log(pdf(D, x))
    X = rand(StableRNG(31), D, 4)
    @test size(X) == (2, 4)
    @test loglikelihood(D, X) isa Real

    S = subsetdims(D, (2, 1))
    @test length(S) == 2
    @test S.C == subsetdims(C, (2, 1))
    conditional = condition(D, 1, x[1])
    @test minimum(conditional) == 0
    @test cdf(conditional, quantile(conditional, 0.5)) >= 0.5 - sqrt(eps())

    R = rosenblatt(D, X)
    @test size(R) == size(X)
    @test inverse_rosenblatt(D, R) ≈ X atol=2e-5 rtol=2e-5

    D3 = SklarDist(GaussianCopula{3}(0.3), (Normal(), Exponential(), Gamma(2, 1)))
    x3 = [0.1, 1.2, 0.8]
    joint = condition(D3, 1, x3[1])
    @test length(joint) == 2
    @test 0 <= cdf(joint, x3[2:3]) <= 1
    @test length(subsetdims(D3, (3, 1))) == 2
end
