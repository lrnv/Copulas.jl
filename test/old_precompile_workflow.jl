
@testitem "standard functionality test" begin
    using Random, Distributions
    using StableRNGs
    rng = StableRNG(123)
    biv_cops = [
        GaussianCopula([1 0.7; 0.7 1]),
        TCopula(2,[1 0.7; 0.7 1]),
        ClaytonCopula(2,7),
        JoeCopula(2,3),
        GumbelCopula(2,8),
        GumbelCopula(2,1), # should be equivalent to an independent copula. 
        FrankCopula(2,0.5),
        AMHCopula(2,0.7)
    ]
    for C in biv_cops
        u = Random.rand(rng,C,10)
        pdf(C,[0.5,0.5])
        cdf(C,[0.5,0.5])
        D = SklarDist(C,[Gamma(1,1),Normal(1,1)])
        u = Random.rand(rng,D,10)
        pdf(D,[0.5,0])
        cdf(D,[0.5,0])
    end
    @test true
end
