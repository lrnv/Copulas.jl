
@testitem "Test of Trivariate Liouville Copulas" begin
    using Random, Distributions
    using StableRNGs
    using StatsBase
    rng = StableRNG(123)

    for G in (
        Copulas.AMHGenerator(0.6),
        Copulas.AMHGenerator(-0.3),
        Copulas.ClaytonGenerator(-0.05),
        Copulas.IndependentGenerator(),
        Copulas.GumbelBarnettGenerator(0.7),
        Copulas.InvGaussianGenerator(0.05),
        Copulas.InvGaussianGenerator(8),
        Copulas.WilliamsonGenerator(LogNormal(),20),
    )
        C = LiouvilleCopula(rand(1:6,3),G)
        u = rand(C,10)
        cdf(C,u)
        # check margins uniformity ? 
        @test true
    end
end