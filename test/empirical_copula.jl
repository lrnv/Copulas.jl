@testitem "Check issue #105 (empirical copula pdf & cdf)" begin
    using Random, Distributions
    using InteractiveUtils
    using StableRNGs
    rng = StableRNG(123)

    d = 3
    u = rand(d,1000)
    C₁ = EmpiricalCopula(u)
    
    x = randn(d,1000)
    C₂ = EmpiricalCopula(x, pseudo_values=false)

    pdf(C₁, ones(d)/2)
    pdf(C₂, ones(d)/2)
    cdf(C₁, ones(d)/2)
    cdf(C₂, ones(d)/2)

    @test true
end