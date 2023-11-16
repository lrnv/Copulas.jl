
@testitem "williamson test" begin
    using Distributions, Random
    using StableRNGs
    rng = StableRNG(123)
    taus = [0.0, 0.1, 0.5, 0.9, 1.0]

    ϕ_clayton(x, θ) = max((1 + θ * x),zero(x))^(-1/θ)

    Cops = (
        WilliamsonCopula(Dirac(1),10),
        WilliamsonCopula(x -> exp(-x),10),
        WilliamsonCopula(x -> ϕ_clayton(x,2),2),
        WilliamsonCopula(x -> ϕ_clayton(x,-0.3),2)
    )
    for C in Cops
        x = rand(rng,C,1000)
    end
end
