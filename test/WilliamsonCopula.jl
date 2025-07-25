
@testitem "williamson test" begin
    using Distributions, Random
    using StableRNGs
    rng = StableRNG(12)
    taus = [0.0, 0.1, 0.5, 0.9, 1.0]

    # ϕ_clayton(x, θ) = max((1 + θ * x),zero(x))^(-1/θ)

    Cops = (
        ArchimedeanCopula(10,i𝒲(Dirac(1),10)),
        ArchimedeanCopula(2,i𝒲(Pareto(1),5)),
        ArchimedeanCopula(2,i𝒲(LogNormal(3),5)),
    )
    for C in Cops
        rand(rng,C,10)
    end
end

@testitem "williamson test" begin
    using Distributions, Random
    using StableRNGs
    rng = StableRNG(12)
    taus = [0.0, 0.1, 0.5, 0.9, 1.0]

    # ϕ_clayton(x, θ) = max((1 + θ * x),zero(x))^(-1/θ)

    Cops = (
        ArchimedeanCopula(10,i𝒲(MixtureModel([Dirac(1), Dirac(2)]),11)), # borken, dont knwo why. 
        ArchimedeanCopula(2,i𝒲(Pareto(1),5)),
        ArchimedeanCopula(2,i𝒲(LogNormal(3),5)),
    )
    
    for C in Cops
        u = rand(C, 10)
        v = rosenblatt(C, u)
        w = inverse_rosenblatt(C, v)
        @test u ≈ w
    end
end



