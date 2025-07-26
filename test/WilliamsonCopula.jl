
@testitem "williamson test" begin
    using Distributions, Random
    using StableRNGs
    rng = StableRNG(12)

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

    Cops = (
        ArchimedeanCopula(10,i𝒲(MixtureModel([Dirac(1), Dirac(2)]),11)), 
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



