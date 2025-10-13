@testset "SklarDist fitting" begin
    # [GenericTests integration]: No. This exercises fitting pathways and RNG; belongs to integration tests for SklarDist rather than generic copula properties.
    
    Random.seed!(rng,123)
    MyD = SklarDist(ClaytonCopula(3,7),[LogNormal(),Pareto(),Beta()]) # with vector and not tuple as input
    u = rand(rng,MyD,1000)
    rand!(rng, MyD,u)
    fit(SklarDist{ClaytonCopula,Tuple{LogNormal,Pareto,Beta}},u)
    @test 1==1
    # loglikelyhood(MyD,u)
end
@testset "Generic API plumbing" begin
    GenericModels = (
        # 3D Gaussian copula with modest correlations
        Copulas.GaussianCopula([1.0 0.3 0.2; 0.3 1.0 0.25; 0.2 0.25 1.0]),
        # 3D Clayton (Archimedean) copula
        Copulas.ClaytonCopula(3, 0.8),
        # 4D Independence copula
        Copulas.IndependentCopula(4),
    )

    for C in GenericModels
        d = length(C)
        Z = Copulas.SklarDist(C, ntuple(_ -> Normal(), d))
        spl10 = rand(rng, C, 10)
        splZ10 = rand(rng, Z, 10)

        # subsetdims should work and agree through SklarDist wrapping
        @testset "subsetdims wiring (d=$(d), $(typeof(C)))" begin
            sC = Copulas.subsetdims(C, (2, 1))
            # Resulting cdf must remain within [0,1] on valid inputs
            @test all(0 .<= Distributions.cdf(sC, spl10[1:2, :]) .<= 1)
            # Subsetting a SklarDist should yield the same copula
            @test sC == Copulas.subsetdims(Z, (2, 1)).C
        end

        # Fit smoke for SklarDist-shaped wrapper
        @testset "fit plumbing (d=$(d), $(typeof(C)))" begin
            r3 = fit(SklarDist{typeof(C), NTuple{d, Normal}}, splZ10)
            @test r3 isa SklarDist
            @test r3.C isa typeof(C)
        end
    end
end
