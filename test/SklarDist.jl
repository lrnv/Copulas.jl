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
        Z = Copulas.SklarDist(C, ntuple(_->Normal(), d))
        spl10 = rand(rng, C, 10)
        splZ1 = rand(rng, Z)
        splZ10 = rand(rng, Z, 10)
        @test length(splZ1) == d

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

@testset "SklarDist work buffers promote all numeric inputs" begin
    S = SklarDist(IndependentCopula(2), (Normal(), Normal()))
    @test cdf(S, [0, 0]) ≈ 0.25

    Smixed = SklarDist(
        IndependentCopula(2),
        (Normal(0f0, 1f0), Normal(0.0, 1.0)),
    )
    @test cdf(Smixed, Float32[0, 0]) isa Float64
    @test logpdf(Smixed, Float32[0, 0]) isa Float64

    integer_data = [
        -2 -1 0 1 2
        2 1 0 -1 -2
    ]
    Sinteger = fit(
        SklarDist{typeof(S.C),Tuple{Normal,Normal}},
        integer_data,
    )
    @test Sinteger isa SklarDist
    @test all(margin -> margin isa Normal, Sinteger.m)

    Sbig = SklarDist(
        IndependentCopula(2),
        (Normal(big"0", big"1"), Normal(big"0", big"1")),
    )
    xbig = BigFloat[0, 0]
    @test cdf(Sbig, xbig) isa BigFloat
    @test logpdf(Sbig, xbig) isa BigFloat
end
