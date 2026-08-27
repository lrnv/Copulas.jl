# Family-regression layer: the generic Sklar contract lives in
# `contracts/sklar.jl`; only numeric-promotion regressions remain here.
@testset "SklarDist work buffers promote all numeric inputs" begin
    S = SklarDist(IndependentCopula{2}(), (Normal(), Normal()))
    @test cdf(S, [0, 0]) ≈ 0.25

    Smixed = SklarDist(
        IndependentCopula{2}(),
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
        IndependentCopula{2}(),
        (Normal(big"0", big"1"), Normal(big"0", big"1")),
    )
    xbig = BigFloat[0, 0]
    @test cdf(Sbig, xbig) isa BigFloat
    @test logpdf(Sbig, xbig) isa BigFloat
end
