@testitem "Test samples have uniform maginals in [0,1]" begin
    using HypothesisTests, Distributions, Random
    using StableRNGs
    rng = StableRNG(123)
    cops = (
        # true represent the fact that cdf(williamson_dist(C),x) is defined or not. 
        AMHCopula(3,0.6),
        AMHCopula(4,-0.3),
        ClaytonCopula(2,-0.7),
        ClaytonCopula(3,-0.1),
        ClaytonCopula(4,7),
        FrankCopula(2,-5),
        FrankCopula(3,12),
        FrankCopula(4,6),
        FrankCopula(4,150),
        JoeCopula(3,7),
        GumbelCopula(4,7),
        GumbelCopula(4,20),
        GumbelCopula(4,100),
        GumbelBarnettCopula(3,0.7),
        InvGaussianCopula(4,0.05),
        InvGaussianCopula(3,8),
        GaussianCopula([1 0.5; 0.5 1]),
        TCopula(4, [1 0.5; 0.5 1]),
        FGMCopula(2,1),
        MCopula(4),
        PlackettCopula(2.0),
        # Others ? Yes probably others too ! 
    )
    n = 1000
    U = Uniform(0,1)
    for C in cops
        nfail = 0
        d = length(C)
        @show C
        spl = rand(rng,C,n)
        @assert all(0 <= x <= 1 for x in spl)
        for i in 1:d
            @test pvalue(ApproximateOneSampleKSTest(spl[i,:], U),tail=:right) > 0.01 # quite weak but enough at these samples sizes to detect really bad behaviors.
        end
    end
end