
@testitem "Test of coherence for archimedeans" begin
    using HypothesisTests, Distributions
    cops = (
        # true represent the fact that cdf(williamson_dist(C),x) is defined or not. 
        (AMHCopula(3,0.6), true),
        (AMHCopula(4,-0.3), true),
        (ClaytonCopula(2,-0.7), true),
        (ClaytonCopula(3,-0.1), true),
        (ClaytonCopula(4,7), true),
        (FrankCopula(3,-12), false),
        (FrankCopula(4,6), false),
        (JoeCopula(3,7), false),
        (GumbelCopula(4,7), false),
    )
    n = 1000
    spl = rand(n)
    spl2 = rand(n)
    for (C,will_dist_has_a_cdf_implemented) in cops
        spl .= dropdims(sum(Copulas.ϕ.(Ref(C),rand(C,n)),dims=1),dims=1)
        will_dist = Copulas.williamson_dist(C)
        if will_dist_has_a_cdf_implemented
            pval = pvalue(ExactOneSampleKSTest(spl, will_dist))
            @test pval < 0.05
        end
        # even without a cdf we can still test approximately:
        spl2 .= rand(will_dist,n)
        pval2 = pvalue(ApproximateTwoSampleKSTest(spl,spl2))
        @test pval2 < 0.05
    end

end