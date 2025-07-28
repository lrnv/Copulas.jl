@testitem "Check that subsetcopula works as intended" begin
    using Copulas, Distributions

    cops = (
        IndependentCopula(3),
        AMHCopula(3,0.6),
        AMHCopula(4,-0.3),
        ClaytonCopula(2,-0.7),
        ClaytonCopula(3,-0.1),
        ClaytonCopula(4,7.),
        FrankCopula(2,-5.),
        FrankCopula(3,12.),
        FrankCopula(4,6.),
        FrankCopula(4,150.),
        JoeCopula(3,7.),
        GumbelCopula(4,7.),
        GumbelCopula(4,20.),
        GumbelBarnettCopula(3,0.7),
        InvGaussianCopula(4,0.05),
        InvGaussianCopula(3,8.),
        GaussianCopula([1 0.5; 0.5 1]),
        TCopula(4, [1 0.5; 0.5 1]),
        FGMCopula(2,1),
        MCopula(4),
        WCopula(2),
        PlackettCopula(2.0),
        EmpiricalCopula(randn(2,100),pseudo_values=false),
        SurvivalCopula(ClaytonCopula(2,-0.7),(1,2)),
        RafteryCopula(2, 0.2),
        RafteryCopula(3, 0.5),
    )

    for C in cops
        d = length(C)
        D = SklarDist(C, Tuple(LogNormal() for i in 1:d))
        sC = Copulas.subsetdims(C,(1,2))
        sD = Copulas.subsetdims(D,(2,1))
        
        @test isa(Copulas.subsetdims(C,(1,)),Distributions.Uniform)
        @test isa(Copulas.subsetdims(D,1), Distributions.LogNormal)

        # The following methods have to work: 
        u = rand(sC,10)
        v = rand(sD,10)
        cdf(sC,u)
        cdf(sD,v)

        @test sD.C == Copulas.subsetdims(C,(2,1)) # check for coherence. 
    end
end
