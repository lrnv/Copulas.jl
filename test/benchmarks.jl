# This file simply contains a small function to run for every copula. 
# Maybe this could be the precomiple statements, reducing a bit the number of copulas. 
# that would be nice. 

# or, it could be used to find culprits and slower code all around the package. 

using Copulas, Distributions, StatsBase

function main()
    Bestiary = unique([
        AMHCopula(2,-0.6),
        AMHCopula(3,0.2),
        AMHCopula(4,-0.01),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.6), Copulas.GalambosTail(2.5)),
        ArchimedeanCopula(10,𝒲(Dirac(1),10)),
        ArchimedeanCopula(10,𝒲(MixtureModel([Dirac(1), Dirac(2)]),11)),
        ArchimedeanCopula(2, EmpiricalGenerator(randn(4, 150))),
        ArchimedeanCopula(2,𝒲(LogNormal(),2)),
        ArchimedeanCopula(2,𝒲(Pareto(1),5)),
        ArchimedeanCopula(3, EmpiricalGenerator(randn(3, 200))),
        AsymGalambosCopula(2, 0.6, 0.8, 0.2),
        AsymLogCopula(2, 1.0, 0.0, 0.0),
        AsymMixedCopula(2, 0.1, 0.2),
        BB10Copula(2, 4.5, 0.6),
        BB1Copula(2, 2.5, 1.5),
        BB2Copula(2, 2.0, 1.5),
        BB3Copula(2, 3.0, 1.0),
        BB4Copula(2, 3.0, 2.1),
        BB5Copula(2, 5.0, 0.5),
        BB6Copula(2, 2.0, 1.5),
        BB7Copula(2, 2.0, 1.5),
        BB8Copula(2, 1.5, 0.6),
        BB9Copula(2, 2.8, 2.6),
        BC2Copula(2, 0.7,0.3),
        # BernsteinCopula(ArchimaxCopula(2, Copulas.FrankGenerator(0.8), Copulas.HuslerReissTail(0.6)); m=5),
        # BernsteinCopula(ClaytonCopula(3, 3.3); m=5),
        # BernsteinCopula(GalambosCopula(2, 2.5); m=5),
        # BernsteinCopula(GaussianCopula(2, 0.3); m=5),
        # BernsteinCopula(IndependentCopula(4); m=5),
        # BernsteinCopula(randn(2,100), pseudo_values=false),
        # BetaCopula(randn(2,50)),
        # BetaCopula(randn(3,50)),
        CheckerboardCopula(randn(2,10); pseudo_values=false),
        CheckerboardCopula(randn(3,10); pseudo_values=false),
        CheckerboardCopula(randn(4,10); pseudo_values=false),
        ClaytonCopula(2, -0.7),
        ClaytonCopula(2, 7),
        ClaytonCopula(3, -0.36),
        ClaytonCopula(3, 7.3),
        ClaytonCopula(4, -0.22),
        ClaytonCopula(4, 3.7),
        CuadrasAugeCopula(2, 0.2),
        # EmpiricalCopula(randn(2,10),pseudo_values=false),
        # EmpiricalCopula(randn(2,10),pseudo_values=false),
        EmpiricalEVCopula(randn(2,10); method=:cfg, pseudo_values=false),
        EmpiricalEVCopula(randn(2,10); method=:ols, pseudo_values=false),
        EmpiricalEVCopula(randn(2,10); method=:pickands, pseudo_values=false),
        FGMCopula(2, 0.4),
        FGMCopula(2,1),
        FGMCopula(3,[0.1,0.2,0.3,0.4]),
        FrankCopula(2,-5),
        FrankCopula(3,1.0),
        FrankCopula(4,30),
        GalambosCopula(2, 0.3),
        GaussianCopula([1 0.5; 0.5 1]),
        GumbelBarnettCopula(2,0.7),
        # GumbelBarnettCopula(3,0.35), # Dont understadn why its an issue ? 
        # GumbelBarnettCopula(4,0.2),
        GumbelCopula(2, 1.2),
        GumbelCopula(3,1-log(0.2)),
        GumbelCopula(4,1-log(0.3)),
        HuslerReissCopula(2, 1.6287031392529938),
        IndependentCopula(2),
        IndependentCopula(3),
        IndependentCopula(4),
        InvGaussianCopula(2,0.2),
        InvGaussianCopula(3,0.4),
        InvGaussianCopula(4,0.05),
        JoeCopula(2,1-log(0.5)),
        JoeCopula(3,7),
        LogCopula(2, 5.5),
        MCopula(2),
        MCopula(3),
        MCopula(4),
        MixedCopula(2, 0.5),
        MOCopula(2, 0.5960710257852946, 0.3313524247810329, 0.09653466861970061),
        PlackettCopula(0.5),
        RafteryCopula(2, 0.2),
        RafteryCopula(3, 0.5),
        SurvivalCopula(RafteryCopula(2, 0.2), (2,1)),
        TCopula(2, [1 0.7; 0.7 1]),
        tEVCopula(2, 5.466564460573727, -0.6566645244416698),
        WCopula(2),
    ]);

    function exercise(C)
        CT = typeof(C)
        d = length(C)

        # Excercise minimal interface: 
        rand(C)
        spl = rand(C, 3)
        cdf(C, spl)
        pdf(C, spl)
        logpdf(C, spl)
        Copulas.measure(C, zeros(d), spl[:,1])
        inverse_rosenblatt(C, rosenblatt(C, spl))
        
        # Same for the sklardist: 
        X = SklarDist(C, ntuple(_ -> Normal(), d))
        rand(X)
        splX = rand(X, 3)
        cdf(X, splX)
        pdf(X, splX)
        logpdf(X, splX)
        inverse_rosenblatt(X, rosenblatt(X, splX))
        
        if d > 2
            # exercvise the subsetcopula: 
            sC = subsetdims(C, (2,1))
            rand(sC)
            splsC = rand(sC, 3)
            cdf(sC, splsC)
            pdf(sC, splsC)
            logpdf(sC, splsC)
            Copulas.measure(sC, zeros(d), splsC[:,1])
            inverse_rosenblatt(sC, rosenblatt(sC, splsC))

            # exercise the conditional copula: 
            CC1 = condition(C, 1, 0.5)
            rand(CC1)
            splCC1 = rand(CC1, 3)
            cdf(CC1, splCC1)
            pdf(CC1, splCC1)
            logpdf(CC1, splCC1)
            inverse_rosenblatt(CC1, rosenblatt(CC1, splCC1))
        end

        # exercise a distortion: 
        CC2 = condition(C, 2:d, fill(0.5, d-1))
        rand(CC2)
        splCC2 = rand(CC2, 3)
        cdf(CC2, splCC2)
        pdf(CC2, splCC2)
        logpdf(CC2, splCC2)
        quantile(CC2, splCC2)
        
        # # dependence metrics: 
        # Copulas.τ(C)
        # Copulas.ρ(C)
        # Copulas.β(C)
        # Copulas.γ(C)
        # Copulas.ι(C)

        # StatsBase.corkendall(C)
        # StatsBase.corspearman(C)
        # Copulas.corblomqvist(C)
        # Copulas.corgini(C)
        # Copulas.corentropy(C)

        # and finally fitting: 
        # fit(CT, spl)
        # for m in Copulas._available_fitting_methods(CT, d)
        #     fit(CT, spl, m)
        # end
    end

    for C in Bestiary
        @info "$C..."
        exercise(C)
    end
end