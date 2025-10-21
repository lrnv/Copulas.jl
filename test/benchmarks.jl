# This file simply contains a small function to run for every copula. 
# Maybe this could be the precomiple statements, reducing a bit the number of copulas. 
# that would be nice. 

# or, it could be used to find culprits and slower code all around the package. 

using Copulas, Distributions, StatsBase

function main()
    Bestiary = unique([
        AMHCopula(2,-0.6),
        AMHCopula(2,-1.0),
        AMHCopula(2,0.7),
        AMHCopula(3,-0.003),
        AMHCopula(3,0.2),
        AMHCopula(4,-0.01),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.BB1Generator(2.0, 2.0), Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(3.0),  Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.FrankGenerator(0.8),    Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(4.0),   Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.JoeGenerator(1.2),      Copulas.LogTail(2.0)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.GalambosTail(2.5)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.HuslerReissTail(1.8)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.LogTail(2.0)),
        ArchimedeanCopula(10,𝒲(Dirac(1),10)),
        ArchimedeanCopula(10,𝒲(MixtureModel([Dirac(1), Dirac(2)]),11)),
        ArchimedeanCopula(2, EmpiricalGenerator(randn(4, 150))),
        ArchimedeanCopula(2,𝒲(LogNormal(),2)),
        ArchimedeanCopula(2,𝒲(Pareto(1),5)),
        ArchimedeanCopula(3, EmpiricalGenerator(randn(3, 200))),
        AsymGalambosCopula(2, 0.1, 0.2, 0.6),
        AsymGalambosCopula(2, 0.3, 0.8, 0.1),
        AsymGalambosCopula(2, 0.6129496106778634, 0.820474440393214, 0.22304578643880224),
        AsymGalambosCopula(2, 0.9, 1.0, 1.0),
        AsymGalambosCopula(2, 10+5*0.3, 1.0, 1.0),
        AsymGalambosCopula(2, 10+5*0.7, 0.2, 0.9),
        AsymGalambosCopula(2, 11.647356700032505, 0.6195348270893413, 0.4197760589260566),
        AsymGalambosCopula(2, 5.0, 0.8, 0.3),
        AsymGalambosCopula(2, 5+4*0.1, 0.2, 0.6),
        AsymGalambosCopula(2, 5+4*0.4, 1.0, 1.0),
        AsymGalambosCopula(2, 8.810168494949659, 0.5987759444612732, 0.5391280234619427),
        AsymLogCopula(2, 1.0, 0.0, 0.0),
        AsymLogCopula(2, 1.0, 0.1, 0.6),
        AsymLogCopula(2, 1.0, 1.0, 1.0),
        AsymLogCopula(2, 1.2, 0.3,0.6),
        AsymLogCopula(2, 1.5, 0.5, 0.2),
        AsymLogCopula(2, 1+4*0.01, 1.0, 1.0),
        AsymLogCopula(2, 1+4*0.2, 0.3, 0.4),
        AsymLogCopula(2, 1+4*0.9, 0.0, 0.0),
        AsymLogCopula(2, 10+5*0.5, 0.0, 0.0),
        AsymLogCopula(2, 10+5*0.6, 1.0, 1.0),
        AsymLogCopula(2, 10+5*0.7, 0.8, 0.2),
        AsymMixedCopula(2, 0.1, 0.2),
        AsymMixedCopula(2, 0.12, 0.13),
        BB10Copula(2, 1.5, 0.7),
        BB10Copula(2, 3.0, 0.8),
        BB10Copula(2, 4.5, 0.6),
        BB1Copula(2, 0.35, 1.0),
        BB1Copula(2, 1.2, 1.5),
        BB1Copula(2, 2.5, 1.5),
        BB2Copula(2, 1.2, 0.5),
        BB2Copula(2, 1.5, 1.8),
        BB2Copula(2, 2.0, 1.5),
        BB3Copula(2, 2.0, 1.5),
        BB3Copula(2, 2.5, 0.5),
        BB3Copula(2, 3.0, 1.0),
        BB4Copula(2, 0.50, 1.60),
        BB4Copula(2, 2.50, 0.40),
        BB4Copula(2, 3.0, 2.1),
        BB5Copula(2, 1.50, 1.60),
        BB5Copula(2, 2.50, 0.40),
        BB5Copula(2, 5.0, 0.5),
        BB6Copula(2, 1.2, 1.6),
        BB6Copula(2, 1.5, 1.4),
        BB6Copula(2, 2.0, 1.5),
        BB7Copula(2, 1.2, 1.6),
        BB7Copula(2, 1.5, 0.4),
        BB7Copula(2, 2.0, 1.5),
        BB8Copula(2, 1.2, 0.4),
        BB8Copula(2, 1.5, 0.6),
        BB8Copula(2, 2.5, 0.8),
        BB9Copula(2, 1.5, 2.4),
        BB9Copula(2, 2.0, 1.5),
        BB9Copula(2, 2.8, 2.6),
        BC2Copula(2, 0.5, 0.3),
        BC2Copula(2, 0.5, 0.5),
        BC2Copula(2, 0.5516353577049822, 0.33689370624999193),
        BC2Copula(2, 0.6, 0.8),
        BC2Copula(2, 0.7,0.3),
        BC2Copula(2, 1.0, 0.0),
        BC2Copula(2, 1/2,1/2),
        # BernsteinCopula(ArchimaxCopula(2, Copulas.FrankGenerator(0.8), Copulas.HuslerReissTail(0.6)); m=5),
        # BernsteinCopula(ClaytonCopula(3, 3.3); m=5),
        # BernsteinCopula(GalambosCopula(2, 2.5); m=5),
        # BernsteinCopula(GaussianCopula(2, 0.3); m=5),
        # BernsteinCopula(IndependentCopula(4); m=5),
        # BernsteinCopula(randn(2,100), pseudo_values=false),
        # BetaCopula(randn(2,50)),
        # BetaCopula(randn(3,50)),
        CheckerboardCopula(randn(2,50); pseudo_values=false),
        CheckerboardCopula(randn(3,50); pseudo_values=false),
        CheckerboardCopula(randn(4,50); pseudo_values=false),
        ClaytonCopula(2, -0.7),
        ClaytonCopula(2, 0.3),
        ClaytonCopula(2, 0.9),
        ClaytonCopula(2, 7),
        ClaytonCopula(3, -0.36),
        ClaytonCopula(3, 7.3),
        ClaytonCopula(4, -0.22),
        ClaytonCopula(4, 3.7),
        ClaytonCopula(4,7.),
        Copulas.SubsetCopula(RafteryCopula(3, 0.5), (2,1)),
        CuadrasAugeCopula(2, 0.0),
        CuadrasAugeCopula(2, 0.1),
        CuadrasAugeCopula(2, 0.2),
        CuadrasAugeCopula(2, 0.3437537135972244),
        CuadrasAugeCopula(2, 0.7103550345192344),
        CuadrasAugeCopula(2, 0.8),
        CuadrasAugeCopula(2, 1.0),
        # EmpiricalCopula(randn(2,50),pseudo_values=false),
        # EmpiricalCopula(randn(2,50),pseudo_values=false),
        EmpiricalEVCopula(randn(2,50); method=:cfg, pseudo_values=false),
        EmpiricalEVCopula(randn(2,50); method=:ols, pseudo_values=false),
        EmpiricalEVCopula(randn(2,50); method=:pickands, pseudo_values=false),
        FGMCopula(2, 0.0),
        FGMCopula(2, 0.4),
        FGMCopula(2,1),
        FGMCopula(3, [0.3,0.3,0.3,0.3]),
        FGMCopula(3,[0.1,0.2,0.3,0.4]),
        FrankCopula(2,-5),
        FrankCopula(2,0.5),
        FrankCopula(2,1-log(0.9)),
        FrankCopula(2,1.0),
        FrankCopula(3,1-log(0.1)),
        FrankCopula(3,1.0),
        FrankCopula(3,12),
        FrankCopula(4,1-log(0.3)),
        FrankCopula(4,1.0),
        # FrankCopula(4,150),
        FrankCopula(4,30),
        FrankCopula(4,37),
        GalambosCopula(2, 0.3),
        GalambosCopula(2, 0.7),
        GalambosCopula(2, 1+4*0.5),
        GalambosCopula(2, 120),
        GalambosCopula(2, 20),
        GalambosCopula(2, 210),
        GalambosCopula(2, 4.3),
        GalambosCopula(2, 8),
        GalambosCopula(2, 80),
        GaussianCopula([1 0.5; 0.5 1]),
        GaussianCopula([1 0.7; 0.7 1]),
        GumbelBarnettCopula(2,0.7),
        GumbelBarnettCopula(2,1.0),
        # GumbelBarnettCopula(3,0.35), # Dont understadn why its an issue ? 
        # GumbelBarnettCopula(4,0.2),
        GumbelCopula(2, 1.2),
        GumbelCopula(2,1-log(0.9)),
        GumbelCopula(2,8),
        # GumbelCopula(3,1-log(0.2)),
        # GumbelCopula(4,1-log(0.3)),
        # GumbelCopula(4,100),
        # GumbelCopula(4,20),
        # GumbelCopula(4,7),
        HuslerReissCopula(2, 0.1),
        HuslerReissCopula(2, 0.256693308150987),
        HuslerReissCopula(2, 1.6287031392529938),
        HuslerReissCopula(2, 3.5),
        HuslerReissCopula(2, 5.319851350643586),
        IndependentCopula(2),
        IndependentCopula(3),
        InvGaussianCopula(2,-log(0.9)),
        InvGaussianCopula(2,0.2),
        InvGaussianCopula(2,1.0),
        InvGaussianCopula(3,-log(0.6)),
        InvGaussianCopula(3,0.4),
        InvGaussianCopula(4,-log(0.1)),
        InvGaussianCopula(4,0.05),
        InvGaussianCopula(4,1.0),
        JoeCopula(2,1-log(0.5)),
        JoeCopula(2,3),
        JoeCopula(2,Inf),
        JoeCopula(3,1-log(0.3)),
        JoeCopula(3,7),
        JoeCopula(4,1-log(0.1)),
        LogCopula(2, 1.5),
        LogCopula(2, 1+9*0.4),
        LogCopula(2, 5.5),
        MCopula(2),
        MCopula(4),
        MixedCopula(2, 0.0),
        MixedCopula(2, 0.2),
        MixedCopula(2, 0.5),
        MixedCopula(2, 1.0),
        MOCopula(2, 0.1, 0.5, 0.9),
        MOCopula(2, 0.1,0.2,0.3),
        MOCopula(2, 0.5, 0.5, 0.5),
        MOCopula(2, 0.5960710257852946, 0.3313524247810329, 0.09653466861970061),
        MOCopula(2, 1.0, 1.0, 1.0),
        PlackettCopula(0.5),
        PlackettCopula(0.8),
        PlackettCopula(2.0),
        RafteryCopula(2, 0.2),
        RafteryCopula(3, 0.5),
        SurvivalCopula(ClaytonCopula(2,-0.7),(1,2)),
        SurvivalCopula(RafteryCopula(2, 0.2), (2,1)),
        TCopula(2, [1 0.7; 0.7 1]),
        TCopula(20,[1 -0.5; -0.5 1]),
        TCopula(4, [1 0.5; 0.5 1]),
        tEVCopula(2, 10.0, 1.0),
        tEVCopula(2, 2.0, 0.5),
        tEVCopula(2, 3.0, 0.0),
        tEVCopula(2, 4.0, 0.5),
        tEVCopula(2, 4+6*0.5, -0.9+1.9*0.3),
        tEVCopula(2, 5.0, -0.5),
        tEVCopula(2, 5.466564460573727, -0.6566645244416698),
        WCopula(2),
    ]);

    function exercise_a_cop(C)
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
        exercise_a_cop(C)
    end
end