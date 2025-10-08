using Chairmarks
using BenchmarkTools
using Copulas
using StatsBase
using Distributions
using StableRNGs

function exercise(rng, C)
    d = length(C)
    u = rand(rng, C, 5)
    c = cdf(C, u)
    p = pdf(C, u)
    r = rosenblatt(C, u)
    ir = inverse_rosenblatt(C, r)
    return nothing
end
function metrics(C)
    Copulas.τ(C)
    Copulas.ρ(C)
    Copulas.β(C)
    Copulas.γ(C)
    Copulas.ι(C)
    return nothing
end
function fitting(rng, C)
    d = length(C)
    CT = typeof(C)
    u = rand(rng, C, 10)
    for m in Copulas._available_fitting_methods(CT, d)
        fit(CT, u, m)
        fit(CopulaModel, CT, u, m)
    end
    return nothing
end
function conditioning(rng, C)
    d = length(C)
    u = rand(rng, C)
    D = condition(C, 2:d, u[2:d,1])
    cdf(D, rand(rng))
    quantile(D, rand(rng))
    return nothing
end

function run_benches()
    rng = StableRNG(123)
    
    EXAMPLES = unique([
        AMHCopula(3, 0.2),
        ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.GalambosTail(0.7)),
        ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.HuslerReissTail(0.6)),
        ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.LogTail(1.5)),
        ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
        ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.GalambosTail(2.5)),
        ArchimedeanCopula(3, EmpiricalGenerator(randn(rng, 3, 200))),
        AsymGalambosCopula(2, 5.0, 0.8, 0.3),
        AsymLogCopula(2, 1.5, 0.5, 0.2),
        AsymMixedCopula(2, 0.12, 0.13),
        BB10Copula(2, 3.0, 0.8),
        BB1Copula(2, 1.2, 1.5),
        BB2Copula(2, 1.5, 1.8),
        BB3Copula(2, 2.5, 0.5),
        BB6Copula(2, 1.5, 1.4),
        BB7Copula(2, 1.5, 0.4),
        BB8Copula(2, 1.5, 0.6),
        BB9Copula(2, 2.0, 1.5),
        BC2Copula(2, 0.5, 0.5),
        BC2Copula(2, 0.7, 0.3),
        BernsteinCopula(ClaytonCopula(3, 3.3); m=5),
        BernsteinCopula(randn(rng, 3,100); m=5, pseudo_values=false),
        BetaCopula(randn(rng, 3,100)),
        CheckerboardCopula(randn(rng, 3,100); pseudo_values=false),
        ClaytonCopula(2, -0.7),
        ClaytonCopula(4, 3.0),
        Copulas.SubsetCopula(RafteryCopula(3, 0.5), (2,1)),
        CuadrasAugeCopula(2, 0.0),
        CuadrasAugeCopula(2, 0.8),
        EmpiricalCopula(randn(2,20), pseudo_values=false),
        EmpiricalEVCopula(randn(rng, 2,20); method=:pickands, pseudo_values=false),
        FGMCopula(2, 1),
        FGMCopula(3, [0.1,0.2,0.3,0.4]),
        FrankCopula(2, -5),
        FrankCopula(4, 5),
        GalambosCopula(2, 4.3),
        GalambosCopula(2, 120),
        GaussianCopula([1 0.7; 0.7 1]),
        GumbelBarnettCopula(2, 1.0),
        GumbelBarnettCopula(3, 0.35),
        GumbelCopula(2, 1.2),
        GumbelCopula(4, 7.0),
        HuslerReissCopula(2, 3.5),
        IndependentCopula(4),
        InvGaussianCopula(2, 1.0),
        InvGaussianCopula(4, 0.05),
        JoeCopula(2, Inf),
        JoeCopula(3, 7),
        LogCopula(2, 1.5),
        MCopula(4),
        MixedCopula(2, 0.5),
        MOCopula(2, 0.5, 0.5, 0.5),
        PlackettCopula(2.0),
        RafteryCopula(3, 0.5),
        TCopula(4, [1 0.5; 0.5 1]),
        tEVCopula(2, 4.0, 0.5),
        WCopula(2),
    ]);

    G = BenchmarkGroup()

    for C in EXAMPLES # only the first two for the moment to see.
        tp = typeof(C).name.wrapper
        nm = sprint(show, C)
        @show nm
        G["exercise"][tp][nm] = @be exercise(rng, C)
        # G["metrics"][nm] = @be metrics(C)
        # G["fitting"][nm] = @be fitting(rng, C)
        # G["conditioning"][tp][nm] = @be conditioning(rng, C)
    end
    return(G)
end

# PkgBenchmark entrypoint; must define SUITE
const SUITE = run_benches();