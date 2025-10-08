using BenchmarkTools
using Copulas
using Distributions
using StableRNGs

# PkgBenchmark entrypoint; must define SUITE
const SUITE = BenchmarkGroup()
module M
using StableRNGs
const rng = StableRNG(123)
end

# Separate SklarDist benchmarks on a single representative model (Clayton d=5)
let rng = StableRNG(321)
    top = SUITE["sklar"] = BenchmarkGroup()
    C = ClaytonCopula(5, 2.0)
    m = (Distributions.Normal(), Distributions.LogNormal(), Distributions.Gamma(2,2), Distributions.Beta(2,5), Distributions.Uniform())
    S = SklarDist(C, m)
    X = rand(rng, S, 128)
    U = pseudos(X)
    top["rand/128"] = @benchmarkable rand($rng, $S, 128)
    top["cdf/128"]  = @benchmarkable cdf($S, $X)
    top["pdf/128"]  = @benchmarkable pdf($S, $X)
    top["rosenblatt/128"] = @benchmarkable rosenblatt($S, $X)
    V = rand(rng, 5, 128)
    top["inverse_rosenblatt/128"] = @benchmarkable inverse_rosenblatt($S, $V)
    # Fitting pathways (keep light)
    top["fit/ifm"]  = @benchmarkable Distributions.fit(CopulaModel, SklarDist{ClaytonCopula, typeof(m)}, $X; sklar_method=:ifm, copula_method=:itau, summaries=false)
    top["fit/ecdf"] = @benchmarkable Distributions.fit(CopulaModel, SklarDist{ClaytonCopula, typeof(m)}, $X; sklar_method=:ecdf, copula_method=:itau, summaries=false)
end

const EXAMPLES = unique([
    AMHCopula(2,-1.0),
    AMHCopula(2,-rand(M.rng)),
    AMHCopula(2,0.7),
    AMHCopula(2,rand(M.rng)),
    AMHCopula(3,-rand(M.rng)*0.1),
    AMHCopula(3,0.6),
    AMHCopula(3,rand(M.rng)),
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
    ArchimedeanCopula(10,i𝒲(Dirac(1),10)),
    ArchimedeanCopula(10,i𝒲(MixtureModel([Dirac(1), Dirac(2)]),11)),
    ArchimedeanCopula(2, EmpiricalGenerator(randn(M.rng, 4, 150))),
    ArchimedeanCopula(2,i𝒲(LogNormal(),2)),
    ArchimedeanCopula(2,i𝒲(LogNormal(3),5)),
    ArchimedeanCopula(2,i𝒲(Pareto(1),5)),
    ArchimedeanCopula(3, EmpiricalGenerator(randn(M.rng, 3, 200))),
    AsymGalambosCopula(2, 0.1, 0.2, 0.6),
    AsymGalambosCopula(2, 0.6129496106778634, 0.820474440393214, 0.22304578643880224),
    AsymGalambosCopula(2, 10+5*rand(M.rng), 1.0, 1.0),
    AsymGalambosCopula(2, 10+5*rand(M.rng), rand(M.rng), rand(M.rng)),
    AsymGalambosCopula(2, 11.647356700032505, 0.6195348270893413, 0.4197760589260566),
    AsymGalambosCopula(2, 5.0, 0.8, 0.3),
    AsymGalambosCopula(2, 5+4*rand(M.rng), 1.0, 1.0),
    AsymGalambosCopula(2, 5+4*rand(M.rng), rand(M.rng), rand(M.rng)),
    AsymGalambosCopula(2, 8.810168494949659, 0.5987759444612732, 0.5391280234619427),
    AsymGalambosCopula(2, rand(M.rng), 1.0, 1.0),
    AsymGalambosCopula(2, rand(M.rng), rand(M.rng), rand(M.rng)),
    AsymLogCopula(2, 1.0, 0.0, 0.0),
    AsymLogCopula(2, 1.0, 1.0, 1.0),
    AsymLogCopula(2, 1.0, rand(M.rng), rand(M.rng)),
    AsymLogCopula(2, 1.2, 0.3,0.6),
    AsymLogCopula(2, 1.5, 0.5, 0.2),
    AsymLogCopula(2, 1+4*rand(M.rng), 0.0, 0.0),
    AsymLogCopula(2, 1+4*rand(M.rng), 1.0, 1.0),
    AsymLogCopula(2, 1+4*rand(M.rng), rand(M.rng), rand(M.rng)),
    AsymLogCopula(2, 10+5*rand(M.rng), 0.0, 0.0),
    AsymLogCopula(2, 10+5*rand(M.rng), 1.0, 1.0),
    AsymLogCopula(2, 10+5*rand(M.rng), rand(M.rng), rand(M.rng)),
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
    BC2Copula(2, 0.7,0.3),
    BC2Copula(2, 1.0, 0.0),
    BC2Copula(2, 1/2,1/2),
    BC2Copula(2, rand(M.rng), rand(M.rng)),
    BernsteinCopula(ArchimaxCopula(2, Copulas.FrankGenerator(0.8), Copulas.HuslerReissTail(0.6)); m=5),
    BernsteinCopula(ClaytonCopula(3, 3.3); m=5),
    BernsteinCopula(GalambosCopula(2, 2.5); m=5),
    BernsteinCopula(GaussianCopula(2, 0.3); m=5),
    BernsteinCopula(IndependentCopula(4); m=5),
    BernsteinCopula(IndependentCopula(4); m=5),
    BernsteinCopula(randn(M.rng, 2,100), pseudo_values=false),
    BetaCopula(randn(M.rng, 2,100)),
    BetaCopula(randn(M.rng, 3,100)),
    CheckerboardCopula(randn(M.rng, 2,100); pseudo_values=false),
    CheckerboardCopula(randn(M.rng, 3,100); pseudo_values=false),
    CheckerboardCopula(randn(M.rng, 4,100); pseudo_values=false),
    ClaytonCopula(2,-0.7),
    ClaytonCopula(2,-log(rand(M.rng))),
    ClaytonCopula(2,-rand(M.rng)),
    ClaytonCopula(2,7),
    ClaytonCopula(3,-log(rand(M.rng))),
    ClaytonCopula(3,-rand(M.rng)/2),
    ClaytonCopula(4,-log(rand(M.rng))),
    ClaytonCopula(4,-rand(M.rng)/3),
    ClaytonCopula(4,7.),
    Copulas.SubsetCopula(RafteryCopula(3, 0.5), (2,1)),
    CuadrasAugeCopula(2, 0.0),
    CuadrasAugeCopula(2, 0.1),
    CuadrasAugeCopula(2, 0.3437537135972244),
    CuadrasAugeCopula(2, 0.7103550345192344),
    CuadrasAugeCopula(2, 0.8),
    CuadrasAugeCopula(2, 1.0),
    CuadrasAugeCopula(2, rand(M.rng)),
    EmpiricalCopula(randn(2,10),pseudo_values=false),
    EmpiricalCopula(randn(2,20),pseudo_values=false),
    EmpiricalEVCopula(randn(M.rng, 2,10); method=:cfg, pseudo_values=false),
    EmpiricalEVCopula(randn(M.rng, 2,10); method=:ols, pseudo_values=false),
    EmpiricalEVCopula(randn(M.rng, 2,10); method=:pickands, pseudo_values=false),
    EmpiricalEVCopula(randn(M.rng, 2,20); method=:cfg, pseudo_values=false),
    EmpiricalEVCopula(randn(M.rng, 2,20); method=:ols, pseudo_values=false),
    EmpiricalEVCopula(randn(M.rng, 2,20); method=:pickands, pseudo_values=false),
    FGMCopula(2, 0.0),
    FGMCopula(2, rand(M.rng)),
    FGMCopula(2,1),
    FGMCopula(3, [0.3,0.3,0.3,0.3]),
    FGMCopula(3,[0.1,0.2,0.3,0.4]),
    FrankCopula(2,-5),
    FrankCopula(2,0.5),
    FrankCopula(2,1-log(rand(M.rng))),
    FrankCopula(2,1.0),
    FrankCopula(3,1-log(rand(M.rng))),
    FrankCopula(3,1.0),
    FrankCopula(3,12),
    FrankCopula(4,1-log(rand(M.rng))),
    FrankCopula(4,1.0),
    FrankCopula(4,150),
    FrankCopula(4,30),
    FrankCopula(4,37),
    GalambosCopula(2, 0.3),
    GalambosCopula(2, 1+4*rand(M.rng)),
    GalambosCopula(2, 120),
    GalambosCopula(2, 20),
    GalambosCopula(2, 210),
    GalambosCopula(2, 4.3),
    GalambosCopula(2, 5+5*rand(M.rng)),
    GalambosCopula(2, 80),
    GalambosCopula(2, rand(M.rng)),
    GaussianCopula([1 0.5; 0.5 1]),
    GaussianCopula([1 0.7; 0.7 1]),
    GumbelBarnettCopula(2,1.0),
    GumbelBarnettCopula(2,rand(M.rng)),
    GumbelBarnettCopula(3,0.1),
    GumbelBarnettCopula(3,0.35),
    GumbelBarnettCopula(3,rand(M.rng)*0.38),
    GumbelBarnettCopula(4,0.2),
    GumbelCopula(2, 1.2),
    GumbelCopula(2,1-log(rand(M.rng))),
    GumbelCopula(2,8),
    GumbelCopula(3,1-log(rand(M.rng))),
    GumbelCopula(4,1-log(rand(M.rng))),
    GumbelCopula(4,100),
    GumbelCopula(4,20),
    GumbelCopula(4,7),
    HuslerReissCopula(2, 0.1),
    HuslerReissCopula(2, 0.256693308150987),
    HuslerReissCopula(2, 1.6287031392529938),
    HuslerReissCopula(2, 3.5),
    HuslerReissCopula(2, 5.319851350643586),
    IndependentCopula(2),
    IndependentCopula(3),
    InvGaussianCopula(2,-log(rand(M.rng))),
    InvGaussianCopula(2,1.0),
    InvGaussianCopula(2,rand(M.rng)),
    InvGaussianCopula(3,-log(rand(M.rng))),
    InvGaussianCopula(3,rand(M.rng)),
    InvGaussianCopula(4,-log(rand(M.rng))),
    InvGaussianCopula(4,0.05),
    InvGaussianCopula(4,1.0),
    JoeCopula(2,1-log(rand(M.rng))),
    JoeCopula(2,3),
    JoeCopula(2,Inf),
    JoeCopula(3,1-log(rand(M.rng))),
    JoeCopula(3,7),
    JoeCopula(4,1-log(rand(M.rng))),
    LogCopula(2, 1.5),
    LogCopula(2, 1+9*rand(M.rng)),
    LogCopula(2, 5.5),
    MCopula(2),
    MCopula(4),
    MixedCopula(2, 0.0),
    MixedCopula(2, 0.2),
    MixedCopula(2, 0.5),
    MixedCopula(2, 1.0),
    MOCopula(2, 0.1,0.2,0.3),
    MOCopula(2, 0.5, 0.5, 0.5),
    MOCopula(2, 0.5960710257852946, 0.3313524247810329, 0.09653466861970061),
    MOCopula(2, 1.0, 1.0, 1.0),
    MOCopula(2, rand(M.rng), rand(M.rng), rand(M.rng)),
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
    tEVCopula(2, 4+6*rand(M.rng), -0.9+1.9*rand(M.rng)),
    tEVCopula(2, 5.0, -0.5),
    tEVCopula(2, 5.466564460573727, -0.6566645244416698),
    WCopula(2),
])

let rng = StableRNG(123)
    top = SUITE["copulas"] = BenchmarkGroup()
    # helper to get stable display names
    _show_name(C) = sprint(show, C)
    _base_from_show(s::AbstractString) = (m = match(r"^[^\s(]+", s); m === nothing ? s : m.match)
    for C in EXAMPLES
        CT, d = typeof(C), length(C)
        name_full = _show_name(C)
        name_base = _base_from_show(name_full)
        grp = top[name_base] = get(top, name_base, BenchmarkGroup())
        cgrp = grp[name_full] = BenchmarkGroup()

        # Pre-generate 10 points for cdf/pdf/rosenblatt
        u = rand(rng, C, 64)
        v = rand(rng, d, 64) # uniform

        cgrp["sample"] = @benchmarkable rand($rng, $C, 64)
        cgrp["cdf"] = @benchmarkable cdf($C, $u)
        cgrp["pdf"] = @benchmarkable pdf($C, $u)
        cgrp["rosenblatt"] = @benchmarkable rosenblatt($C, $u)
        cgrp["inverse_rosenblatt"] = @benchmarkable inverse_rosenblatt($C, $v)

        mgrp = cgrp["metrics"] = BenchmarkGroup()
        mgrp["τ(C)"] = @benchmarkable Copulas.τ($C)
        mgrp["ρ(C)"] = @benchmarkable Copulas.ρ($C)
        mgrp["β(C)"] = @benchmarkable Copulas.β($C)
        mgrp["γ(C)"] = @benchmarkable Copulas.γ($C)
        mgrp["ι(C)"] = @benchmarkable Copulas.ι($C)
        mgrp["corkendall(C)"] = @benchmarkable StatsBase.corkendall($C)
        mgrp["corspearman(C)"] = @benchmarkable StatsBase.corspearman($C)
        mgrp["corblomqvist(C)"] = @benchmarkable Copulas.corblomqvist($C)
        mgrp["corgini(C)"] = @benchmarkable Copulas.corgini($C)
        mgrp["corentropy(C)"] = @benchmarkable Copulas.corentropy($C)
        
        # Fitting: iterate available methods for this copula type
        fitgrp = cgrp["fit"] = BenchmarkGroup()
        for m in Copulas._available_fitting_methods(CT, d)
            fitgrp[string(m)] = @benchmarkable fit($CT, $u, $m)
        end

        # Benchmarks on conditioning: distortions (any d>=2) and bivariate conditional copulas (if d>3)
        condgrp = cgrp["conditioning"] = BenchmarkGroup()
        D1 = condition(C, 2:d, u[1,2:d]) # condition the conpula on the first sampled point. 
        t = rand($rng, 32)
        condgrp["distortion/cdf/32"] = @benchmarkable Distributions.cdf($D1, $t)
        condgrp["distortion/quantile/32"] = @benchmarkable Distributions.quantile($D1, $t)
        
        if d > 3
            CC = condition(C, 3:d, u[1,:3:d])
            Ucc = rand($rng, CC, 64)
            Vcc = rand($rng, 2, 64)
            condgrp["cc/sample"] = @benchmarkable rand($rng, $CC, 64)
            condgrp["cc/cdf"]    = @benchmarkable cdf($CC, $Ucc)
            condgrp["cc/pdf"]    = @benchmarkable pdf($CC, $Ucc)
            condgrp["cc/rosenblatt"] = @benchmarkable rosenblatt($CC, $Ucc)
            condgrp["cc/inverse_rosenblatt"] = @benchmarkable inverse_rosenblatt($CC, $Vcc)
        end
    end
end
