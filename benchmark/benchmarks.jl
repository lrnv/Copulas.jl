using BenchmarkTools
using Copulas
using StatsBase
using Distributions
using StableRNGs
const rng = StableRNG(123)

# PkgBenchmark entrypoint; must define SUITE
const SUITE = BenchmarkGroup()


const EXAMPLES = unique([
    # AMH: keep one non-2D representative
    AMHCopula(3, 0.6),

    # Archimax: cover each generator family and each tail type at least once (no full cartesian product)
    ArchimaxCopula(2, Copulas.BB1Generator(1.3, 1.4), Copulas.GalambosTail(0.7)),
    ArchimaxCopula(2, Copulas.ClaytonGenerator(1.5),  Copulas.HuslerReissTail(0.6)),
    ArchimaxCopula(2, Copulas.FrankGenerator(6.0),    Copulas.LogTail(1.5)),
    ArchimaxCopula(2, Copulas.GumbelGenerator(2.0),   Copulas.AsymGalambosTail(0.35, 0.65, 0.3)),
    ArchimaxCopula(2, Copulas.JoeGenerator(2.5),      Copulas.GalambosTail(2.5)),

    # Archimedean: one empirical generator (non-2D)
    ArchimedeanCopula(3, EmpiricalGenerator(randn(rng, 3, 200))),

    # Asymmetric families: one representative each
    AsymGalambosCopula(2, 5.0, 0.8, 0.3),
    AsymLogCopula(2, 1.5, 0.5, 0.2),
    AsymMixedCopula(2, 0.12, 0.13),

    # BB families: one per family
    BB10Copula(2, 3.0, 0.8),
    BB1Copula(2, 1.2, 1.5),
    BB2Copula(2, 1.5, 1.8),
    BB3Copula(2, 2.5, 0.5),
    BB6Copula(2, 1.5, 1.4),
    BB7Copula(2, 1.5, 0.4),
    BB8Copula(2, 1.5, 0.6),
    BB9Copula(2, 2.0, 1.5),

    # BC2: symmetric and asymmetric
    BC2Copula(2, 0.5, 0.5),
    BC2Copula(2, 0.7, 0.3),

    # Bernstein copulas: a couple of bases
    BernsteinCopula(ClaytonCopula(3, 3.3); m=5),
    BernsteinCopula(GaussianCopula(2, 0.3); m=5),

    # Beta & Checkerboard: prefer non-2D where available
    BetaCopula(randn(rng, 3,100)),
    CheckerboardCopula(randn(rng, 3,100); pseudo_values=false),

    # Clayton: keep one negative and one positive; plus one non-2D
    ClaytonCopula(2, -0.7),
    ClaytonCopula(4, 3.0),

    # Subset copula
    Copulas.SubsetCopula(RafteryCopula(3, 0.5), (2,1)),

    # Cuadras-Auge: two extremes
    CuadrasAugeCopula(2, 0.0),
    CuadrasAugeCopula(2, 0.8),

    # Empirical copulas
    EmpiricalCopula(randn(2,20), pseudo_values=false),
    EmpiricalEVCopula(randn(rng, 2,20); method=:pickands, pseudo_values=false),

    # FGM: one 2D max and one 3D vector case
    FGMCopula(2, 1),
    FGMCopula(3, [0.1,0.2,0.3,0.4]),

    # Frank: negative and positive, with a non-2D example
    FrankCopula(2, -5),
    FrankCopula(4, 30),

    # Galambos: small and large parameter
    GalambosCopula(2, 4.3),
    GalambosCopula(2, 120),

    # Gaussian
    GaussianCopula([1 0.7; 0.7 1]),

    # Gumbel-Barnett
    GumbelBarnettCopula(2, 1.0),
    GumbelBarnettCopula(3, 0.35),

    # Gumbel: 2D and non-2D
    GumbelCopula(2, 1.2),
    GumbelCopula(4, 7.0),

    # Hüsler–Reiss
    HuslerReissCopula(2, 3.5),

    # Independent: prefer non-2D
    IndependentCopula(4),

    # Inverse Gaussian: 2D and non-2D
    InvGaussianCopula(2, 1.0),
    InvGaussianCopula(4, 0.05),

    # Joe: include ∞ special case and one non-2D
    JoeCopula(2, Inf),
    JoeCopula(3, 7),

    # Log copula
    LogCopula(2, 1.5),

    # Marshall–Olkin variants
    MCopula(4),
    MixedCopula(2, 0.5),
    MOCopula(2, 0.5, 0.5, 0.5),

    # Plackett
    PlackettCopula(2.0),

    # Raftery: prefer non-2D
    RafteryCopula(3, 0.5),

    # Survival
    SurvivalCopula(ClaytonCopula(2, -0.7), (1,2)),

    # t-Copula: prefer non-2D
    TCopula(4, [1 0.5; 0.5 1]),

    # t-EV: one representative
    tEVCopula(2, 4.0, 0.5),

    # W copula
    WCopula(2),
]);

let rng = StableRNG(123)
    top = SUITE
    # helper to get stable display names
    _show_name(C) = sprint(show, C)
    # helper to ensure nested groups exist and return them
    function ensure_group!(grp::BenchmarkGroup, key)
        if haskey(grp, key)
            return grp[key]
        else
            sub = BenchmarkGroup()
            grp[key] = sub
            return sub
        end
    end

    for C in EXAMPLES # only the first two for the moment to see.
        CT, d = typeof(C), length(C)
        name_full = _show_name(C)

        # Pre-generate inputs for cdf/pdf/rosenblatt and inverse
        u = rand(rng, C, 64)
        v = rand(rng, d, 64) # uniform

        # Core ops grouped first by op, then by copula name
        ensure_group!(top, "rng")[name_full]  = @benchmarkable rand($rng, $C, 64)
        ensure_group!(top, "cdf")[name_full]  = @benchmarkable cdf($C, $u)
        ensure_group!(top, "pdf")[name_full]  = @benchmarkable pdf($C, $u)
        ensure_group!(top, "rbt")[name_full]  = @benchmarkable rosenblatt($C, $u)
        ensure_group!(top, "irbt")[name_full] = @benchmarkable inverse_rosenblatt($C, $v)

        # Metrics: scalar and matrix-style
        mgrp = ensure_group!(top, "metrics")
        ensure_group!(mgrp, "τ")[name_full]     = @benchmarkable Copulas.τ($C)
        ensure_group!(mgrp, "ρ")[name_full]     = @benchmarkable Copulas.ρ($C)
        ensure_group!(mgrp, "β")[name_full]     = @benchmarkable Copulas.β($C)
        ensure_group!(mgrp, "γ")[name_full]     = @benchmarkable Copulas.γ($C)
        ensure_group!(mgrp, "ι")[name_full]     = @benchmarkable Copulas.ι($C)
        # ensure_group!(mgrp, "corτ")[name_full]  = @benchmarkable StatsBase.corkendall($C)
        # ensure_group!(mgrp, "corρ")[name_full]  = @benchmarkable StatsBase.corspearman($C)
        # ensure_group!(mgrp, "corβ")[name_full]  = @benchmarkable Copulas.corblomqvist($C)
        # ensure_group!(mgrp, "corγ")[name_full]  = @benchmarkable Copulas.corgini($C)
        # ensure_group!(mgrp, "corι")[name_full]  = @benchmarkable Copulas.corentropy($C)

        # # Fitting: iterate available methods for this copula type
        # fitgrp = ensure_group!(top, "fit")
        # for m in Copulas._available_fitting_methods(CT, d)
        #     ensure_group!(fitgrp, string(m))[name_full] = @benchmarkable fit($CT, $u, $m)
        # end

        # # Conditioning: distortions (any d>=2) and bivariate conditional copulas (if d>3)
        # condgrp = ensure_group!(top, "cond")
        # D1 = condition(C, 2:d, u[2:d,1]) # condition the copula on the first sampled point.
        # t = rand(rng, 32)
        # ensure_group!(condgrp, "dst/cdf")[name_full]      = @benchmarkable Distributions.cdf($D1, $t)
        # ensure_group!(condgrp, "dst/quantile")[name_full] = @benchmarkable Distributions.quantile($D1, $t)

        # if d > 3
        #     CC = condition(C, 3:d, u[3:d,1])
        #     Ucc = rand(rng, CC, 64)
        #     Vcc = rand(rng, 2, 64)
        #     ensure_group!(condgrp, "cc/rng")[name_full]  = @benchmarkable rand($rng, $CC, 64)
        #     ensure_group!(condgrp, "cc/cdf")[name_full]  = @benchmarkable cdf($CC, $Ucc)
        #     ensure_group!(condgrp, "cc/pdf")[name_full]  = @benchmarkable pdf($CC, $Ucc)
        #     ensure_group!(condgrp, "cc/rbt")[name_full]  = @benchmarkable rosenblatt($CC, $Ucc)
        #     ensure_group!(condgrp, "cc/irbt")[name_full] = @benchmarkable inverse_rosenblatt($CC, $Vcc)
        # end
    end
end


# Separate SklarDist benchmarks on a single representative model (Clayton d=5)
# let rng = StableRNG(321)
#     top = SUITE["sklar"] = BenchmarkGroup()
#     C = ClaytonCopula(5, 2.0)
#     m = (Distributions.Normal(), Distributions.LogNormal(), Distributions.Gamma(2,2), Distributions.Beta(2,5), Distributions.Uniform())
#     S = SklarDist(C, m)
#     X = rand(rng, S, 128)
#     U = pseudos(X)
#     top["rand/128"] = @benchmarkable rand($rng, $S, 128)
#     top["cdf/128"]  = @benchmarkable cdf($S, $X)
#     top["pdf/128"]  = @benchmarkable pdf($S, $X)
#     top["rosenblatt/128"] = @benchmarkable rosenblatt($S, $X)
#     V = rand(rng, 5, 128)
#     top["inverse_rosenblatt/128"] = @benchmarkable inverse_rosenblatt($S, $V)
#     # Fitting pathways (keep light)
#     top["fit/ifm"]  = @benchmarkable Distributions.fit(CopulaModel, SklarDist{ClaytonCopula, typeof(m)}, $X; sklar_method=:ifm, copula_method=:itau, summaries=false)
#     top["fit/ecdf"] = @benchmarkable Distributions.fit(CopulaModel, SklarDist{ClaytonCopula, typeof(m)}, $X; sklar_method=:ecdf, copula_method=:itau, summaries=false)
# end