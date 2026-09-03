test_constructor_reduction_graph()


@testset "primitive compositional boundary routing" begin
    @test Copulas.limit_kind(Copulas.BB1Generator(Inf, 1.0), Val(2)) === Copulas.M_LIMIT
    @test Copulas.limit_kind(Copulas.BB6Generator(Inf, 1.5), Val(2)) === Copulas.M_LIMIT
    @test Copulas.limit_kind(Copulas.BB6Generator(1.5, Inf), Val(2)) === Copulas.M_LIMIT

    for (seed, C) in (
        (2101, BB1Copula{2}(Inf, 1.0)),
        (2102, BB6Copula{2}(Inf, 1.5)),
        (2103, BB6Copula{2}(1.5, Inf)),
    )
        U = rand(StableRNG(seed), C, 8)
        @test view(U, 1, :) == view(U, 2, :)
    end

    asym_log_ind = (
        Copulas.AsymLogTail(1.0, 0.4, 0.6),
        Copulas.AsymLogTail(1.5, 0.0, 0.6),
        Copulas.AsymLogTail(1.5, 0.4, 0.0),
    )
    for tail in asym_log_ind, t in (0.2, 0.5, 0.8)
        @test Copulas.A(tail, t) == 1.0
        @test Copulas.dA(tail, t) == 0.0
        @test Copulas.d²A(tail, t) == 0.0
    end

    asym_log = Copulas.AsymLogTail(1.7, 1.0, 1.0)
    log_tail = Copulas.LogTail(1.7)
    asym_gal = Copulas.AsymGalambosTail(1.7, 1.0, 1.0)
    gal_tail = Copulas.GalambosTail(1.7)
    for t in (0.2, 0.5, 0.8)
        @test Copulas.A(asym_log, t) ≈ Copulas.A(log_tail, t)
        @test Copulas.dA(asym_log, t) ≈ Copulas.dA(log_tail, t)
        @test Copulas.d²A(asym_log, t) ≈ Copulas.d²A(log_tail, t)
        @test Copulas.A(asym_gal, t) ≈ Copulas.A(gal_tail, t)
        @test Copulas.dA(asym_gal, t) ≈ Copulas.dA(gal_tail, t)
        @test Copulas.d²A(asym_gal, t) ≈ Copulas.d²A(gal_tail, t)
    end

    inactive_gal = Copulas.AsymGalambosTail(1.5, 0.0, 0.6)
    @test Copulas.A(inactive_gal, 0.37) == 1.0
    @test Copulas.dA(inactive_gal, 0.37) == 0.0
    @test Copulas.d²A(inactive_gal, 0.37) == 0.0

    for (seed, C) in (
        (2111, AsymLogCopula{2}(Inf, 1.0, 1.0)),
        (2112, AsymGalambosCopula{2}(Inf, 1.0, 1.0)),
    )
        @test Copulas.limit_kind(C.tail, Val(2)) === Copulas.M_LIMIT
        U = rand(StableRNG(seed), C, 8)
        @test view(U, 1, :) == view(U, 2, :)
    end
end


@testset "remaining EV algebraic boundary routing" begin
    x3 = (0.2, 0.7, 1.1)

    mixed0 = Copulas.MixedTail(0.0)
    @test Copulas.ℓ(mixed0, x3) == sum(x3)

    asym_mixed0 = Copulas.AsymMixedTail(0.0, 0.0)
    mixed = Copulas.MixedTail(0.6)
    asym_mixed = Copulas.AsymMixedTail(0.6, 0.0)
    for t in (0.2, 0.5, 0.8)
        @test Copulas.A(asym_mixed0, t) == 1.0
        @test Copulas.dA(asym_mixed0, t) == 0.0
        @test Copulas.d²A(asym_mixed0, t) == 0.0
        @test Copulas.A(asym_mixed, t) ≈ Copulas.A(mixed, t)
        @test Copulas.dA(asym_mixed, t) ≈ Copulas.dA(mixed, t)
        @test Copulas.d²A(asym_mixed, t) ≈ Copulas.d²A(mixed, t)
    end

    tawn = Copulas.TawnTail(1.7, ones(3))
    @test Copulas.ℓ(tawn, x3) ≈ Copulas.ℓ(Copulas.LogTail(1.7), x3)
    tawn_M = Copulas.TawnTail(Inf, ones(3))
    @test Copulas.limit_kind(tawn_M, Val(3)) === Copulas.M_LIMIT
    U = rand(StableRNG(2121), Copulas.ExtremeValueCopula(3, tawn_M), 8)
    @test all(view(U, i, :) == view(U, 1, :) for i in 2:3)

    ca0 = Copulas.CuadrasAugeTail(0.0)
    ca1 = Copulas.CuadrasAugeTail(1.0)
    @test Copulas.copula_measure_style(Copulas.ExtremeValueCopula(2, ca0)) isa
          Copulas.AbsolutelyContinuousMeasure
    @test Copulas.limit_kind(ca1, Val(2)) === Copulas.M_LIMIT

    rng1, rng2 = StableRNG(2131), StableRNG(2131)
    @test rand(rng1, Copulas.ExtremeValueCopula(2, ca0), 8) ==
          rand(rng2, IndependentCopula{2}(), 8)
    Uca = rand(StableRNG(2132), Copulas.ExtremeValueCopula(2, ca1), 8)
    @test view(Uca, 1, :) == view(Uca, 2, :)
end

@testset "EV genuine degeneracy routing" begin
    x3 = (0.2, 0.7, 1.1)
    u3 = [0.23, 0.61, 0.84]

    hr_ind = Copulas.HuslerReissTail(0.0)
    @test Copulas.limit_kind(hr_ind, Val(3)) === Copulas.Π_LIMIT
    @test Copulas.ℓ(hr_ind, x3) == sum(x3)
    @test Copulas.dA(hr_ind, 0.37) == 0.0
    @test Copulas.d²A(hr_ind, 0.37) == 0.0
    @test Copulas.ellpartial(hr_ind, x3, (1,)) == 1.0
    @test Copulas.ellpartial(hr_ind, x3, (1, 2)) == 0.0

    C_hr_ind = Copulas.ExtremeValueCopula(3, hr_ind)
    @test cdf(C_hr_ind, u3) ≈ prod(u3)
    rng1, rng2 = StableRNG(2140), StableRNG(2140)
    @test rand(rng1, C_hr_ind, 8) == rand(rng2, IndependentCopula{3}(), 8)

    hr_limits = (
        Copulas.HuslerReissTail(Inf),
        Copulas.HuslerReissTail(zeros(3, 3)),
    )
    for (k, tail) in enumerate(hr_limits)
        @test Copulas.limit_kind(tail, Val(3)) === Copulas.M_LIMIT
        @test Copulas.ℓ(tail, x3) == maximum(x3)
        C = Copulas.ExtremeValueCopula(3, tail)
        @test Copulas.copula_measure_style(C) isa Copulas.NonAbsolutelyContinuousMeasure
        @test cdf(C, u3) == minimum(u3)
        U = rand(StableRNG(2140 + k), C, 8)
        @test all(view(U, i, :) == view(U, 1, :) for i in 2:3)
    end

    tev_limits = (
        Copulas.tEVTail(4.0, 1.0),
        Copulas.tEVTail(4.0, ones(3, 3)),
    )
    for (k, tail) in enumerate(tev_limits)
        @test Copulas.limit_kind(tail, Val(3)) === Copulas.M_LIMIT
        @test Copulas.ℓ(tail, x3) == maximum(x3)
        @test Copulas.A(tail, 0.37) == 0.63
        C = Copulas.ExtremeValueCopula(3, tail)
        @test Copulas.copula_measure_style(C) isa Copulas.NonAbsolutelyContinuousMeasure
        @test cdf(C, u3) == minimum(u3)
        U = rand(StableRNG(2150 + k), C, 8)
        @test all(view(U, i, :) == view(U, 1, :) for i in 2:3)
    end
end

@testset "Archimedean boundary architecture closure" begin
    @test !isdefined(Copulas, :_reduced_generator)
    @test !isdefined(Copulas, :_reduced_tail)

    t = 0.73
    u = 0.41
    independence_generators = (
        Copulas.AMHGenerator(0.0),
        Copulas.GumbelBarnettGenerator(0.0),
        Copulas.InvGaussianGenerator(0.0),
        Copulas.JoeGenerator(1.0),
    )
    for G in independence_generators
        @test Copulas.ϕ(G, t) ≈ exp(-t)
        @test Copulas.ϕ⁻¹(G, u) ≈ -log(u)
        @test Copulas.ϕ⁽¹⁾(G, t) ≈ -exp(-t)
        @test Copulas.ϕ⁻¹⁽¹⁾(G, u) ≈ -inv(u)
        @test Copulas.ϕ⁽ᵏ⁾(G, 3, t) ≈ -exp(-t)
    end

    Rgb = Copulas.𝒲₋₁(Copulas.GumbelBarnettGenerator(0.0), 3)
    @test Rgb isa Distributions.Gamma
    @test Distributions.params(Rgb) == (3.0, 1.0)
    @test Copulas.frailty(Copulas.InvGaussianGenerator(0.0)) isa Distributions.Dirac

    point = [0.31, 0.72]
    for C in (
        AMHCopula{2}(0.0),
        GumbelBarnettCopula{2}(0.0),
        InvGaussianCopula{2}(0.0),
        JoeCopula{2}(1.0),
    )
        @test cdf(C, point) ≈ prod(point)
        @test logpdf(C, point) ≈ 0.0
        U = rand(StableRNG(2160), C, 8)
        @test all(isfinite, U)
        @test all(x -> 0 <= x <= 1, U)
    end

    bb7_M = BB7Copula{2}(1.0, Inf)
    bb8_M = BB8Copula{2}(Inf, 1.0)
    @test Copulas.limit_kind(bb7_M.G, Val(2)) === Copulas.M_LIMIT
    @test Copulas.limit_kind(bb8_M.G, Val(2)) === Copulas.M_LIMIT
    for (seed, C) in ((2171, bb7_M), (2172, bb8_M))
        @test cdf(C, point) == minimum(point)
        U = rand(StableRNG(seed), C, 8)
        @test view(U, 1, :) == view(U, 2, :)
    end
end
