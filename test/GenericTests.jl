@testitem "Generic tests on every copulas" begin
    using HypothesisTests, Distributions, Random, WilliamsonTransforms
    using InteractiveUtils
    using ForwardDiff
    using StatsBase: corkendall
    using StableRNGs
    rng = StableRNG(123)

    bestiary = unique([
        AMHCopula(2, -1.0),
        AMHCopula(2, -rand(rng)),
        AMHCopula(2, 0.0),
        AMHCopula(2, 0.7),
        AMHCopula(2, rand(rng)),
        AMHCopula(3, -1.0),
        AMHCopula(3, -rand(rng)),
        AMHCopula(3, 0.0),
        AMHCopula(3, 0.6),
        AMHCopula(3, rand(rng)),
        ArchimedeanCopula(2, i𝒲(LogNormal(), 2)),
        AsymGalambosCopula(0.0, [0.0, 0.0]),
        AsymGalambosCopula(0.0, [1.0, 1.0]),
        AsymGalambosCopula(0.0, [rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))]),
        AsymGalambosCopula(0.1, [0.2, 0.6]),
        AsymGalambosCopula(0.6129496106778634, [0.820474440393214, 0.22304578643880224]),
        AsymGalambosCopula(11.647356700032505, [0.6195348270893413, 0.4197760589260566]),
        AsymGalambosCopula(5.0, [0.8, 0.3]),
        AsymGalambosCopula(8.810168494949659, [0.5987759444612732, 0.5391280234619427]),
        AsymGalambosCopula(rand(rng, Uniform(10.0, 15.0)), [0.0, 0.0]),
        AsymGalambosCopula(rand(rng, Uniform(10.0, 15.0)), [1.0, 1.0]),
        AsymGalambosCopula(
            rand(rng, Uniform(10.0, 15.0)),
            [rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))],
        ),
        AsymGalambosCopula(rand(rng, Uniform(5.0, 9.0)), [0.0, 0.0]),
        AsymGalambosCopula(rand(rng, Uniform(5.0, 9.0)), [1.0, 1.0]),
        AsymGalambosCopula(
            rand(rng, Uniform(5.0, 9.0)),
            [rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))],
        ),
        AsymGalambosCopula(rand(rng), [0.0, 0.0]),
        AsymGalambosCopula(rand(rng), [1.0, 1.0]),
        AsymGalambosCopula(rand(rng), [rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))]),
        AsymLogCopula(1.0, [0.8360692316060747, 0.68704221750134]),
        AsymLogCopula(rand(rng, Uniform(1.0, 5.0)), [1.0, 1.0]),
        AsymLogCopula(1.0, [1.0, 1.0]),
        AsymLogCopula(1.0, [rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))]),
        AsymLogCopula(1.2, [0.3, 0.6]),
        AsymLogCopula(1.5, [0.5, 0.2]),
        AsymLogCopula(12.29006035397328, [0.7036713552821277, 0.7858058549340399]),
        AsymLogCopula(12.29006035397328, [1.0, 1.0]),
        AsymLogCopula(2.8130363753722403, [0.3539590866764071, 0.15146985093210463]),
        AsymLogCopula(2.8130363753722403, [1.0, 1.0]),
        AsymLogCopula(
            rand(rng, Uniform(1.0, 5.0)),
            [rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))],
        ),
        AsymLogCopula(rand(rng, Uniform(10.0, 15.0)), [1.0, 1.0]),
        AsymLogCopula(
            rand(rng, Uniform(10.0, 15.0)),
            [rand(rng, Uniform(0, 1)), rand(rng, Uniform(0, 1))],
        ),
        AsymLogCopula(1.0, [0.0, 0.0]),
        AsymLogCopula(12.29006035397328, [0.0, 0.0]),
        AsymLogCopula(2.8130363753722403, [0.0, 0.0]),
        AsymLogCopula(rand(rng, Uniform(1.0, 5.0)), [0.0, 0.0]),
        AsymLogCopula(rand(rng, Uniform(10.0, 15.0)), [0.0, 0.0]),
        AsymMixedCopula([0.2, 0.4]),
        AsymMixedCopula([0.0, 0.0]),
        AsymMixedCopula([0.1, 0.2]),
        AsymMixedCopula([0.1, 0.2]),
        AsymMixedCopula([0.2, 0.4]),
        BC2Copula(1.0, 0.0),
        BC2Copula(0.5, 0.3),
        BC2Copula(0.5, 0.5),
        BC2Copula(0.5516353577049822, 0.33689370624999193),
        BC2Copula(0.7, 0.3),
        BC2Copula(1 / 2, 1 / 2),
        BC2Copula(rand(rng), rand(rng)),
        ClaytonCopula(2, -0.7),
        ClaytonCopula(2, -1),
        ClaytonCopula(2, -log(rand(rng))),
        ClaytonCopula(2, -rand(rng)),
        ClaytonCopula(2, 0.0),
        ClaytonCopula(2, 7),
        ClaytonCopula(2, Inf),
        ClaytonCopula(3, -0.1),
        ClaytonCopula(3, -log(rand(rng))),
        ClaytonCopula(3, -rand(rng) / 2),
        ClaytonCopula(3, 0.0),
        ClaytonCopula(3, Inf),
        ClaytonCopula(4, -log(rand(rng))),
        ClaytonCopula(4, -rand(rng) / 3),
        ClaytonCopula(4, 0.0),
        ClaytonCopula(4, 7.0),
        ClaytonCopula(4, 7.0),
        ClaytonCopula(4, 7),
        ClaytonCopula(4, Inf),
        CuadrasAugeCopula(0.0),
        CuadrasAugeCopula(0.1),
        CuadrasAugeCopula(0.3437537135972244),
        CuadrasAugeCopula(0.7103550345192344),
        CuadrasAugeCopula(0.8),
        CuadrasAugeCopula(1.0),
        CuadrasAugeCopula(rand(rng)),
        EmpiricalCopula(randn(2, 100); pseudo_values=false),
        EmpiricalCopula(randn(2, 200); pseudo_values=false),
        FGMCopula(2, 0.0),
        FGMCopula(2, rand(rng)),
        FGMCopula(2, 1),
        FGMCopula(3, [0.3, 0.3, 0.3, 0.3]),
        FGMCopula(3, [0.1, 0.2, 0.3, 0.4]),
        FrankCopula(2, 0.5),
        FrankCopula(2, -Inf),
        FrankCopula(2, 1 - log(rand(rng))),
        FrankCopula(2, 1.0),
        FrankCopula(2, Inf),
        FrankCopula(2, log(rand(rng))),
        FrankCopula(3, 1 - log(rand(rng))),
        FrankCopula(3, 1.0),
        FrankCopula(3, 12),
        FrankCopula(3, Inf),
        FrankCopula(4, 1 - log(rand(rng))),
        FrankCopula(4, 1.0),
        FrankCopula(4, 150),
        FrankCopula(4, 30),
        FrankCopula(4, 37),
        FrankCopula(4, 6.0),
        FrankCopula(4, 6),
        FrankCopula(4, Inf),
        GalambosCopula(0.0),
        GalambosCopula(0.3),
        GalambosCopula(120),
        GalambosCopula(20),
        GalambosCopula(210),
        GalambosCopula(4.3),
        GalambosCopula(80),
        GalambosCopula(Inf),
        GalambosCopula(rand(rng, Uniform(5.0, 10.0))),
        GalambosCopula(rand(rng)),
        GalambosCopula(rand(rng, Uniform(1.0, 5.0))),
        GaussianCopula([1 0.5; 0.5 1]),
        GaussianCopula([1 0.7; 0.7 1]),
        GumbelBarnettCopula(2, 0.0),
        GumbelBarnettCopula(2, 1.0),
        GumbelBarnettCopula(2, rand(rng)),
        GumbelBarnettCopula(3, 0.0),
        GumbelBarnettCopula(3, 0.7),
        GumbelBarnettCopula(3, 1.0),
        GumbelBarnettCopula(3, rand(rng)),
        GumbelBarnettCopula(4, 0.0),
        GumbelBarnettCopula(4, 1.0),
        GumbelBarnettCopula(4, rand(rng)),
        GumbelCopula(2, 1.2),
        GumbelCopula(2, 1 - log(rand(rng))),
        GumbelCopula(2, 1.0),
        GumbelCopula(2, 1), # should be equivalent to an independent copula.
        GumbelCopula(2, 8),
        GumbelCopula(2, Inf),
        GumbelCopula(3, 1 - log(rand(rng))),
        GumbelCopula(3, 1.0),
        GumbelCopula(3, Inf),
        GumbelCopula(4, 1 - log(rand(rng))),
        GumbelCopula(4, 1.0),
        GumbelCopula(4, 20),
        GumbelCopula(4, 7),
        GumbelCopula(4, 100),
        GumbelCopula(4, Inf),
        HuslerReissCopula(0.0),
        HuslerReissCopula(0.1),
        HuslerReissCopula(0.256693308150987),
        HuslerReissCopula(rand(rng, Uniform(1.0, 5.0))),
        HuslerReissCopula(rand(rng, Uniform(5.0, 10.0))),
        HuslerReissCopula(rand(rng)),
        HuslerReissCopula(3.5),
        HuslerReissCopula(1.6287031392529938),
        HuslerReissCopula(5.319851350643586),
        HuslerReissCopula(Inf),
        IndependentCopula(2),
        IndependentCopula(3),
        InvGaussianCopula(2, -log(rand(rng))),
        InvGaussianCopula(2, 1.0),
        InvGaussianCopula(2, rand(rng)),
        InvGaussianCopula(3, -log(rand(rng))),
        InvGaussianCopula(3, rand(rng)),
        InvGaussianCopula(4, -log(rand(rng))),
        InvGaussianCopula(4, 0.05),
        InvGaussianCopula(4, 1.0),
        JoeCopula(2, 1 - log(rand(rng))),
        JoeCopula(2, 1.0),
        JoeCopula(2, 3),
        JoeCopula(2, Inf),
        JoeCopula(3, 1 - log(rand(rng))),
        JoeCopula(3, 1.0),
        JoeCopula(3, 7),
        JoeCopula(4, 1 - log(rand(rng))),
        LogCopula(1.0),
        LogCopula(rand(rng, Uniform(1.0, 10.0))),
        LogCopula(1.5),
        LogCopula(5.5),
        LogCopula(Inf),
        MCopula(2),
        MCopula(4),
        MixedCopula(0.0),
        MixedCopula(0.2),
        MixedCopula(0.5),
        MixedCopula(1.0),
        MOCopula(0.5960710257852946, 0.3313524247810329, 0.09653466861970061),
        MOCopula(0.1, 0.2, 0.3),
        MOCopula(0.5, 0.5, 0.5),
        MOCopula(1.0, 1.0, 1.0),
        MOCopula(rand(rng), rand(rng), rand(rng)),
        PlackettCopula(0.5),
        PlackettCopula(0.8),
        PlackettCopula(2.0),
        RafteryCopula(2, 0.2),
        RafteryCopula(3, 0.5),
        SurvivalCopula(ClaytonCopula(2, -0.7), (1, 2)),
        Copulas.SubsetCopula(RafteryCopula(3, 0.5), (2, 1)),
        TCopula(2, [1 0.7; 0.7 1]),
        TCopula(4, [1 0.5; 0.5 1]),
        TCopula(4, [1 0.5; 0.5 1]),
        tEVCopula(10.0, 1.0),  # ν > 0 y ρ == 1, MCopula
        tEVCopula(3.0, 0.0),  # ν > 0 y ρ == 0, Independent Copula
        tEVCopula(4.0, 0.5),
        tEVCopula(5.0, -0.5),  # ν > 0 y -1 < ρ <= 1
        tEVCopula(5.466564460573727, -0.6566645244416698),
        tEVCopula(rand(rng, Uniform(4.0, 10.0)), rand(rng, Uniform(-0.9, 1.0))),
        tEVCopula(2.0, 0.5),  # ν > 0 y -1 < ρ <= 1
        WCopula(2),
    ])

    #### Ensure we got everyone.

    function _subtypes(type::Type)
        out = Any[]
        return _subtypes!(out, type)
    end
    function _subtypes!(out, type::Type)
        if !isabstracttype(type)
            push!(out, type)
        else
            foreach(T -> _subtypes!(out, T), InteractiveUtils.subtypes(type))
        end
        return out
    end
    @testset "Generic tests: missing copulas in the list" begin
        @test all(any(isa(C, CT) for C in bestiary) for CT in _subtypes(Copulas.Copula))
        @test all(
            any(isa(C.G, TG) for C in bestiary if typeof(C) <: Copulas.ArchimedeanCopula)
            for TG in _subtypes(Copulas.Generator)
        )
    end

    #### methods to numerically derivate the pdf from the cdf :
    # Not really efficient as in some cases this return zero while the true pdf is clearly not zero.
    function _v(u, j, uj)
        return [(i == j ? uj : u[i]) for i in 1:length(u)]
    end
    function _der(j, C, u)
        if j == 1
            return ForwardDiff.derivative(u1 -> cdf(C, _v(u, 1, u1)), u[1])
        else
            return ForwardDiff.derivative(uj -> _der(j - 1, C, _v(u, j, uj)), u[j])
        end
    end
    function get_numerical_pdf(C, u)
        return _der(length(C), C, u)
    end

    # Filter on archimedeans for fitting tests.
    function is_archimedean_with_agenerator(CT)
        if CT <: ArchimedeanCopula
            GT = Copulas.generatorof(CT)
            if !isnothing(GT)
                if !(GT <: Copulas.ZeroVariateGenerator)
                    if !(GT <: Copulas.WilliamsonGenerator)
                        return true
                    end
                end
            end
        end
        return false
    end

    for C in bestiary
        Random.seed!(rng, 123) # set seed.
        @testset "$(C) - Generic tests" begin
            d = length(C)
            CT = typeof(C)
            D = SklarDist(C, [Normal() for i in 1:length(C)])
            spl10 = rand(rng, C, 10)
            spl1000 = rand(rng, C, 10000)

            @testset "Sampling shapes" begin
                @test length(rand(rng, C)) == d # to sample only one value
                @test size(spl10) == (d, 10)
            end

            @testset "Check samples in  [0,1]" begin
                @test all(0 .<= spl10 .<= 1)
                @test all(0 .<= spl1000 .<= 1)
            end

            @testset "Check cdf > 0 and bounday conditions." begin
                @test iszero(cdf(C, zeros(d)))
                @test isone(cdf(C, ones(d)))
                @test 0 <= cdf(C, rand(rng, d)) <= 1
                @test all(0 .<= spl10 .<= 1)
                @test cdf(D, zeros(d)) >= 0
            end

            # Test that the measure function works correctly:
            @testset "Check measure >0 and boundary conditions" begin
                @test Copulas.measure(C, zeros(d), ones(d)) ≈ 1
                u_mes = ones(d) * 0.2
                v_mes = ones(d) * 0.4
                @test Copulas.measure(C, u_mes, v_mes) >= 0
            end

            @testset "Check cdf's margins uniformity" begin
                if !(CT <: EmpiricalCopula) # this one is not a true copula :)
                    for i in 1:d
                        for val in [0, 1, 0.5, rand(rng, 5)...]
                            u = ones(d)
                            u[i] = val
                            @test cdf(C, u) ≈ val atol = 1e-5
                        end
                        # extra check for zeros:
                        u = rand(rng, d)
                        u[i] = 0
                        @test iszero(cdf(C, u))
                    end
                end
            end

            @testset "Check pdf > 0" begin
                has_pdf(C) = applicable(Distributions._logpdf, C, rand(length(C), 3))
                if has_pdf(C)
                    @test pdf(C, ones(length(C)) / 2) >= 0
                    @test all(pdf(C, spl10) .>= 0)
                    @test pdf(D, zeros(d)) >= 0 # also the sklardist pdf should be strictily positive there.
                end
            end

            if hasmethod(Copulas.A, (typeof(C), Float64))
                @testset "ExtremeValues: Check A's boundary conditions" begin
                    @test Copulas.A(C, 0.0) == 1
                    @test Copulas.A(C, 1.0) == 1
                    t = rand(rng)
                    A_value = Copulas.A(C, t)
                    @test 0.0 <= A_value <= 1.0
                    @test isapprox(A_value, max(t, 1 - t); atol=1e-6) ||
                        A_value >= max(t, 1 - t)
                    @test A_value <= 1.0
                end

                # Maybe we should also check the \ell and C connections to A ?
                # to be certain to catch any mistake.

            end

            @testset "Match theoretical and empirical kendall taus" begin
                K = corkendall(spl1000')
                Kth = corkendall(C)
                @test all(-1 .<= Kth .<= 1)
                @test all(isapprox.(Kth, K; atol=0.1))
            end

            # Extra checks, only for archimedeans.
            if is_archimedean_with_agenerator(CT)
                if applicable(Copulas.τ, C.G) && applicable(Copulas.τ⁻¹, typeof(C.G), 1.0)
                    @testset "Archimedean with generator: check τ ∘ τ_inv == Id" begin
                        tau = Copulas.τ(C)
                        @test Copulas.τ(Copulas.generatorof(CT)(Copulas.τ⁻¹(CT, tau))) ≈ tau
                    end
                end

                if applicable(Copulas.ρ, C.G) && applicable(Copulas.ρ⁻¹, typeof(C.G), 1.0)
                    @testset "Archimedean with generator: check ρ ∘ ρ_inv == Id" begin
                        rho = Copulas.ρ(C)
                        @test -1 <= rho <= 1
                        @test Copulas.ρ(Copulas.generatorof(CT)(Copulas.ρ⁻¹(CT, rho))) ≈ rho
                    end
                end

                @testset "Check fit() runs" begin
                    fit(CT, spl10)
                end

                @testset "Check ϕ ∘ ϕ_inv == Id" begin
                    for x in 0:0.1:1
                        @test Copulas.ϕ⁻¹(C, Copulas.ϕ(C, x)) ≈ x
                    end
                end

                if !isa(C, ClaytonCopula)
                    @testset "Check that williamson_dist and ϕ⁻¹ actually corresponds" begin
                        splW_method1 = dropdims(
                            sum(Copulas.ϕ⁻¹.(Ref(C), rand(rng, C, 1000)); dims=1); dims=1
                        )
                        splW_method2 = rand(rng, Copulas.williamson_dist(C), 1000)
                        @test pvalue(
                            ApproximateTwoSampleKSTest(splW_method1, splW_method2);
                            tail=:right,
                        ) > 0.01
                    end
                end
            end
        end
    end
end
