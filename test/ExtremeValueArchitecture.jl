using Random

@testset "Extreme-value architecture" begin
    @testset "dimension-aware constructors" begin
        for (C, d) in (
            (LogCopula(5, 2.0), 5),
            (GalambosCopula(4, 0.7), 4),
            (HuslerReissCopula(3, 1.0), 3),
            (MixedCopula(4, 0.5), 4),
            (tEVCopula(3, 4.0, 0.2), 3),
        )
            @test length(C) == d
        end

        @test_throws ArgumentError AsymLogCopula(3, 1.5, 0.4, 0.6)
        @test_throws ArgumentError Copulas.ExtremeValueCopula(1, Copulas.GalambosTail(0.7))

        Cind = LogCopula(3, 1.0)
        Cdep = LogCopula(3, Inf)
        @test length(Cind) == 3
        @test length(Cdep) == 3
        @test cdf(Cind, fill(0.5, 3)) ≈ 0.5^3
        @test cdf(Cdep, fill(0.5, 3)) ≈ 0.5
    end

    @testset "parameter-structured constructors" begin
        Γ = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
        Chr = HuslerReissCopula(Γ)
        @test length(Chr) == 3
        @test Chr.tail isa Copulas.HuslerReissVariogramTail

        Γ2 = [0.0 1.0; 1.0 0.0]
        Chr2 = HuslerReissCopula(Γ2)
        @test Chr2.tail isa Copulas.HuslerReissTail
        @test cdf(Chr2, [0.4, 0.7]) ≈ cdf(HuslerReissCopula(2, 2.0), [0.4, 0.7])

        R = [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0]
        Ctev = tEVCopula(4.0, R)
        @test length(Ctev) == 3
        @test Ctev.tail isa Copulas.tEVCorrelationTail

        R2 = [1.0 0.3; 0.3 1.0]
        Ctev2 = tEVCopula(4.0, R2)
        @test Ctev2.tail isa Copulas.tEVTail
        @test cdf(Ctev2, [0.4, 0.7]) ≈ cdf(tEVCopula(2, 4.0, 0.3), [0.4, 0.7])

        Ctawn = TawnCopula(2.0, [0.6, 0.7, 0.8])
        @test length(Ctawn) == 3
        @test Ctawn.tail isa Copulas.TawnTail
        @test length(TawnCopula(2, [0.6, 0.7, 0.8])) == 3
        @test length(tEVCopula(4, R)) == 3
        @test length(AsymGalambosCopula(1, [0.6, 0.7, 0.8])) == 3

        asy = [[0.4], [0.3], [0.6, 0.7]]
        @test length(TawnCopula(2, [2.0], asy)) == 2
        @test length(AsymGalambosCopula(2, [0.7], asy)) == 2

        Cag = AsymGalambosCopula(0.7, [0.6, 0.7, 0.8])
        @test length(Cag) == 3
        @test Cag.tail isa Copulas.AsymGalambosMultiTail

        Cag2 = AsymGalambosCopula(0.7, [0.6, 0.7])
        Cagref = AsymGalambosCopula(2, 0.7, 0.6, 0.7)
        @test cdf(Cag2, [0.4, 0.7]) ≈ cdf(Cagref, [0.4, 0.7])

        @test length(BC2Copula([0.2, 0.5, 0.8])) == 3
        @test BC2Copula([0.2, 0.5]).tail isa Copulas.BC2Tail

        λ = ones(7)
        Cmo = MOCopula(λ)
        @test length(Cmo) == 3
        @test Cmo.tail isa Copulas.MOMultivariateTail
        @test length(MOCopula(3, λ)) == 3

        @test_throws DimensionMismatch HuslerReissCopula(4, Γ)
        @test_throws DimensionMismatch tEVCopula(4, 4.0, R)
        @test_throws DimensionMismatch MOCopula(ones(5))
    end

    @testset "bivariate density specialization" begin
        u = [0.31, 0.67]
        x, y = -log.(u)

        for C in (
            GalambosCopula(2, 0.7),
            HuslerReissCopula(2, 1.0),
            MixedCopula(2, 0.5),
            tEVCopula(2, 4.0, 0.5),
        )
            val, du, dv, dudv = Copulas._biv_der_ℓ(C.tail, (x, y))
            core = -dudv + du * dv
            expected = -val + log(core) + x + y
            @test logpdf(C, u) == expected
        end
    end

    @testset "strong logistic density" begin
        for θ in (2.0, 13.5, 210.0)
            C = LogCopula(2, θ)
            G = Copulas.GumbelCopula(2, θ)
            for u in ([1e-3, 0.99], [0.01, 0.9], [0.99, 0.5], [0.99, 0.99])
                @test logpdf(C, u) ≈ logpdf(G, u) atol=2e-12 rtol=2e-12
            end
        end
    end

    @testset "sampling direct dispatch" begin
        function check_direct_sampler(C, sampler!)
            Xdispatch = zeros(length(C), 32)
            Xdirect = similar(Xdispatch)

            rng_dispatch = Random.Xoshiro(20260820)
            rng_direct = Random.Xoshiro(20260820)

            Distributions._rand!(rng_dispatch, C, Xdispatch)
            sampler!(rng_direct, C, Xdirect)

            @test Xdispatch == Xdirect
        end

        # BivariatePickandsTail default: the bivariate Logistic model uses the native
        # Ghoudi/Pickands sampler.
        check_direct_sampler(
            LogCopula(2, 2.0),
            Copulas._rand_ghoudi!,
        )

        # These families have preferable exact/spectral samplers in d=2.
        check_direct_sampler(
            MixedCopula(2, 0.5),
            (rng, C, X) ->
                Copulas._mixed_rand_multivariate!(rng, C.tail, X),
        )

        check_direct_sampler(
            GalambosCopula(2, 0.7),
            Copulas._rand_galambos_spectral!,
        )

        check_direct_sampler(
            HuslerReissCopula(2, 1.0),
            Copulas._rand_hr_exchangeable!,
        )

        check_direct_sampler(
            tEVCopula(2, 4.0, 0.5),
            (rng, C, X) ->
                Copulas._tev_rand_multivariate!(
                    rng,
                    C.tail.ν,
                    Copulas._tev_exchangeable_correlation(
                        2,
                        C.tail.ρ,
                    ),
                    X,
                ),
        )

        # The same public families remain sampleable in higher dimensions;
        # dispatch selects their concrete multivariate implementation.
        for C in (
            LogCopula(10, 2.0),
            MixedCopula(10, 0.5),
            GalambosCopula(10, 0.7),
            HuslerReissCopula(10, 1.0),
            tEVCopula(10, 4.0, 0.2),
        )
            U = rand(Random.Xoshiro(20260820), C, 16)
            @test size(U) == (10, 16)
            @test all((0 .< U) .& (U .< 1))
        end
    end


    @testset "Galambos inverse dependence-measure boundaries" begin
        @test Copulas.β⁻¹(GalambosCopula, -0.1) == 0.0
        @test Copulas.β⁻¹(GalambosCopula, 0.0) == 0.0
        @test Copulas.β⁻¹(GalambosCopula, 1.0) == Inf

        for θ in (0.1, 0.3, 1.0, 3.0)
            C = GalambosCopula(2, θ)
            @test Copulas.β⁻¹(GalambosCopula, Copulas.β(C)) ≈ θ
        end
    end

end
