using Random

# Test-only tail implementing exactly the minimal multivariate EV contract: ℓ.
struct ADOnlyLogisticTail{T} <: Copulas.Tail
    θ::T
end

Copulas.ℓ(tail::ADOnlyLogisticTail, x) =
    sum(xi^tail.θ for xi in x)^(inv(tail.θ))

@testset "Extreme-value architecture" begin
    @testset "canonical dimension constructors" begin
        for (Ctyped, Cruntime, d) in (
            (LogCopula{5}(2.0), LogCopula(5, 2.0), 5),
            (GalambosCopula{4}(0.7), GalambosCopula(4, 0.7), 4),
            (HuslerReissCopula{3}(1.0), HuslerReissCopula(3, 1.0), 3),
            (MixedCopula{4}(0.5), MixedCopula(4, 0.5), 4),
            (CuadrasAugeCopula{4}(0.5), CuadrasAugeCopula(4, 0.5), 4),
            (tEVCopula{3}(4.0, 0.2), tEVCopula(3, 4.0, 0.2), 3),
        )
            @test length(Ctyped) == d
            @test length(Cruntime) == d
            @test typeof(Ctyped) == typeof(Cruntime)
            @test Distributions.params(Ctyped) == Distributions.params(Cruntime)
        end

        # Integer-valued parameters remain parameters once d is encoded.
        @test Distributions.params(LogCopula{2}(2)).θ == 2.0
        @test Distributions.params(tEVCopula{2}(4, 0.2)).ν == 4

        # Generic fitting reconstructs from concrete typeof(C) through the
        # internal hook, not through an ambiguous FamilyCopula{d,T}(d, ...) call.
        C0 = GalambosCopula{2}(0.9)
        C1 = Copulas._construct_from_params(typeof(C0), 2, 0.9)
        @test typeof(C1) == typeof(C0)
        @test Distributions.params(C1) == Distributions.params(C0)

        # Scalar-parameter families no longer infer an implicit d=2.
        @test_throws MethodError GalambosCopula(2.3)
        @test_throws MethodError MixedCopula(0.5)

        @test_throws ArgumentError AsymLogCopula(3, 1.5, 0.4, 0.6)
        @test_throws ArgumentError Copulas.ExtremeValueCopula(
            1,
            Copulas.GalambosTail(0.7),
        )

        @test cdf(
            AsymLogCopula{2}(1.5, 0.4, 0.6),
            [0.31, 0.67],
        ) ≈ cdf(
            AsymLogCopula(2, 1.5, 0.4, 0.6),
            [0.31, 0.67],
        )
        @test cdf(
            BC2Copula{2}(0.2, 0.5),
            [0.31, 0.67],
        ) ≈ cdf(
            BC2Copula(2, 0.2, 0.5),
            [0.31, 0.67],
        )
        @test cdf(
            MOCopula{2}(1.0, 2.0, 0.5),
            [0.31, 0.67],
        ) ≈ cdf(
            MOCopula(2, 1.0, 2.0, 0.5),
            [0.31, 0.67],
        )

        Cind = LogCopula{3}(1.0)
        Cdep = LogCopula{3}(Inf)
        @test length(Cind) == 3
        @test length(Cdep) == 3
        @test cdf(Cind, fill(0.5, 3)) ≈ 0.5^3
        @test cdf(Cdep, fill(0.5, 3)) ≈ 0.5
    end

    @testset "parameter-structured constructors" begin
        Γ = [0.0 1.0 1.0; 1.0 0.0 1.0; 1.0 1.0 0.0]
        Chr_typed = HuslerReissCopula{3}(Γ)
        Chr_runtime = HuslerReissCopula(3, Γ)
        Chr_inferred = HuslerReissCopula(Γ)
        @test typeof(Chr_typed) == typeof(Chr_runtime) == typeof(Chr_inferred)
        @test Chr_typed.tail isa Copulas.HuslerReissVariogramTail

        Γ2 = [0.0 1.0; 1.0 0.0]
        Chr2 = HuslerReissCopula{2}(Γ2)
        @test Chr2.tail isa Copulas.HuslerReissTail
        @test cdf(Chr2, [0.4, 0.7]) ≈
              cdf(HuslerReissCopula{2}(2.0), [0.4, 0.7])

        R = [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0]
        Ctev_typed = tEVCopula{3}(4.0, R)
        Ctev_runtime = tEVCopula(3, 4.0, R)
        Ctev_inferred = tEVCopula(4.0, R)
        @test typeof(Ctev_typed) == typeof(Ctev_runtime) == typeof(Ctev_inferred)
        @test Ctev_typed.tail isa Copulas.tEVCorrelationTail

        R2 = [1.0 0.3; 0.3 1.0]
        Ctev2 = tEVCopula{2}(4.0, R2)
        @test Ctev2.tail isa Copulas.tEVTail
        @test cdf(Ctev2, [0.4, 0.7]) ≈
              cdf(tEVCopula{2}(4.0, 0.3), [0.4, 0.7])

        weights = [0.6, 0.7, 0.8]
        Ctawn_typed = TawnCopula{3}(2.0, weights)
        Ctawn_runtime = TawnCopula(3, 2.0, weights)
        Ctawn_inferred = TawnCopula(2.0, weights)
        @test typeof(Ctawn_typed) ==
              typeof(Ctawn_runtime) ==
              typeof(Ctawn_inferred)
        @test Ctawn_typed.tail isa Copulas.TawnTail
        @test length(TawnCopula{3}(2, weights)) == 3
        @test length(TawnCopula(2, weights)) == 3

        asy = [[0.4], [0.3], [0.6, 0.7]]
        dep_tawn = [2.0]
        Ctawn_full_typed = TawnCopula{2}(dep_tawn, asy)
        Ctawn_full_runtime = TawnCopula(2, dep_tawn, asy)
        Ctawn_full_inferred = TawnCopula(dep_tawn, asy)
        @test typeof(Ctawn_full_typed) ==
              typeof(Ctawn_full_runtime) ==
              typeof(Ctawn_full_inferred)

        Cag_typed = AsymGalambosCopula{3}(0.7, weights)
        Cag_runtime = AsymGalambosCopula(3, 0.7, weights)
        Cag_inferred = AsymGalambosCopula(0.7, weights)
        @test typeof(Cag_typed) == typeof(Cag_runtime) == typeof(Cag_inferred)
        @test Cag_typed.tail isa Copulas.AsymGalambosMultiTail
        @test length(AsymGalambosCopula{3}(1, weights)) == 3
        @test length(AsymGalambosCopula(1, weights)) == 3

        dep_gal = [0.7]
        Cag_full_typed = AsymGalambosCopula{2}(dep_gal, asy)
        Cag_full_runtime = AsymGalambosCopula(2, dep_gal, asy)
        Cag_full_inferred = AsymGalambosCopula(dep_gal, asy)
        @test typeof(Cag_full_typed) ==
              typeof(Cag_full_runtime) ==
              typeof(Cag_full_inferred)

        Cag2 = AsymGalambosCopula{2}(0.7, [0.6, 0.7])
        Cagref = AsymGalambosCopula{2}(0.7, 0.6, 0.7)
        @test cdf(Cag2, [0.4, 0.7]) ≈ cdf(Cagref, [0.4, 0.7])

        a = [0.2, 0.5, 0.8]
        Cbc_typed = BC2Copula{3}(a)
        Cbc_runtime = BC2Copula(3, a)
        Cbc_inferred = BC2Copula(a)
        @test typeof(Cbc_typed) == typeof(Cbc_runtime) == typeof(Cbc_inferred)
        @test BC2Copula{2}([0.2, 0.5]).tail isa Copulas.BC2Tail

        λ = ones(7)
        Cmo_typed = MOCopula{3}(λ)
        Cmo_runtime = MOCopula(3, λ)
        Cmo_inferred = MOCopula(λ)
        @test typeof(Cmo_typed) == typeof(Cmo_runtime) == typeof(Cmo_inferred)
        @test Cmo_typed.tail isa Copulas.MOMultivariateTail

        Uemp = [
            0.20 0.40 0.70
            0.30 0.60 0.80
            0.25 0.55 0.75
        ]
        Cemp_typed = EmpiricalEVMultivariateCopula{3}(Uemp; degree=1)
        Cemp_runtime = EmpiricalEVMultivariateCopula(3, Uemp; degree=1)
        Cemp_inferred = EmpiricalEVMultivariateCopula(Uemp; degree=1)
        @test typeof(Cemp_typed) ==
              typeof(Cemp_runtime) ==
              typeof(Cemp_inferred)

        @test_throws DimensionMismatch HuslerReissCopula{4}(Γ)
        @test_throws DimensionMismatch HuslerReissCopula(4, Γ)
        @test_throws DimensionMismatch tEVCopula{4}(4.0, R)
        @test_throws DimensionMismatch tEVCopula(4, 4.0, R)
        @test_throws DimensionMismatch TawnCopula{4}(2.0, weights)
        @test_throws DimensionMismatch AsymGalambosCopula{4}(0.7, weights)
        @test_throws DimensionMismatch BC2Copula{4}(a)
        @test_throws DimensionMismatch MOCopula{4}(λ)
        @test_throws DimensionMismatch EmpiricalEVMultivariateCopula{4}(
            Uemp;
            degree=1,
        )
        @test_throws DimensionMismatch MOCopula(ones(5))
    end

    @testset "shared generic mixed-partial interface" begin
        f(z) = z[1]^2 * z[2]^3 + z[3]
        z = [0.4, 0.7, 1.1]
        expected12 = 6 * z[1] * z[2]^2
        @test Copulas._mixed_partial(f, z, (1, 2)) ≈ expected12
        @test Copulas._mixed_partial(f, Tuple(z), [1, 2]) ≈ expected12

        θ = 2.0
        tail = ADOnlyLogisticTail(θ)
        x = (0.4, 0.7, 1.1)
        S = sum(xi^θ for xi in x)
        for I in ((1,), (1, 3), (1, 2, 3))
            k = length(I)
            coeff = k == 1 ? one(θ) : prod(1 - j * θ for j in 1:(k - 1))
            expected = coeff * S^(inv(θ) - k) * prod(x[i]^(θ - 1) for i in I)
            got = Copulas.ellpartial(tail, x, I)
            @test got ≈ expected atol=3e-12 rtol=3e-11
            sign, logabs = Copulas._ellpartial_signlog(tail, x, I)
            @test sign == (signbit(expected) ? -1 : 1)
            @test exp(logabs) ≈ abs(expected) atol=3e-12 rtol=3e-11
        end

        Cgeneric = Copulas.ExtremeValueCopula{3}(tail)
        Canalytic = LogCopula{3}(θ)
        u = [0.31, 0.57, 0.82]
        @test logpdf(Cgeneric, u) ≈ logpdf(Canalytic, u) atol=2e-10 rtol=2e-10
    end
    @testset "multivariate EV generic conditioning and Rosenblatt" begin
        # The generic conditioning framework is dimension-agnostic. Smooth EV
        # families whose CDF/STDF path is ForwardDiff-compatible inherit it.
        for C in (
            LogCopula{3}(2.0),
            GalambosCopula{3}(0.7),
        )
            # Conditioning on two coordinates leaves a univariate distortion.
            D = condition(C, (1, 2), (0.31, 0.58))
            @test D isa Copulas.Distortion
            for α in (0.2, 0.6, 0.85)
                q = Distributions.quantile(D, α)
                @test Distributions.cdf(D, q) ≈ α atol=2e-7 rtol=2e-7
            end

            # Conditioning on one coordinate leaves a two-dimensional
            # conditional distribution.
            H = condition(C, (1,), (0.31,))
            @test H isa SklarDist
            h = Distributions.cdf(H, [0.42, 0.73])
            @test isfinite(h)
            @test 0.0 <= h <= 1.0

            # Rosenblatt and its inverse use those same sequential conditional
            # distortions in d > 2.
            u = [0.21, 0.53, 0.74]
            s = rosenblatt(C, u)
            @test all(isfinite, s)
            @test all(x -> 0.0 <= x <= 1.0, s)
            @test inverse_rosenblatt(C, s) ≈ u atol=2e-7 rtol=2e-7
        end
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
