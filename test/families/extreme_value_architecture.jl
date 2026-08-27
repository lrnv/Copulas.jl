# Family-regression layer: developer-level extreme-value extension,
# automatic-differentiation, sampler, and fallback-dispatch regressions.
using Random

@testset "Extreme-value architecture" begin
    @testset "canonical dimension constructors" begin
        # Integer-valued parameters remain parameters once d is encoded.
        @test Distributions.params(LogCopula{2}(2)).θ == 2.0
        @test Distributions.params(MixedCopula{2}(1)).θ == 1.0
        @test Distributions.params(HuslerReissCopula{2}(1)).θ == 1.0
        @test Distributions.params(tEVCopula{2}(4, 0.2)).ν == 4

        Cint = LogCopula{2}(2)
        @test Distributions.params(typeof(Cint)(2)).θ == 2.0
        @test Distributions.params(LogCopula(2, 2)).θ == 2.0

        # Scalar-parameter families no longer infer an implicit d=2.
        @test_throws MethodError GalambosCopula(2.3)
        @test_throws MethodError MixedCopula(0.5)

        @test Copulas.AsymLogTail(1.0, 0.4, 0.6) isa Copulas.NoTail
        @test Copulas.AsymLogTail(1.5, 0.0, 0.6) isa Copulas.NoTail
        @test Copulas.AsymLogTail(1.5, 1.0, 1.0) isa Copulas.LogTail

        @test_throws ArgumentError AsymLogCopula(3, 1.5, 0.4, 0.6)
        @test_throws ArgumentError Copulas.ExtremeValueCopula(
            1,
            Copulas.GalambosTail(0.7),
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
        @test Chr_typed.tail isa Copulas.HuslerReissTail{<:AbstractMatrix}

        Γ2 = [0.0 1.0; 1.0 0.0]
        Chr2 = HuslerReissCopula{2}(Γ2)
        @test Chr2.tail isa Copulas.HuslerReissTail{<:AbstractMatrix}
        @test Distributions.params(Chr2).Γ == Γ2
        @test cdf(Chr2, [0.4, 0.7]) ≈
              cdf(HuslerReissCopula{2}(2.0), [0.4, 0.7])

        Chr2scalar = HuslerReissCopula{2}(2.0)
        @test pdf(Chr2, [0.4, 0.7]) ≈ pdf(Chr2scalar, [0.4, 0.7])
        @test all(isapprox.(
            (Copulas.τ(Chr2), Copulas.ρ(Chr2), Copulas.β(Chr2), Copulas.λᵤ(Chr2)),
            (Copulas.τ(Chr2scalar), Copulas.ρ(Chr2scalar), Copulas.β(Chr2scalar), Copulas.λᵤ(Chr2scalar)),
        ))
        @test rand(Random.Xoshiro(4101), Chr2, 16) ==
              rand(Random.Xoshiro(4101), Chr2scalar, 16)

        R = [1.0 0.2 0.1; 0.2 1.0 0.3; 0.1 0.3 1.0]
        Ctev_typed = tEVCopula{3}(4.0, R)
        @test Ctev_typed.tail isa Copulas.tEVTail{<:Any,<:AbstractMatrix}

        R2 = [1.0 0.3; 0.3 1.0]
        Ctev2 = tEVCopula{2}(4.0, R2)
        @test Ctev2.tail isa Copulas.tEVTail{<:Any,<:AbstractMatrix}
        @test Distributions.params(Ctev2).R == R2
        @test cdf(Ctev2, [0.4, 0.7]) ≈
              cdf(tEVCopula{2}(4.0, 0.3), [0.4, 0.7])

        Ctev2scalar = tEVCopula{2}(4.0, 0.3)
        @test pdf(Ctev2, [0.4, 0.7]) ≈ pdf(Ctev2scalar, [0.4, 0.7])
        @test all(isapprox.(
            (Copulas.τ(Ctev2), Copulas.ρ(Ctev2), Copulas.β(Ctev2), Copulas.λᵤ(Ctev2)),
            (Copulas.τ(Ctev2scalar), Copulas.ρ(Ctev2scalar), Copulas.β(Ctev2scalar), Copulas.λᵤ(Ctev2scalar)),
        ))
        @test rand(Random.Xoshiro(4102), Ctev2, 16) ==
              rand(Random.Xoshiro(4102), Ctev2scalar, 16)

        weights = [0.6, 0.7, 0.8]

        asy = [[0.4], [0.3], [0.6, 0.7]]
        dep_tawn = [2.0]
        @test TawnCopula{2}(dep_tawn, asy).tail isa Copulas.TawnTail

        dep_gal = [0.7]
        @test AsymGalambosCopula{2}(dep_gal, asy).tail isa
              Copulas.AsymGalambosTail

        Cag2 = AsymGalambosCopula{2}(0.7, [0.6, 0.7])
        Cagref = AsymGalambosCopula{2}(0.7, 0.6, 0.7)
        @test cdf(Cag2, [0.4, 0.7]) ≈ cdf(Cagref, [0.4, 0.7])

        a = [0.2, 0.5, 0.8]
        λ = ones(7)

        Uemp = [
            0.20 0.40 0.70
            0.30 0.60 0.80
            0.25 0.55 0.75
        ]
        @test_throws ArgumentError HuslerReissCopula{4}(Γ)
        @test_throws ArgumentError HuslerReissCopula(4, Γ)
        @test_throws ArgumentError tEVCopula{4}(4.0, R)
        @test_throws ArgumentError tEVCopula(4, 4.0, R)
        @test_throws ArgumentError TawnCopula{4}(2.0, weights)
        @test_throws ArgumentError AsymGalambosCopula{4}(0.7, weights)
        @test_throws ArgumentError BC2Copula{4}(a)
        @test_throws ArgumentError MOCopula{4}(λ)
        @test_throws DimensionMismatch EmpiricalEVCopula{4}(
            Uemp;
            degree=1,
        )
        @test_throws ArgumentError MOCopula(ones(5))
    end

    @testset "multivariate EV generic conditioning and Rosenblatt" begin
        # The public contract already exercises the common path for logistic,
        # Galambos, Tawn, and asymmetric Galambos. Mixed d=3 remains here as the
        # additional representation-specific dimension path.
        for C in (MixedCopula{3}(0.5),)
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
            for u in (
                [1e-8, 0.999999],
                [1e-3, 0.99],
                [0.01, 0.9],
                [0.99, 0.5],
                [0.99, 0.99],
            )
                expected = logpdf(G, u)
                generic = invoke(
                    Distributions._logpdf,
                    Tuple{Copulas.ExtremeValueCopula{2},typeof(u)},
                    C,
                    u,
                )
                @test logpdf(C, u) ≈ expected atol=2e-12 rtol=2e-12
                @test logpdf(C, u) ≈ generic atol=2e-12 rtol=2e-12
            end
        end
    end

    @testset "Galambos inverse dependence-measure boundaries" begin
        @test Copulas.β⁻¹(GalambosCopula, -0.1) == 0.0
        @test Copulas.β⁻¹(GalambosCopula, 0.0) == 0.0
        @test Copulas.β⁻¹(GalambosCopula, 1.0) == Inf
        @test Copulas.λᵤ⁻¹(GalambosCopula, 0.0) == 0.0
        @test Copulas.λᵤ⁻¹(GalambosCopula, 1.0) == Inf

        for θ in (0.1, 0.3, 1.0, 3.0)
            C = GalambosCopula(2, θ)
            @test Copulas.β⁻¹(GalambosCopula, Copulas.β(C)) ≈ θ
        end
    end

end

function _test_ev_sample(
    C,
    seed,
    n;
    marginal_atol,
    point=nothing,
    cdf_atol=0.04,
)
    d = length(C)
    U = rand(StableRNG(seed), C, n)

    @test size(U) == (d, n)
    @test all(isfinite, U)
    @test all(u -> 0 < u < 1, U)
    @test all(abs(mean(@view U[i, :]) - 0.5) < marginal_atol for i in 1:d)

    if !isnothing(point)
        reference = cdf(C, point)
        empirical = mean(vec(all(U .<= point, dims=1)))
        se = sqrt(max(reference * (1 - reference), 1e-12) / n)
        @test abs(empirical - reference) < max(cdf_atol, 6 * se)
    end

    return U
end

@testset "Extreme-value numerical regressions" begin
    @testset "ExtremeDist support and typed safeguards" begin
        E = Copulas.ExtremeDist(Copulas.LogTail(2.0))
        @test cdf(E, -0.1) == 0.0
        @test cdf(E, 1.1) == 1.0
        @test pdf(E, -0.1) == 0.0
        @test pdf(E, 1.1) == 0.0
        @test isfinite(logpdf(E, 0.5))
        @test 0f0 < Copulas._safett(1f0) < 1f0
        x = BigFloat("1e-30")
        @test Copulas._safett(x) == x
        @test Copulas._safett(one(x) - x) == one(x) - x
    end

    @testset "Smooth EV conditional endpoints" begin
        C = Copulas.ExtremeValueCopula(2, Copulas.LogTail(2.0))
        for j in 1:2
            D0 = Copulas.condition(C, j, 0.0)
            D1 = Copulas.condition(C, j, 1.0)
            @test cdf(D0, 0.4) ≈ 1.0 atol=1e-12
            @test cdf(D1, 0.4) ≈ 0.0 atol=1e-12
            @test quantile(D0, 0.5) == 0.0
            @test quantile(D1, 0.5) == 1.0
        end
    end

    @testset "Marshall-Olkin conditional regression" begin
        λ1, λ2, λ12 = 0.4, 0.7, 0.8
        C = Copulas.ExtremeValueCopula(2, Copulas.MOTail(λ1, λ2, λ12))
        a, b = λ2 / (λ2 + λ12), λ1 / (λ1 + λ12)
        u, v = 0.37, 0.61
        @test cdf(Copulas.condition(C, 2, v), u) ≈
              ForwardDiff.derivative(vv -> cdf(C, [u, vv]), v) atol=2e-8 rtol=2e-7
        @test cdf(Copulas.condition(C, 1, u), v) ≈
              ForwardDiff.derivative(uu -> cdf(C, [uu, v]), u) atol=2e-8 rtol=2e-7
        @test cdf(Copulas.condition(C, 2, 0.0), u) ≈ u^a
        @test cdf(Copulas.condition(C, 1, 0.0), v) ≈ v^b
        @test cdf(Copulas.condition(C, 2, 1.0), u) ≈ b*u
        @test cdf(Copulas.condition(C, 1, 1.0), v) ≈ a*v
        @test quantile(Copulas.condition(C, 2, 1.0), 0.8) == 1.0
        @test quantile(Copulas.condition(C, 1, 1.0), 0.8) == 1.0
    end

    @testset "Strong logistic EV stability" begin
        for θ in (13.5, 210.0)
            L = Copulas.LogTail(θ)
            C = Copulas.ExtremeValueCopula(2, L)
            G = Copulas.GumbelCopula(2, θ)
            E = Copulas.ExtremeDist(L)

            @test Copulas._ghoudi_mixture_probability(L, 0.01) ≈ (θ - 1) / θ
            for z in (1e-4, 0.01, 0.5, 0.99, 1 - 1e-4)
                @test pdf(E, z) >= 0
                @test isfinite(logpdf(E, z))
                @test cdf(E, quantile(E, z)) ≈ z atol=2e-12 rtol=2e-12
            end

            for uv in ([1e-3, 0.99], [0.01, 0.9], [0.99, 0.5], [0.99, 0.99])
                @test cdf(C, uv) ≈ cdf(G, uv) atol=2e-13 rtol=2e-13
                @test logpdf(C, uv) ≈ logpdf(G, uv) atol=2e-12 rtol=2e-12
            end
        end

        C = Copulas.ExtremeValueCopula(2, Copulas.LogTail(13.5))
        for j in 1:2, ucond in (1e-3, 0.9, 0.99), z in (1e-3, 0.01, 0.9)
            D = Copulas.condition(C, j, ucond)
            uv = j == 2 ? [z, ucond] : [ucond, z]
            @test logpdf(D, z) ≈ logpdf(C, uv) atol=2e-12 rtol=2e-12
        end
    end

    @testset "Multivariate logistic EV" begin
        for d in (3, 5), θ in (1.5, 3.5, 13.5)
            C = Copulas.ExtremeValueCopula(d, Copulas.LogTail(θ))
            G = Copulas.GumbelCopula(d, θ)
            u = collect(range(0.21, 0.87; length=d))
            @test cdf(C, u) ≈ cdf(G, u) atol=2e-13 rtol=2e-13
            @test logpdf(C, u) ≈ logpdf(G, u) atol=2e-10 rtol=2e-10

            rng1, rng2 = StableRNG(1700 + d), StableRNG(1700 + d)
            @test rand(rng1, C, 16) == rand(rng2, G, 16)
        end

        θ = 2.3
        L = Copulas.LogTail(θ)
        x = (0.4, 0.7, 1.1, 1.6)
        S = sum(xi^θ for xi in x)
        for I in ((1,), (1, 3), (1, 2, 4), (1, 2, 3, 4))
            k = length(I)
            coeff = k == 1 ? one(θ) : prod(1 - j * θ for j in 1:k-1)
            expected = coeff * S^(inv(θ) - k) * prod(x[i]^(θ - 1) for i in I)
            @test Copulas.ellpartial(L, x, I) ≈ expected atol=2e-13 rtol=2e-12
        end

        Gtail = Copulas.GalambosTail(0.7)
        x2 = (0.4, 0.7)
        _, d1, d2, d12 = Copulas._biv_der_ℓ(Gtail, x2)
        @test Copulas.ellpartial(Gtail, x2, (1,)) ≈ d1
        @test Copulas.ellpartial(Gtail, x2, (2,)) ≈ d2
        @test Copulas.ellpartial(Gtail, x2, (1, 2)) ≈ d12

        C4 = Copulas.ExtremeValueCopula(4, L)
        C2 = Copulas.ExtremeValueCopula(2, L)
        u, v = 0.37, 0.81
        @test cdf(C4, [u, v, 1.0, 1.0]) ≈ cdf(C2, [u, v]) atol=2e-14 rtol=2e-14
        @test cdf(C4, [0.0, v, 0.7, 0.9]) == 0.0
        @test isinf(Copulas.ℓ(L, (Inf, 0.7, 0.0, 0.0)))
        @test Copulas.ℓ(L, (0.4, 0.7, 0.0, 0.0)) ≈ Copulas.ℓ(L, (0.4, 0.7))
    end

    @testset "Multivariate Galambos EV" begin
        function galambos_stdf_reference(x, θ)
            out = sum(x)
            d = length(x)
            for k in 2:d, I in Copulas.Combinatorics.combinations(1:d, k)
                any(i -> iszero(x[i]), I) && continue
                term = sum(x[i]^(-θ) for i in I)^(-inv(θ))
                out += (isodd(k) ? 1 : -1) * term
            end
            return out
        end

        for d in (3, 4), θ in (0.4, 1.5, 5.0)
            tail = Copulas.GalambosTail(θ)
            C = Copulas.ExtremeValueCopula(d, tail)
            x = collect(range(0.31, 1.27; length=d))
            u = exp.(-x)
            ref = galambos_stdf_reference(x, θ)

            @test Copulas.ℓ(tail, x) ≈ ref atol=2e-13 rtol=2e-13
            @test cdf(C, u) ≈ exp(-ref) atol=2e-13 rtol=2e-13
            @test isfinite(logpdf(C, u))
        end

        # The multivariate STDF must reproduce the historical bivariate model.
        for θ in (0.3, 1.2, 7.0)
            tail = Copulas.GalambosTail(θ)
            for x in ((0.23, 0.91), (0.7, 0.4), (1.6, 0.2))
                s = sum(x)
                @test Copulas.ℓ(tail, x) ≈ s * Copulas.A(tail, x[1] / s) atol=2e-13 rtol=2e-13
                _, d1, d2, d12 = Copulas._biv_der_ℓ(tail, x)
                @test Copulas.ellpartial(tail, x, (1,)) ≈ d1 atol=2e-12 rtol=2e-11
                @test Copulas.ellpartial(tail, x, (2,)) ≈ d2 atol=2e-12 rtol=2e-11
                @test Copulas.ellpartial(tail, x, (1, 2)) ≈ d12 atol=2e-11 rtol=2e-10
            end
        end

        tail = Copulas.GalambosTail(1.7)
        C3 = Copulas.ExtremeValueCopula(3, tail)
        C4 = Copulas.ExtremeValueCopula(4, tail)
        x3 = (0.35, 0.72, 1.18)
        @test Copulas.ℓ(tail, (x3..., 0.0)) ≈ Copulas.ℓ(tail, x3) atol=2e-14 rtol=2e-14
        @test cdf(C4, [0.37, 0.61, 0.83, 1.0]) ≈ cdf(C3, [0.37, 0.61, 0.83]) atol=2e-14 rtol=2e-14
        @test cdf(C4, [0.0, 0.61, 0.83, 0.92]) == 0.0
        @test isinf(Copulas.ℓ(tail, (Inf, 0.7, 0.2, 0.0)))

        # Strong dependence: the generic AD fallback loses tiny mixed partials
        # through cancellation, so keep high-precision reference regressions.
        strong_cases = (
            (3, 20.0, collect(range(0.31, 1.37; length=3)), -41.822573144200335),
            (4, 50.0, collect(range(0.31, 1.37; length=4)), -156.06017188457903),
            (5, 210.0, collect(range(0.2, 2.0; length=5)), -1491.783077378844),
        )
        for (d, θ, x, reference) in strong_cases
            C = Copulas.ExtremeValueCopula(d, Copulas.GalambosTail(θ))
            lp = logpdf(C, exp.(-x))
            @test isfinite(lp)
            @test lp ≈ reference atol=2e-8 rtol=2e-10
        end

        # This first partial is about 4.6e-659 and therefore underflows as a
        # Float64 value; its sign/log representation must nevertheless survive.
        x = collect(range(0.2, 2.0; length=5))
        sgn, logabs = Copulas._ellpartial_signlog(Copulas.GalambosTail(210.0), x, (1,))
        @test sgn == 1
        @test isfinite(logabs)
        @test logabs ≈ -1515.8850568704655 atol=2e-8 rtol=2e-10
    end

    @testset "Multivariate Galambos EV sampling" begin
        cases = (
            (3, 0.7, 2713),
            (3, 3.0, 2714),
            (3, 20.0, 2715),
            (4, 1.5, 2716),
        )
        n = 5_000

        for (d, θ, seed) in cases
            C = Copulas.ExtremeValueCopula(d, Copulas.GalambosTail(θ))
            u = collect(range(0.34, 0.82; length=d))
            U = _test_ev_sample(
                C, seed, n;
                marginal_atol=0.02,
                point=u,
                cdf_atol=0.025,
            )

            # Every pairwise margin recovers the historical bivariate Galambos.
            B = Copulas.ExtremeValueCopula(2, Copulas.GalambosTail(θ))
            uv = (0.42, 0.76)
            empirical2 = count(j -> U[1, j] <= uv[1] && U[d, j] <= uv[2], 1:n) / n
            reference2 = cdf(B, collect(uv))
            mc_tol2 = max(0.025, 6sqrt(reference2 * (1 - reference2) / n))
            @test abs(empirical2 - reference2) < mc_tol2
        end
    end

    @testset "BC2 and Cuadras-Auge singular conditionals" begin
        Cbc2 = Copulas.ExtremeValueCopula(2, Copulas.BC2Tail(0.65, 0.25))
        for j in 1:2, t in (0.2, 0.8), α in (0.25, 0.6)
            D = Copulas.condition(Cbc2, j, t)
            q = quantile(D, α)
            @test 0.0 <= q <= 1.0
            @test cdf(D, q) >= α - 5e-10
        end
        for j in 1:2
            @test cdf(Copulas.condition(Cbc2, j, 0.0), 0.3) ≈ 1.0 atol=1e-12
            @test cdf(Copulas.condition(Cbc2, j, 1.0), 0.7) ≈ 0.0 atol=1e-12
        end

        θ = 0.6
        Cca = Copulas.ExtremeValueCopula(2, Copulas.CuadrasAugeTail(θ))
        for j in 1:2
            @test cdf(Copulas.condition(Cca, j, 0.0), 0.37) ≈ 0.37^(1-θ)
            @test cdf(Copulas.condition(Cca, j, 1.0), 0.37) ≈ (1-θ)*0.37
            @test quantile(Copulas.condition(Cca, j, 1.0), 0.8) == 1.0
        end
    end
end


@testset "Multivariate Hüsler-Reiss EV" begin
    @testset "Exchangeable scalar parameterization" begin
        cases = (
            (3, 0.7, 3701),
            (3, 3.0, 3702),
            (4, 1.5, 3703),
        )
        n = 5_000

        for (d, θ, seed) in cases
            tail = Copulas.HuslerReissTail(θ)
            C = Copulas.ExtremeValueCopula(d, tail)
            u = collect(range(0.34, 0.78; length=d))
            _test_ev_sample(C, seed, n; marginal_atol=0.025, point=u, cdf_atol=0.03)

            @test isfinite(logpdf(C, collect(range(0.29, 0.83; length=d))))
        end
    end

    @testset "General variogram parameterization" begin
        Σ = [1.20 0.35 0.18;
             0.35 0.90 0.22;
             0.18 0.22 1.05]

        Γ = zeros(4, 4)
        for i in 1:3
            Γ[i, 4] = Γ[4, i] = Σ[i, i]
            for j in 1:3
                Γ[i, j] = Σ[i, i] + Σ[j, j] - 2Σ[i, j]
            end
        end

        tail = Copulas.HuslerReissTail(Γ)
        C = Copulas.ExtremeValueCopula(4, tail)

        @test Copulas._is_valid_in_dim(tail, 4)
        @test !Copulas._is_valid_in_dim(tail, 3)

        u = [0.31, 0.49, 0.67, 0.82]
        @test 0.0 < cdf(C, u) < 1.0
        @test isfinite(logpdf(C, u))

        pidx = [3, 1, 4, 2]
        Cp = Copulas.ExtremeValueCopula(
            4,
            Copulas.HuslerReissTail(Γ[pidx, pidx]),
        )
        @test cdf(Cp, u[pidx]) ≈ cdf(C, u) atol=5e-4 rtol=5e-4
        @test logpdf(Cp, u[pidx]) ≈ logpdf(C, u) atol=5e-3 rtol=5e-3

        n = 6_000
        U = _test_ev_sample(C, 3710, n; marginal_atol=0.025)
        q = (0.42, 0.74)

        for i in 1:3, j in i+1:4
            θij = 2 / sqrt(Γ[i, j])
            Cij = Copulas.ExtremeValueCopula(2, Copulas.HuslerReissTail(θij))
            target = cdf(Cij, collect(q))
            empirical = mean(((@view U[i, :]) .<= q[1]) .& ((@view U[j, :]) .<= q[2]))
            se = sqrt(max(target * (1 - target), 1e-12) / n)
            @test abs(empirical - target) < max(0.03, 6 * se)
        end

        @test_throws DimensionMismatch Copulas.HuslerReissTail(zeros(3, 4))
        @test Copulas.HuslerReissTail([0.0 1.0; 1.0 0.0]) isa
              Copulas.HuslerReissTail{<:AbstractMatrix}

        Γbad = [0.0 1.0 10.0;
                1.0 0.0 1.0;
                10.0 1.0 0.0]
        @test_throws ArgumentError Copulas.HuslerReissTail(Γbad)
    end
end


@testset "Multivariate extremal-t EV" begin
    @testset "rho=0 regression and exchangeable scalar API" begin
        tail0 = Copulas.tEVTail(1.0, 0.0)
        @test tail0 isa Copulas.tEVTail
        @test 2 * Copulas.A(tail0, 0.5) ≈ 1 + inv(sqrt(2)) atol=2e-12 rtol=2e-12

        cases = (
            (3, 1.7, 0.0, 4701),
            (3, 2.3, -0.2, 4702),
            (4, 2.0, 0.4, 4703),
        )

        for (d, ν, ρ, seed) in cases
            tail = Copulas.tEVTail(ν, ρ)
            @test Copulas._is_valid_in_dim(tail, d)

            C = Copulas.ExtremeValueCopula(d, tail)
            u = collect(range(0.31, 0.82; length=d))

            @test 0.0 < cdf(C, u) < 1.0
            @test isfinite(logpdf(C, u))

            _test_ev_sample(C, seed, 4_000; marginal_atol=0.03, point=u)
        end

        @test !Copulas._is_valid_in_dim(Copulas.tEVTail(1.7, -0.7), 3)
        @test !Copulas._is_valid_in_dim(Copulas.tEVTail(1.7, -0.4), 4)
    end

    @testset "general correlation parameterization" begin
        R = [1.0  0.35 -0.20;
             0.35 1.0   0.15;
            -0.20 0.15  1.0]
        ν = 1.7

        tail = Copulas.tEVTail(ν, R)
        C = Copulas.ExtremeValueCopula(3, tail)

        @test Copulas._is_valid_in_dim(tail, 3)
        @test !Copulas._is_valid_in_dim(tail, 4)
        @test Distributions.params(tail).ν == ν
        @test Distributions.params(tail).R ≈ R

        u = [0.34, 0.58, 0.79]
        @test 0.0 < cdf(C, u) < 1.0
        @test isfinite(logpdf(C, u))

        U = _test_ev_sample(C, 4710, 6_000; marginal_atol=0.03)

        q = [0.41, 0.75]
        for i in 1:2, j in (i + 1):3
            Cij = Copulas.ExtremeValueCopula(
                2,
                Copulas.tEVTail(ν, R[i, j]),
            )
            target = cdf(Cij, q)
            empirical = mean(
                ((@view U[i, :]) .<= q[1]) .&
                ((@view U[j, :]) .<= q[2])
            )
            se = sqrt(max(target * (1 - target), 1e-12) / size(U, 2))
            @test abs(empirical - target) < max(0.04, 6 * se)
        end
    end

    @testset "general R agrees with exchangeable scalar model" begin
        for (d, ν, ρ) in (
            (3, 1.3, 0.25),
            (4, 2.2, 0.4),
        )
            R = fill(ρ, d, d)
            for i in 1:d
                R[i, i] = 1.0
            end

            Cscalar = Copulas.ExtremeValueCopula(
                d,
                Copulas.tEVTail(ν, ρ),
            )
            Cmatrix = Copulas.ExtremeValueCopula(
                d,
                Copulas.tEVTail(ν, R),
            )

            u = collect(range(0.29, 0.83; length=d))
            @test cdf(Cscalar, u) ≈ cdf(Cmatrix, u) atol=3e-7 rtol=3e-7
            @test logpdf(Cscalar, u) ≈ logpdf(Cmatrix, u) atol=3e-6 rtol=3e-6
        end
    end

    @testset "invalid correlation matrices" begin
        @test_throws DimensionMismatch Copulas.tEVTail(
            1.5,
            zeros(3, 4),
        )
        @test_throws ArgumentError Copulas.tEVTail(
            0.0,
            Matrix{Float64}(I, 3, 3),
        )
        @test_throws ArgumentError Copulas.tEVTail(
            1.5,
            [1.0 0.3 0.0;
             0.1 1.0 0.2;
             0.0 0.2 1.0],
        )
        @test_throws ArgumentError Copulas.tEVTail(
            1.5,
            [1.0 0.95 0.95;
             0.95 1.0 -0.95;
             0.95 -0.95 1.0],
        )
    end
end


@testset "Multivariate Tawn EV" begin
    @testset "historical asymmetric-logistic reduction" begin
        α = 2.1
        θ1 = 0.67
        θ2 = 0.38

        Cold = Copulas.ExtremeValueCopula(
            2,
            Copulas.AsymLogTail(α, θ1, θ2),
        )
        Ctawn = Copulas.ExtremeValueCopula(
            2,
            Copulas.TawnTail(α, [θ2, θ1]),
        )

        for u in (
            [0.34, 0.76],
            [0.71, 0.49],
            [0.57, 0.62],
        )
            @test cdf(Ctawn, u) ≈ cdf(Cold, u) atol=3e-13 rtol=3e-13
            @test logpdf(Ctawn, u) ≈ logpdf(Cold, u) atol=3e-11 rtol=3e-11
        end

        _test_ev_sample(Ctawn, 4801, 4_000; marginal_atol=0.03)
    end

    @testset "symmetric logistic reduction" begin
        @test Copulas.TawnTail(2, [2.0], [[0.0], [0.0], [1.0, 1.0]]) isa
              Copulas.LogTail
        @test Copulas.TawnTail(2, [2.0], [[1.0], [1.0], [0.0, 0.0]]) isa
              Copulas.NoTail

        for d in (3, 4), α in (1.2, 2.5)
            Ctawn = Copulas.ExtremeValueCopula(
                d,
                Copulas.TawnTail(α, ones(d)),
            )
            Clog = Copulas.ExtremeValueCopula(d, Copulas.LogTail(α))
            u = collect(range(0.29, 0.82; length=d))

            @test cdf(Ctawn, u) ≈ cdf(Clog, u) atol=5e-13 rtol=5e-13
            @test logpdf(Ctawn, u) ≈ logpdf(Clog, u) atol=3e-10 rtol=3e-10
        end
    end

    @testset "full trivariate Tawn regression" begin
        dep = [1.4, 2.0, 1.7, 2.3]
        asy = [
            [0.15],
            [0.20],
            [0.10],
            [0.25, 0.15],
            [0.20, 0.20],
            [0.25, 0.30],
            [0.40, 0.40, 0.40],
        ]

        tail = Copulas.TawnTail(3, dep, asy)
        C = Copulas.ExtremeValueCopula(3, tail)
        x = (0.37, 0.79, 1.28)

        @test Copulas.ℓ(tail, x) ≈ 1.824598177317017 atol=2e-14 rtol=2e-14

        refs = (
            ((1,), 0.4661019902512721),
            ((2,), 0.6461594736259313),
            ((3,), 0.8919331693434068),
            ((1, 2), -0.08353819466501686),
            ((1, 3), -0.08853049949907850),
            ((2, 3), -0.18780484302735106),
            ((1, 2, 3), 0.05249667842788736),
        )

        for (I, ref) in refs
            @test Copulas.ellpartial(tail, x, I) ≈ ref atol=3e-13 rtol=3e-12
            sign, logabs = Copulas._ellpartial_signlog(tail, x, I)
            @test sign == (isodd(length(I)) ? 1 : -1)
            @test exp(logabs) ≈ abs(ref) atol=3e-13 rtol=3e-12
        end

        u = [0.34, 0.57, 0.81]
        @test logpdf(C, u) ≈ -0.2449881198991001 atol=3e-12 rtol=3e-12

        _test_ev_sample(C, 4802, 6_000; marginal_atol=0.03, point=u)
    end

    @testset "constructor validation" begin
        dep = [1.4, 2.0, 1.7, 2.3]
        good = [
            [0.15],
            [0.20],
            [0.10],
            [0.25, 0.15],
            [0.20, 0.20],
            [0.25, 0.30],
            [0.40, 0.40, 0.40],
        ]

        @test_throws DimensionMismatch Copulas.TawnTail(3, dep[1:3], good)

        badsum = deepcopy(good)
        badsum[end][1] = 0.30
        @test_throws ArgumentError Copulas.TawnTail(3, dep, badsum)

        baddep = copy(dep)
        baddep[2] = 0.8
        @test_throws ArgumentError Copulas.TawnTail(3, baddep, good)
    end
end


@testset "Multivariate asymmetric Galambos EV" begin
    @testset "historical bivariate reduction" begin
        α = 1.4
        θ1 = 0.67
        θ2 = 0.38

        Cold = Copulas.ExtremeValueCopula(
            2,
            Copulas.AsymGalambosTail(α, θ1, θ2),
        )
        Cnew = Copulas.ExtremeValueCopula(
            2,
            Copulas.AsymGalambosTail(
                2,
                [α],
                [[1 - θ1], [1 - θ2], [θ1, θ2]],
            ),
        )

        for u in (
            [0.34, 0.76],
            [0.71, 0.49],
            [0.57, 0.62],
        )
            @test cdf(Cnew, u) ≈ cdf(Cold, u) atol=3e-12 rtol=3e-12
            @test logpdf(Cnew, u) ≈ logpdf(Cold, u) atol=3e-9 rtol=3e-9
        end

        _test_ev_sample(Cnew, 4901, 4_000; marginal_atol=0.03)
    end

    @testset "symmetric Galambos reduction" begin
        @test Copulas.AsymGalambosTail(2, [0.7], [[0.0], [0.0], [1.0, 1.0]]) isa
              Copulas.GalambosTail
        @test Copulas.AsymGalambosTail(2, [0.7], [[1.0], [1.0], [0.0, 0.0]]) isa
              Copulas.NoTail

        for d in (3, 4), α in (0.7, 1.7)
            Casym = Copulas.ExtremeValueCopula(
                d,
                Copulas.AsymGalambosTail(α, ones(d)),
            )
            Csym = Copulas.ExtremeValueCopula(
                d,
                Copulas.GalambosTail(α),
            )

            u = collect(range(0.29, 0.82; length=d))
            @test cdf(Casym, u) ≈ cdf(Csym, u) atol=3e-12 rtol=3e-12
            @test logpdf(Casym, u) ≈ logpdf(Csym, u) atol=2e-8 rtol=2e-8
        end
    end

    @testset "full trivariate asymmetric Galambos regression" begin
        dep = [0.7, 1.3, 0.9, 1.8]
        asy = [
            [0.15],
            [0.20],
            [0.10],
            [0.25, 0.15],
            [0.20, 0.20],
            [0.25, 0.30],
            [0.40, 0.40, 0.40],
        ]

        tail = Copulas.AsymGalambosTail(3, dep, asy)
        C = Copulas.ExtremeValueCopula(3, tail)
        x = (0.37, 0.79, 1.28)

        reconstructed = Copulas.AsymGalambosTail(values(params(tail))...)
        @test reconstructed.α == tail.α
        @test reconstructed.β == tail.β

        @test Copulas.ℓ(tail, x) ≈ 1.8097921615972135 atol=3e-13 rtol=3e-12

        refs = (
            ((1,), 0.42313975454450815),
            ((2,), 0.6425106046268907),
            ((3,), 0.8950367771566420),
            ((1, 2), -0.09404641593762274),
            ((1, 3), -0.07302518430676948),
            ((2, 3), -0.19707592644863348),
            ((1, 2, 3), 0.04639197718394051),
        )

        for (I, ref) in refs
            @test Copulas.ellpartial(tail, x, I) ≈ ref atol=3e-11 rtol=3e-10

            sign, logabs = Copulas._ellpartial_signlog(tail, x, I)
            @test sign == (isodd(length(I)) ? 1 : -1)
            @test exp(logabs) ≈ abs(ref) atol=3e-11 rtol=3e-10
        end

        u = [0.34, 0.57, 0.81]
        @test logpdf(C, u) ≈ -0.3221640487545458 atol=3e-10 rtol=3e-10

        _test_ev_sample(C, 4902, 6_000; marginal_atol=0.03, point=u)
    end

    @testset "constructor validation" begin
        dep = [0.7, 1.3, 0.9, 1.8]
        good = [
            [0.15],
            [0.20],
            [0.10],
            [0.25, 0.15],
            [0.20, 0.20],
            [0.25, 0.30],
            [0.40, 0.40, 0.40],
        ]

        @test_throws DimensionMismatch Copulas.AsymGalambosTail(
            3,
            dep[1:3],
            good,
        )

        badsum = deepcopy(good)
        badsum[end][1] = 0.30
        @test_throws ArgumentError Copulas.AsymGalambosTail(
            3,
            dep,
            badsum,
        )

        baddep = copy(dep)
        baddep[2] = -0.1
        @test_throws ArgumentError Copulas.AsymGalambosTail(
            3,
            baddep,
            good,
        )
    end
end


@testset "Multivariate Mixed EV" begin
    @testset "historical bivariate reduction and tail dependence" begin
        for θ in (0.15, 0.55, 1.0)
            tail = Copulas.MixedTail(θ)
            C = Copulas.ExtremeValueCopula(2, tail)
            x = [0.37, 1.29]

            ref = sum(x) * Copulas.A(tail, x[1] / sum(x))
            @test Copulas.ℓ(tail, x) ≈ ref atol=4e-14 rtol=4e-14
            @test Copulas.λᵤ(C) ≈ θ / 2 atol=2e-15 rtol=2e-15
        end
    end

    @testset "multivariate Galambos-mixture identity" begin
        θ = 0.63
        tail = Copulas.MixedTail(θ)

        for d in (3, 4)
            x = collect(range(0.31, 1.28; length=d))
            ref = (1 - θ) * sum(x) +
                  θ * Copulas.ℓ(Copulas.GalambosTail(1.0), x)

            @test Copulas._is_valid_in_dim(tail, d)
            @test Copulas.ℓ(tail, x) ≈ ref atol=3e-13 rtol=3e-12

            for I in (
                (1,),
                (1, 2),
                ntuple(identity, d),
            )
                got = Copulas.ellpartial(tail, x, I)

                if length(I) == 1
                    refp = (1 - θ) +
                           θ * Copulas.ellpartial(
                               Copulas.GalambosTail(1.0),
                               x,
                               I,
                           )
                else
                    refp = θ * Copulas.ellpartial(
                        Copulas.GalambosTail(1.0),
                        x,
                        I,
                    )
                end

                @test got ≈ refp atol=3e-11 rtol=3e-10
            end
        end
    end

    @testset "logpdf and exact sampling" begin
        θ = 0.63
        C = Copulas.ExtremeValueCopula(3, Copulas.MixedTail(θ))
        u = [0.34, 0.57, 0.81]

        @test logpdf(C, u) ≈ -0.118043090304781 atol=3e-12 rtol=3e-12

        _test_ev_sample(C, 5001, 6_000; marginal_atol=0.03, point=u)
    end
end

@testset "AsymMixed feasible-set parameterization" begin
    @testset "positive and negative asymmetry are reachable" begin
        positive = Copulas._rebound_params(
            Copulas.AsymMixedTail,
            2,
            [-3.0, 3.0],
        )
        negative = Copulas._rebound_params(
            Copulas.AsymMixedTail,
            2,
            [3.0, -3.0],
        )

        @test positive.θ₂ > 0
        @test negative.θ₂ < 0

        for p in (positive, negative)
            @test p.θ₁ >= 0
            @test p.θ₁ + p.θ₂ <= 1
            @test p.θ₁ + 2p.θ₂ <= 1
            @test p.θ₁ + 3p.θ₂ >= 0
            @test Copulas.AsymMixedTail(p.θ₁, p.θ₂) isa Copulas.AsymMixedTail
        end
    end

    @testset "unconstrained round trip" begin
        for z in (
            [-3.0, 2.0],
            [-1.0, -0.7],
            [0.0, 0.5],
            [1.0, -2.0],
            [3.0, -3.0],
        )
            p = Copulas._rebound_params(Copulas.AsymMixedTail, 2, z)
            zback = Copulas._unbound_params(Copulas.AsymMixedTail, 2, p)

            @test zback ≈ z atol=3e-11 rtol=3e-11
        end
    end

    @testset "parameter round trip across feasible interior" begin
        for (θ1, θ2) in (
            (0.25, 0.10),
            (0.65, 0.08),
            (0.85, -0.08),
            (1.20, -0.30),
        )
            p = (; θ₁=θ1, θ₂=θ2)
            z = Copulas._unbound_params(Copulas.AsymMixedTail, 2, p)
            back = Copulas._rebound_params(Copulas.AsymMixedTail, 2, z)

            @test back.θ₁ ≈ θ1 atol=3e-11 rtol=3e-11
            @test back.θ₂ ≈ θ2 atol=3e-11 rtol=3e-11
        end
    end
end


@testset "AsymMixed fitting example remains asymmetric" begin
    CT = Copulas.AsymMixedCopula
    Cex = Copulas._example(CT, 2)
    p = Distributions.params(Cex)

    @test Cex isa CT
    @test keys(p) == (:θ₁, :θ₂)
    @test p.θ₂ != 0

    z = Copulas._unbound_params(CT, 2, p)
    back = Copulas._rebound_params(CT, 2, z)

    @test back.θ₁ ≈ p.θ₁ atol=3e-12 rtol=3e-12
    @test back.θ₂ ≈ p.θ₂ atol=3e-12 rtol=3e-12
end

@testset "Discrete spectral multivariate EV" begin
    B = [
        0.40 0.20 0.10 0.30
        0.10 0.50 0.20 0.20
        0.30 0.10 0.40 0.20
    ]

    tail = Copulas.DiscreteSpectralTail(B)
    C = ExtremeValueCopula{3}(tail)
    x = [0.37, 0.79, 1.28]

    ref = sum(maximum(B[i, k] * x[i] for i in axes(B, 1))
              for k in axes(B, 2))

    @test Copulas._is_valid_in_dim(tail, 3)
    @test !Copulas._is_valid_in_dim(tail, 2)
    @test Copulas.ℓ(tail, x) ≈ ref atol=3e-14 rtol=3e-14
    @test maximum(x) <= ref <= sum(x)

    for i in 1:3
        e = zeros(3)
        e[i] = 1
        @test Copulas.ℓ(tail, e) ≈ 1 atol=3e-14 rtol=3e-14
    end

    u = [0.34, 0.57, 0.81]
    @test cdf(C, u) ≈ exp(-sum(
        maximum(B[i, k] * (-log(u[i])) for i in axes(B, 1))
        for k in axes(B, 2)
    )) atol=3e-14 rtol=3e-14

    _test_ev_sample(C, 5101, 5_000; marginal_atol=0.035)

    @test_throws ArgumentError Copulas.DiscreteSpectralTail([
        0.4 0.4
        0.5 0.5
    ])
    @test_throws ArgumentError Copulas.DiscreteSpectralTail([
        1.2 -0.2
        0.5  0.5
    ])
    @test_throws ArgumentError logpdf(C, u)
end

@testset "Multivariate Marshall-Olkin EV" begin
    d = 3
    λ = [0.35, 0.55, 0.40, 0.25, 0.30, 0.45, 0.70]
    tail = Copulas.MOTail(d, λ)
    C = Copulas.ExtremeValueCopula(d, tail)

    B = tail.spectral.B
    @test all(abs.(vec(sum(B, dims=2)) .- 1) .< 3e-14)

    x = [0.31, 0.82, 1.41]
    ref = sum(maximum(B[i, k] * x[i] for i in axes(B, 1))
              for k in axes(B, 2))
    @test Copulas.ℓ(tail, x) ≈ ref atol=3e-14 rtol=3e-14

    oldtail = Copulas.MOTail(0.30, 0.50, 0.70)
    newtail = Copulas.MOTail(2, [0.50, 0.30, 0.70])
    @test typeof(oldtail) == typeof(newtail)
    @test oldtail isa Copulas.DiscreteSpectralPickandsTail
    @test Distributions.params(oldtail) == (λ₁=0.30, λ₂=0.50, λ₃=0.70)
    Cold = Copulas.ExtremeValueCopula(2, oldtail)
    Cnew = Copulas.ExtremeValueCopula(2, newtail)

    for xx in ([0.37, 1.29], [1.11, 0.46])
        oldref = sum(xx) * Copulas.A(oldtail, xx[1] / sum(xx))
        @test Copulas.ℓ(newtail, xx) ≈ oldref atol=4e-14 rtol=4e-14
    end

    for u in ([0.34, 0.76], [0.71, 0.49], [0.57, 0.62])
        @test cdf(Cnew, u) ≈ cdf(Cold, u) atol=4e-14 rtol=4e-14
    end

    _test_ev_sample(C, 5102, 5_000; marginal_atol=0.035)

    @test_throws DimensionMismatch Copulas.MOTail(3, λ[1:6])
    @test_throws ArgumentError Copulas.MOTail(
        3,
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0],
    )
end

@testset "Multivariate BC2 EV" begin
    a = [0.20, 0.65, 0.40, 0.75]
    tail = Copulas.BC2Tail(a)
    C = Copulas.ExtremeValueCopula(4, tail)

    x = [0.27, 0.64, 1.03, 1.31]
    ref = maximum(a .* x) + maximum((1 .- a) .* x)

    @test Copulas.ℓ(tail, x) ≈ ref atol=3e-14 rtol=3e-14
    @test tail.spectral.B ≈ hcat(a, 1 .- a) atol=3e-15 rtol=3e-15

    oldtail = Copulas.BC2Tail(0.30, 0.70)
    newtail = Copulas.BC2Tail([0.30, 0.70])
    @test typeof(oldtail) == typeof(newtail)
    @test oldtail isa Copulas.DiscreteSpectralPickandsTail
    @test Distributions.params(oldtail) == (a=0.30, b=0.70)
    Cold = Copulas.ExtremeValueCopula(2, oldtail)
    Cnew = Copulas.ExtremeValueCopula(2, newtail)

    for xx in ([0.37, 1.29], [1.11, 0.46])
        oldref = sum(xx) * Copulas.A(oldtail, xx[1] / sum(xx))
        @test Copulas.ℓ(newtail, xx) ≈ oldref atol=3e-14 rtol=3e-14
    end

    for u in ([0.34, 0.76], [0.71, 0.49], [0.57, 0.62])
        @test cdf(Cnew, u) ≈ cdf(Cold, u) atol=3e-14 rtol=3e-14
    end

    _test_ev_sample(C, 5103, 5_000; marginal_atol=0.035)

    @test_throws ArgumentError Copulas.BC2Tail([0.2])
    @test_throws ArgumentError Copulas.BC2Tail([0.2, 1.1])
end

@testset "Multivariate Cuadras-Auge EV" begin
    θ = 0.62
    tail = Copulas.CuadrasAugeTail(θ)

    for (d, seed) in ((3, 5104), (4, 5105))
        C = Copulas.ExtremeValueCopula(d, tail)
        x = collect(range(0.29, 1.34; length=d))

        @test Copulas._is_valid_in_dim(tail, d)
        @test Copulas.ℓ(tail, x) ≈
              (1 - θ) * sum(x) + θ * maximum(x) atol=3e-14 rtol=3e-14

        u = collect(range(0.34, 0.82; length=d))
        @test cdf(C, u) ≈
              minimum(u)^θ * prod(u)^(1 - θ) atol=3e-14 rtol=3e-14

        _test_ev_sample(C, seed, 5_000; marginal_atol=0.035)
    end

    for xx in ([0.37, 1.29], [1.11, 0.46])
        oldref = sum(xx) * Copulas.A(tail, xx[1] / sum(xx))
        @test Copulas.ℓ(tail, xx) ≈ oldref atol=3e-14 rtol=3e-14
    end
end

@testset "Multivariate empirical EV" begin
    Ctrue = Copulas.ExtremeValueCopula(3, Copulas.LogTail(2.2))
    U = rand(StableRNG(5201), Ctrue, 2_500)

    @testset "shape-valid spectral projection" begin
        for method in (:pickands, :cfg, :ols)
            tail = Copulas.EmpiricalEVMultivariateTail(
                U;
                method=method,
                degree=4,
                pseudo_values=true,
            )

            @test tail.d == 3
            @test tail.method == method
            @test tail.degree == 4
            @test isfinite(tail.projection_rmse)
            @test tail.projection_rmse >= 0
            @test Copulas._is_valid_in_dim(tail, 3)
            @test !Copulas._is_valid_in_dim(tail, 2)

            B = tail.spectral.B
            @test size(B, 1) == 3
            @test all(B .>= 0)
            @test all(abs.(vec(sum(B, dims=2)) .- 1) .< 3e-12)

            for x in (
                [0.31, 0.73, 1.19],
                [1.11, 0.44, 0.69],
            )
                ell = Copulas.ℓ(tail, x)
                @test maximum(x) - 3e-12 <= ell <= sum(x) + 3e-12
                @test Copulas.ℓ(tail, 2.7 .* x) ≈
                      2.7ell atol=3e-11 rtol=3e-11
            end
        end
    end

    @testset "OLS default and oracle accuracy" begin
        tail = Copulas.EmpiricalEVMultivariateTail(
            U;
            degree=5,
            pseudo_values=true,
        )
        @test tail.method == :ols

        maxerr = 0.0
        for w in (
            [1/3, 1/3, 1/3],
            [0.60, 0.25, 0.15],
            [0.10, 0.55, 0.35],
            [0.25, 0.15, 0.60],
        )
            truth = Copulas.ℓ(Ctrue.tail, w)
            estimate = Copulas.ℓ(tail, w)
            maxerr = max(maxerr, abs(estimate - truth))
        end
        @test maxerr < 0.10
    end

    @testset "constructor and exact sampling from projected model" begin
        Cemp = Copulas.EmpiricalEVCopula(
            U;
            method=:ols,
            degree=4,
            pseudo_values=true,
        )

        @test Cemp.tail isa Copulas.EmpiricalEVMultivariateTail

        u0 = [0.36, 0.58, 0.79]
        _test_ev_sample(Cemp, 5202, 6_000; marginal_atol=0.035, point=u0)

        @test_throws ArgumentError logpdf(Cemp, u0)
    end

    @testset "generic fitting route" begin
        fitted = fit(
            Copulas.ExtremeValueCopula,
            U,
            :ols;
            degree=4,
            pseudo_values=true,
        )

        @test fitted.tail isa Copulas.EmpiricalEVMultivariateTail
        @test fitted.tail.method == :ols
        @test fitted.tail.degree == 4
        @test Copulas._is_valid_in_dim(fitted.tail, 3)
    end

    @testset "historical bivariate empirical EV remains unchanged" begin
        C2 = Copulas.ExtremeValueCopula(2, Copulas.LogTail(2.0))
        U2 = rand(StableRNG(5203), C2, 1_000)

        Cold = Copulas.EmpiricalEVCopula(
            U2;
            method=:ols,
            grid=101,
            pseudo_values=true,
        )

        @test Cold.tail isa Copulas.EmpiricalEVTail
        @test isfinite(cdf(Cold, [0.43, 0.71]))
    end
end
