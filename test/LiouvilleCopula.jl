@testset "Liouville copulas" begin
    liouville_rng = StableRNG(405)
    @testset "real Williamson orders" begin
        G = Copulas.𝒲(Dirac(1.0), 5.5)
        C = LiouvilleCopula{3}(G, (0.75, 1.5, 3.0))
        @test C isa Copulas.Copula{3}
        @test Copulas.𝒲₋₁(G, sum(C.α)) isa Copulas.WilliamsonBetaProduct
        @test_throws ArgumentError LiouvilleCopula{2}(G, (3.0, 3.0))

        U = rand(liouville_rng, C, 5)
        @test size(U) == (3, 5)
        @test all(0 .<= U .<= 1)

        C13 = subsetdims(C, (1, 3))
        @test C13 isa LiouvilleCopula{2}
        @test C13.G === C.G
        @test C13.α == (C.α[1], C.α[3])

        source = Copulas.WilliamsonFromFrailty(LogNormal(), 2.0)
        reduced = Copulas.WilliamsonBetaProduct(source, Beta(0.75, 1.25))
        @test reduced isa Copulas.WilliamsonFromFrailty
        @test reduced.order == 0.75

        dirac_radial = Copulas.WilliamsonFromFrailty(Dirac(2.0), 0.75)
        @test cdf(dirac_radial, 0.4) ≈ cdf(Gamma(0.75, 0.5), 0.4)

        clayton_radial = Copulas.𝒲₋₁(Copulas.ClaytonGenerator(1.0), 0.75)
        @test clayton_radial isa Distributions.LocationScale
    end

    @testset "Archimedean identity" begin
        G = Copulas.ClaytonGenerator(1.0)
        L = LiouvilleCopula{2}(G, (1.0, 1.0))
        A = ArchimedeanCopula{2}(G)
        u = [0.35, 0.7]
        @test L isa ArchimedeanCopula{2}
        @test cdf(L, u) ≈ cdf(A, u) atol=1e-7
        @test logpdf(L, u) ≈ logpdf(A, u)
    end

    @testset "finite-support beta-product quantiles" begin
        C = LiouvilleCopula{2}(
            Copulas.ClaytonGenerator(-0.25), (0.75, 1.25),
        )
        for α in C.α
            margin = Copulas.𝒲₋₁(C.G, α)
            @test cdf(margin, maximum(margin)) == 1
            q = quantile(margin, 1 - 1e-5)
            @test q < maximum(margin)
            @test cdf(margin, q) ≈ 1 - 1e-5 atol=1e-8
        end
        @test pdf(C, fill(1e-5, 2)) >= 0
    end

    @testset "conditioning and Rosenblatt" begin
        integer_C = LiouvilleCopula{3}(
            Copulas.ClaytonGenerator(1.0), (1.0, 1.0, 2.0),
        )
        integer_conditional = Copulas.ConditionalCopula(integer_C, (1,), (0.4,))
        @test integer_conditional.G isa Copulas.𝒲
        @test integer_conditional.G.X isa Distributions.LocationScale
        @test integer_conditional.G.X.ρ isa BetaPrime
        @test condition(integer_C, (1,), (0.4,)) isa SklarDist
        integer_distortion = condition(integer_C, (1, 2), (0.4, 0.6))
        integer_p = cdf(integer_distortion, 0.7)
        @test quantile(integer_distortion, integer_p) ≈ 0.7

        fallback_C = LiouvilleCopula{3}(
            Copulas.ClaytonGenerator(-0.25), (1.0, 1.0, 1.5),
        )
        fallback_conditional = Copulas.ConditionalCopula(fallback_C, (1,), (0.4,))
        @test fallback_conditional.G isa Copulas.TiltedGenerator

        fractional_C = LiouvilleCopula{3}(
            Copulas.𝒲(Dirac(1.0), 4.0), (0.6, 1.1, 1.3),
        )
        fractional_conditional = Copulas.ConditionalCopula(fractional_C, (1,), (0.4,))
        @test fractional_conditional.G isa Copulas.𝒲

        discrete_C = LiouvilleCopula{3}(
            Copulas.AMHGenerator(0.5), (0.6, 1.1, 1.3),
        )
        discrete_conditional = Copulas.ConditionalCopula(discrete_C, (1,), (0.4,))
        @test discrete_conditional.G.X.frailty_dist isa Copulas.PowerTiltedFrailty
        @test Distributions.value_support(
            typeof(discrete_conditional.G.X.frailty_dist),
        ) == Distributions.Discrete

        gamma_posterior = Copulas.PowerTiltedFrailty(Gamma(2.0, 3.0), 0.75, 0.4)
        @test gamma_posterior isa Gamma
        @test all(isapprox.(
            params(gamma_posterior), (2.75, inv(inv(3.0) + 0.4)),
        ))
        @test Copulas.WilliamsonFromFrailty(gamma_posterior, 1.2) isa
              Distributions.LocationScale

        D = Copulas.DistortionFromCop(fractional_C, (1,), (0.4,), 2)
        p = Distributions.cdf(D, 0.6)
        @test Distributions.quantile(D, p) ≈ 0.6

        u = [0.3, 0.5, 0.8]
        v = rosenblatt(fractional_C, u)
        @test inverse_rosenblatt(fractional_C, v) ≈ u
    end

    @testset "conditional radial quadrature cache" begin
        radials = (
            Copulas.LiouvilleConditionalRadial(Beta(2.0, 3.0), 0.1, 3.0, 0.7),
            Copulas.LiouvilleConditionalRadial(Gamma(3.0, 1.0), 0.4, 3.0, 1.2),
        )

        for D in radials
            @test isfinite(D.normalizer) && D.normalizer > 0
            @test issorted(D.integration_knots)
            @test issorted(D.cumulative_masses)
            @test first(D.cumulative_masses) == 0
            @test last(D.cumulative_masses) == D.normalizer
            @test cdf(D, minimum(D)) == 0
            @test cdf(D, maximum(D)) == 1
            @test quantile(D, 0) == minimum(D)
            @test quantile(D, 1) == maximum(D)

            ps = (0.01, 0.1, 0.5, 0.9, 0.99)
            qs = quantile.(Ref(D), ps)
            @test issorted(qs)
            @test all(isapprox.(cdf.(Ref(D), qs), ps; atol=1e-7))
            @test all(isapprox.(quantile.(Ref(D), ps), qs))

            for t in D.integration_knots[2:(end - 1)]
                s = Copulas._liouville_conditional_radial_coordinate(D, t)
                left = cdf(D, prevfloat(s))
                right = cdf(D, nextfloat(s))
                @test left <= right + 1e-7
                @test abs(right - left) < 1e-6
            end
        end
    end
end
