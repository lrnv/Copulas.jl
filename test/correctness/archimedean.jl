# Correctness obligation: independent Archimedean reference values, support
# boundaries, and conditional numerical anchors.

@testset "Boundary test for bivariate Joe, Gumbel and Frank" begin
    θ = 1.1
    C = JoeCopula{2}(θ)

    # Exercise both coordinate positions and both boundary branches. Interior
    # values along a given border use the same implementation path.
    @test pdf(C, [0.0, 0.5]) == 0
    @test pdf(C, [0.5, 0.0]) == 0
    @test pdf(C, [1.0, 0.5]) == 0
    @test pdf(C, [0.5, 1.0]) == 0

    G = GumbelCopula{2}(2.5)
    @test pdf(G, [0.1,0.0]) == 0.0
    @test pdf(G, [0.0,0.1]) == 0.0
    @test pdf(G, [0.0,0.0]) == 0.0

    # Issue 247
    @test pdf(FrankCopula{2}(2.5), [1,1]*eps()) ≈ 2.723563724584597
    @test pdf(FrankCopula{2}(-2.5), [1,1]*eps()) ≈ 0.22356372458463078
    @test pdf(FrankCopula{2}(-2.5), [1,1]*0.0) == 0.0
    @test pdf(FrankCopula{2}(2.5), [1,1]*0.0) == 0.0
    @test isapprox(pdf(SklarDist(FrankCopula{2}(-2.5),(Normal(-2.,1),Normal(-0.3,0.1))), [2.,-2.]), 0.0, atol=eps())

end

@testset "empirical generator inverse" begin
    empirical = EmpiricalGenerator(_FIXTURE_DATA)
    @test Copulas.ϕ(empirical, Copulas.ϕ⁻¹(empirical, 0.5)) ≈ 0.5 atol=1e-8
end

@testset "bivariate Clayton CDF/PDF numerical anchors" begin
    # Fix a few cdf and pdf values:
    x = [0:0.25:1;]
    y = x
    cdf1 = [0.0, 0.1796053020267749, 0.37796447300922725, 0.6255432421712244, 1.0]
    cdf2 = [0.0, 0.0, 0.17157287525381, 0.5358983848622453, 1.0]
    pdf1 = [0.0, 2.2965556205046926, 1.481003649342278, 1.614508582188617, 0.0]
    pdf2 = [0.0, 0.0, 1.0, 2 / 3, 0.0]
    # Endpoints are part of the universal copula contract. These three points
    # retain the negative-support cutoff and two distinct interior regimes.
    for i in 2:4
        @test cdf(ClaytonCopula{2}(2),[x[i],y[i]]) ≈ cdf1[i]
        @test cdf(ClaytonCopula{2}(-0.5),[x[i],y[i]]) ≈ cdf2[i]
        @test pdf(ClaytonCopula{2}(2),[x[i],y[i]]) ≈ pdf1[i]
        @test pdf(ClaytonCopula{2}(-0.5),[x[i],y[i]]) ≈ pdf2[i]
    end
end


@testset "Clayton generator is continuous through θ = eps" begin
    # The power forms of the generator cancel below eps: (0.5^-θ - 1)/θ is 0,
    # so every CDF value was 1 and Blomqvist's beta was 3. The logarithmic
    # forms agree with the power forms where those are healthy, and reach the
    # independence branch continuously.
    for θ in (1e-12, 1e-17, 1e-100, 1e-300)
        C = ClaytonCopula{2}(θ)
        @test cdf(C, [0.5, 0.5]) ≈ 0.25 rtol=1e-11
        @test pdf(C, [0.3, 0.7]) ≈ 1 rtol=1e-11
        @test Copulas.β(C) ≈ 0 atol=1e-11
    end
    for θ in (-0.5, 0.3, 1.4, 2.0, 5.0), t in (0.1, 1.0, 7.0), k in 1:3
        G = Copulas.ClaytonGenerator(θ)
        @test Copulas.ϕ(G, t) ≈ max(1 + θ * t, 0.0)^(-1 / θ) rtol=1e-14
        @test Copulas.ϕ⁻¹(G, 0.5) ≈ (0.5^(-θ) - 1) / θ rtol=1e-14
        # Outside the support of a negative θ the derivatives are zero and
        # the inverse returns its edge, as before.
        1 + θ * t > 0 || continue
        @test Copulas.ϕ⁽¹⁾(G, t) ≈ -(1 + θ * t)^(-1 / θ - 1) rtol=1e-14
        P = prod(-1 - ℓ * θ for ℓ in 0:k-1; init=1)
        @test Copulas.ϕ⁽ᵏ⁾(G, k, t) ≈ (1 + θ * t)^(-1 / θ - k) * P rtol=1e-14
        # A zero coefficient or a zero exponent (1 + kθ = 0) leaves nothing
        # to invert.
        (iszero(P) || iszero(1 + k * θ)) && continue
        @test Copulas.ϕ⁽ᵏ⁾⁻¹(G, k, Copulas.ϕ⁽ᵏ⁾(G, k, t)) ≈ t rtol=1e-12
    end
    # The forms take their type from θ and t.
    for T in (Float32, BigFloat)
        G = Copulas.ClaytonGenerator(T(1.5))
        @test Copulas.ϕ(G, T(0.3)) isa T
        @test Copulas.ϕ⁻¹(G, T(0.3)) isa T
        @test Copulas.ϕ⁽¹⁾(G, T(0.3)) isa T
        @test Copulas.ϕ⁽ᵏ⁾(G, 2, T(0.3)) isa T
        @test Copulas.ϕ⁽ᵏ⁾⁻¹(G, 2, Copulas.ϕ⁽ᵏ⁾(G, 2, T(0.3))) isa T
        @test Copulas._archimedean_logpdf(ClaytonCopula{2}(T(1.5)), T[0.3, 0.7]) isa T
    end
    @test Copulas.ϕ⁽¹⁾(Copulas.ClaytonGenerator(-0.5), 7.0) === 0.0
    # The specialized log-density carries the same sum, Σ tᵢ^(-θ) - d + 1.
    for (d, θ) in ((2, 1e-12), (3, 1e-100), (4, 1e-300))
        u = collect(range(0.31, 0.69; length=d))
        @test Copulas._archimedean_logpdf(ClaytonCopula{d}(θ), u) ≈ 0 atol=1e-11
    end
    # The Blomqvist inversion brackets a root across zero, where a false sign
    # change next to θ = 0 used to swallow the true root and return θ = 0.
    U = rand(StableRNG(3), ClaytonCopula{2}(2.0), 200)
    @test Copulas.β(U) ≈ 0.41
    @test only(params(fit(ClaytonCopula, U; method=:ibeta))) ≈ 1.366 atol=1e-3
end

@testset "Clayton conditional numerical anchors" begin
    distortion = condition(ClaytonCopula{2}(7.3), 2, 0.6)
    @test cdf(distortion, [0.2, 0.5, 0.8]) ≈
          [0.00010958096560576897, 0.16963161864932144, 0.8987566352893012]

    conditional = condition(ClaytonCopula{3}(7.3), 3, 0.6951919277176142)
    @test cdf(conditional, [0.2, 0.3]) ≈ 3.0484941754695964e-5
    @test cdf(conditional.C, [0.2, 0.3]) ≈ 0.13034531809769517
end

@testset "Clayton specialized log-density" begin
    for (d, θ) in (
        (2, -0.5),
        (2, 0.5),
        (2, 2.0),
        (3, -0.25),
        (3, 0.5),
        (3, 2.0),
        (4, 1.5),
    )
        C = ClaytonCopula{d}(θ)
        u = collect(range(0.31, 0.69; length=d))

        specialized = Copulas._archimedean_logpdf(C, u)

        generic = @invoke Copulas._archimedean_logpdf(
            C::ArchimedeanCopula,
            u,
        )

        @test specialized ≈ generic rtol=1e-10 atol=1e-12
    end

    @test Copulas._archimedean_logpdf(
        ClaytonCopula{3}(0.0),
        [0.3, 0.5, 0.7],
    ) == 0
end
