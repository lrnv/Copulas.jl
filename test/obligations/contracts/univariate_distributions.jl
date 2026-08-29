# Contract obligation: checks radial and auxiliary univariate distributions
# directly, including finite/infinite support and continuous/discrete paths.
function test_continuous_univariate_contract(D; atol=2e-7)
    Base.@nospecialize D
    lo, hi = minimum(D), maximum(D)
    @test lo <= hi
    @test cdf(D, lo) == 0
    isfinite(hi) && @test cdf(D, hi) == 1

    for p in (0.2, 0.5, 0.8)
        q = quantile(D, p)
        @test lo <= q <= hi
        @test cdf(D, q) ≈ p atol=atol
        density = pdf(D, q)
        @test density >= 0
        @test iszero(density) ? logpdf(D, q) == -Inf :
              logpdf(D, q) ≈ log(density)
    end
    samples = rand(StableRNG(601), D, 4)
    @test all(x -> lo <= x <= hi, samples)
end

@testset "Williamson radial distributions" begin
    compact = Copulas.ClaytonWilliamsonDistribution(-0.25, 3)
    test_continuous_univariate_contract(compact)

    frailty_radial = Copulas.WilliamsonFromFrailty(LogNormal(), 2.5)
    test_continuous_univariate_contract(frailty_radial; atol=2e-6)

    beta_product = Copulas.WilliamsonBetaProduct(Uniform(1.0, 2.0), Beta(1.5, 1.0))
    test_continuous_univariate_contract(beta_product; atol=2e-6)

    # Gamma frailty and compatible beta reductions retain their exact laws.
    exact = Copulas.WilliamsonFromFrailty(Gamma(2.0, 3.0), 1.5)
    @test exact isa Distributions.LocationScale
    reduced = Copulas.WilliamsonBetaProduct(
        Copulas.WilliamsonFromFrailty(LogNormal(), 2.0), Beta(0.75, 1.25))
    @test reduced isa Copulas.WilliamsonFromFrailty
    @test reduced.order == 0.75

    generic_inverse = Copulas.𝒲₋₁(Copulas.GumbelBarnettGenerator(0.5), 2)
    q = quantile(generic_inverse, 0.5)
    @test q > 0
    @test cdf(generic_inverse, q) ≈ 0.5 atol=2e-6
    @test pdf(generic_inverse, q) >= 0
end

@testset "power-tilted frailty distributions" begin
    continuous = Copulas.PowerTiltedFrailty(Uniform(0.5, 2.0), 0.75, 0.4)
    test_continuous_univariate_contract(continuous; atol=2e-6)

    base = DiscreteNonParametric([1, 2, 4], [0.2, 0.5, 0.3])
    discrete = Copulas.PowerTiltedFrailty(base, 0.75, 0.4)
    @test Distributions.value_support(typeof(discrete)) == Distributions.Discrete
    @test sum(pdf(discrete, x) for x in support(base)) ≈ 1
    for p in (0.2, 0.5, 0.8)
        q = quantile(discrete, p)
        @test cdf(discrete, q) >= p
        q > minimum(discrete) && @test cdf(discrete, prevfloat(q)) < p
    end

    gamma = Copulas.PowerTiltedFrailty(Gamma(2.0, 3.0), 0.75, 0.4)
    @test gamma isa Gamma
    @test all(isapprox.(params(gamma), (2.75, inv(inv(3.0) + 0.4))))
end

@testset "conditional Liouville radial cache" begin
    D = Copulas.LiouvilleConditionalRadial(Beta(2.0, 3.0), 0.1, 3.0, 0.7)
    @test isfinite(D.normalizer) && D.normalizer > 0
    @test issorted(D.integration_knots)
    @test issorted(D.cumulative_masses)
    @test first(D.cumulative_masses) == 0
    @test last(D.cumulative_masses) == D.normalizer
    test_continuous_univariate_contract(D; atol=2e-6)
end

@testset "extreme-value radial distribution" begin
    D = Copulas.ExtremeDist(Copulas.LogTail(2.0))
    test_continuous_univariate_contract(D; atol=2e-6)
end

@testset "sampler-only positive stable distributions" begin
    stable = Copulas.PStable(0.7; scale=1.3)
    draws = rand(StableRNG(602), stable, 8)
    @test all(isfinite, draws)
    @test all(>(0), draws)
    @test rand(StableRNG(603), Copulas.PStable(1.0; scale=1.3), 4) == fill(1.3, 4)

    tilted = Copulas.TiltedPositiveStable(0.7, 1.0)
    tilted_draws = rand(StableRNG(604), tilted, 4)
    @test all(isfinite, tilted_draws)
    @test all(>(0), tilted_draws)
end

@testset "frailty sampler implementations" begin
    generators = (
        Copulas.AMHGenerator(0.5), Copulas.BB1Generator(1.2, 1.5),
        Copulas.BB2Generator(1.2, 0.5), Copulas.BB3Generator(2.0, 1.5),
        Copulas.BB6Generator(1.2, 1.6), Copulas.BB7Generator(1.2, 1.6),
        Copulas.BB8Generator(1.2, 0.4), Copulas.BB9Generator(1.5, 2.4),
        Copulas.BB10Generator(1.5, 0.7),
        Copulas.ClaytonGenerator(1.5), Copulas.FrankGenerator(2.0),
        Copulas.GumbelGenerator(1.5), Copulas.InvGaussianGenerator(0.5),
        Copulas.JoeGenerator(1.5),
    )
    frailties = map(Copulas.frailty, generators)
    @test all(x -> !isnothing(x), frailties)
    @test length(Set(typeof.(frailties))) == length(frailties)
    for (i, F) in pairs(frailties)
        draws = rand(StableRNG(700 + i), F, 2)
        @test all(isfinite, draws)
        @test all(>(0), draws)
    end

    sibuya = Copulas.Sibuya(0.6)
    @test cdf(sibuya, 0) == 0
    @test cdf(sibuya, 1) ≈ 0.6
    @test pdf(sibuya, 1) ≈ 0.6
end
