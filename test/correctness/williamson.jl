# Mathematical correctness of real-order Williamson transforms and exact
# order reduction. Public distribution-shape contracts are tested separately.
@testset "real-order Williamson identities" begin
    X = Dirac(2.0)
    G4 = @inferred 𝒲(X, 4)
    G5 = 𝒲(X, 5)
    Greal = 𝒲(X, 4.5)
    @test typeof(G4) == typeof(G5)
    @test Greal.order == 4.5
    @test Copulas.max_monotony(Greal) == 4.5
    @test Copulas.ϕ(Greal, 0.5) ≈ (1 - 0.5 / 2)^3.5
    @test Copulas.ϕ⁽ᵏ⁾(Greal, 2, 0.5) ≈
          3.5 * 2.5 / 2^2 * (1 - 0.5 / 2)^1.5

    Gdiscrete = 𝒲([1.0], [1.0], 4.5)
    @test Copulas.ϕ⁽ᵏ⁾(Gdiscrete, 5, 0.5) ≈
          (-1)^5 * Copulas._falling_factorial(3.5, 5) * 0.5^(-1.5)
    Glognormal = 𝒲(LogNormal(), 2)
    @test Copulas.ϕ⁽¹⁾(Glognormal, 0.1) ≈
          -exp(0.5) * ccdf(Normal(), log(0.1) + 1)

    @test Copulas.𝒲₋₁(Greal, 4.5) === X
    radial = Copulas.𝒲₋₁(Greal, 2.0)
    beta = Beta(2.0, 2.5)
    @test cdf(radial, 0.8) ≈ cdf(beta, 0.4)
    @test pdf(radial, 0.8) ≈ pdf(beta, 0.4) / 2

    pareto_radial = Copulas.𝒲₋₁(𝒲(Pareto(1), 5), 2)
    @test cdf(pareto_radial, 2.0) ≈ 0.8
    @test pdf(pareto_radial, 2.0) ≈ 0.1

    nested = Copulas.WilliamsonBetaProduct(radial, Beta(1.0, 1.0))
    @test nested.X === X
    @test Distributions.params(nested.B) == (1.0, 3.5)
    @test nested.source_order == Greal.order
    recovered = 𝒲(radial, 2.0)
    @test recovered.X === X
    @test recovered.order == 4.5

    generic_radial = Copulas.𝒲₋₁(Copulas.FrankGenerator(-2.0), 2)
    @test 𝒲(generic_radial, 2) === generic_radial.G
    remapped = 𝒲(generic_radial, 3)
    @test remapped.X === generic_radial
    @test remapped.order == 3
end
