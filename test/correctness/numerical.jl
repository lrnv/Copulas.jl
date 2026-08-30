# Correctness obligation: independent numerical regressions for internal
# primitives shared by several public families.
@testset "stable factorial recurrences" begin
    @test Copulas._falling_factorial(19.0, 2) == 342.0
    @test Copulas._falling_factorial(3.5, 2) == 8.75
    @test Copulas._mul_factorial(1.0, 22) ≈ gamma(23)
    @test Copulas._div_factorial(1.0, 22) ≈ inv(gamma(23))
    @test Copulas._rising_factorial(0.5, 9) ≈ gamma(9.5) / gamma(0.5)

    G = Copulas.ClaytonGenerator(1.0)
    generic_derivative = invoke(
        Copulas.ϕ⁽ᵏ⁾,
        Tuple{Copulas.Generator, Int, Any},
        G,
        22,
        1.0,
    )
    @test generic_derivative ≈ Copulas.ϕ⁽ᵏ⁾(G, 22, 1.0)

    radial = Copulas.𝒲₋₁(G, 22)
    @test 0 <= cdf(radial, 1.0) <= 1

    clayton_radial = Copulas.ClaytonWilliamsonDistribution(-0.001, 25)
    @test cdf(clayton_radial, 0.0) == 0
    @test 0 <= cdf(clayton_radial, 500.0) <= 1
    @test isfinite(logpdf(clayton_radial, 500.0))

    @test isfinite(Copulas.γ(rand(rng, 25, 10)))
end
