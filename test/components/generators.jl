const GENERATOR_CASES = (
    Copulas.AMHGenerator(0.5),
    Copulas.BB1Generator(1.2, 1.5),
    Copulas.BB2Generator(1.2, 0.5),
    Copulas.BB3Generator(2.0, 1.5),
    Copulas.BB6Generator(1.2, 1.6),
    Copulas.BB7Generator(1.2, 1.6),
    Copulas.BB8Generator(1.2, 0.4),
    Copulas.BB9Generator(1.5, 2.4),
    Copulas.BB10Generator(1.5, 0.7),
    Copulas.ClaytonGenerator(1.5),
    Copulas.FrankGenerator(2.0),
    Copulas.GumbelBarnettGenerator(0.5),
    Copulas.GumbelGenerator(1.5),
    Copulas.InvGaussianGenerator(0.5),
    Copulas.JoeGenerator(1.5),
    WilliamsonGenerator(Dirac(1.0), 2.0),
)

@testset "public generator primitives" begin
    for G in GENERATOR_CASES
        @testset "$(nameof(typeof(G)))" begin
            @test Copulas.max_monotony(G) >= 2
            @test Copulas.ϕ(G, 0.0) ≈ 1
            @test 0 <= Copulas.ϕ(G, 0.7) <= 1
            p = Copulas.ϕ(G, 0.7)
            @test Copulas.ϕ⁻¹(G, p) ≈ 0.7 atol=2e-6 rtol=2e-6
            @test Copulas.ϕ⁽¹⁾(G, 0.7) <= 0
            @test Copulas.ϕ⁽ᵏ⁾(G, 0, 0.7) ≈ p
        end
    end
end
