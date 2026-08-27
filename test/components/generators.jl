# Public-component layer: exhaustively covers public generator families and
# verifies their transform, inverse, derivative, and reconstruction identities.
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
    Copulas.FrailtyGenerator(Exponential()),
    WilliamsonGenerator(Dirac(1.0), 2.0),
    WilliamsonGenerator(Dirac(1.0), 2.5),
)

const ALL_PUBLIC_GENERATORS = (
    GENERATOR_CASES...,
    Copulas.IndependentGenerator(), Copulas.MGenerator(), Copulas.WGenerator(),
    EmpiricalGenerator(_FIXTURE_DATA),
)

@testset "public generator registry is exhaustive" begin
    public_families = Set(getfield(Copulas, symbol) for symbol in PUBLIC_SYMBOLS
        if getfield(Copulas, symbol) isa Type &&
           symbol !== :Generator &&
           getfield(Copulas, symbol) <: Copulas.Generator)
    represented = Set(typeof(G) for G in ALL_PUBLIC_GENERATORS)
    @test all(F -> any(T -> T <: F, represented), public_families)
    @test all(T -> any(F -> T <: F, public_families), represented)
end

@testset "public generator primitives" begin
    for G in GENERATOR_CASES
        @testset "$(nameof(typeof(G)))" begin
            @test G isa Copulas.Generator
            @test Copulas.max_monotony(G) >= 2
            @test params(G) isa NamedTuple
            rebuilt = typeof(G)(values(params(G))...)
            @test params(rebuilt) == params(G)
            @test Copulas.ϕ(G, 0.0) ≈ 1
            @test 0 <= Copulas.ϕ(G, 0.7) <= 1
            p = Copulas.ϕ(G, 0.7)
            @test Copulas.ϕ⁻¹(G, p) ≈ 0.7 atol=2e-6 rtol=2e-6
            @test Copulas.ϕ⁽¹⁾(G, 0.7) <= 0
            @test Copulas.ϕ⁽ᵏ⁾(G, 0, 0.7) ≈ p
            derivative_rtol = G isa WilliamsonGenerator ? 1e-4 : 2e-7
            @test Copulas.ϕ⁽¹⁾(G, 0.7) ≈
                  ForwardDiff.derivative(t -> Copulas.ϕ(G, t), 0.7) rtol=derivative_rtol
            @test Copulas.ϕ⁽ᵏ⁾(G, 1, 0.7) ≈ Copulas.ϕ⁽¹⁾(G, 0.7)
            @test Copulas.ϕ⁽ᵏ⁾(G, 2, 0.7) ≈
                  ForwardDiff.derivative(t -> Copulas.ϕ⁽¹⁾(G, t), 0.7) rtol=derivative_rtol
            h = 1e-5
            inverse_derivative = (Copulas.ϕ⁻¹(G, 0.5 + h) -
                                  Copulas.ϕ⁻¹(G, 0.5 - h)) / (2h)
            @test Copulas.ϕ⁻¹⁽¹⁾(G, 0.5) ≈ inverse_derivative rtol=2e-5
            y = Copulas.ϕ⁽ᵏ⁾(G, 1, 0.3)
            @test Copulas.ϕ⁽ᵏ⁾⁻¹(G, 1, y) ≈ 0.3 atol=2e-5 rtol=2e-5
        end
    end
end
