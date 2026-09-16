# Downstream-style generator exercising only the documented public extension
# contract and therefore the package's generic Taylor derivative fallback.
struct _CubicGenerator{T} <: Copulas.Generator
    r::T
end
Distributions.params(G::_CubicGenerator) = (; r=G.r)
Copulas.max_monotony(::_CubicGenerator) = 4
function Copulas.ϕ(G::_CubicGenerator, t)
    q = one(t) - t / G.r
    # Spell the polynomial multiplication explicitly. TaylorSeries currently
    # widens `Taylor1{Float32} ^ n` through `float(n)`; that independent upstream
    # promotion is not the padding bug covered by #526.
    return q * q * q
end

@testset "generic generator Taylor padding preserves numeric type" begin
    for T in (Float32, BigFloat)
        # Exercise the shortened-coefficient padding branch directly.
        shortened = t -> Copulas.TaylorSeries.Taylor1(t.coeffs[1:2])
        padded = Copulas.taylor(shortened, T(0.25), 4)
        @test length(padded) == 5
        @test eltype(padded) === T
        @test padded[3:5] == zeros(T, 3)

        G = _CubicGenerator(T(1))
        x = T(0.25)

        third = Copulas.ϕ⁽ᵏ⁾(G, 3, x)
        fourth = Copulas.ϕ⁽ᵏ⁾(G, 4, x)
        @test third isa T
        @test fourth isa T
        @test fourth === zero(T)

        radial = Copulas.𝒲₋₁(G, 4)
        F = cdf(radial, T(0.5))
        f = pdf(radial, T(0.5))
        @test F isa T
        @test f isa T
        @test F == zero(T)
        @test f == zero(T)
    end

    G64 = _CubicGenerator(1.0)
    @test Copulas.ϕ⁽ᵏ⁾(G64, 4, 0.25) === 0.0
end
