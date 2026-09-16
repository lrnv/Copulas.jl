# Downstream-style generator exercising only the documented public extension
# contract and therefore the package's generic Taylor derivative fallback.
struct _CubicCutoffGenerator{T} <: Copulas.Generator
    r::T
end
Distributions.params(G::_CubicCutoffGenerator) = (; r=G.r)
Copulas.max_monotony(::_CubicCutoffGenerator) = 4
function Copulas.ϕ(G::_CubicCutoffGenerator, t)
    t >= G.r && return zero(t)
    return (one(t) - t / G.r)^3
end

@testset "generic generator Taylor padding preserves numeric type" begin
    for T in (Float32, BigFloat)
        G = _CubicCutoffGenerator(T(1))
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

    G64 = _CubicCutoffGenerator(1.0)
    @test Copulas.ϕ⁽ᵏ⁾(G64, 4, 0.25) === 0.0
end
