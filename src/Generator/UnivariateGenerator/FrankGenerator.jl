"""
    FrankGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    FrankGenerator(θ)
    FrankCopula(d,θ)

The [Frank](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-\\infty,\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = -\\frac{\\log\\left(1+e^{-t}(e^{-\\theta-1})\\right)}{\theta}
```

It has a few special cases: 
- When θ = -∞, it is the WCopula (Lower Frechet-Hoeffding bound)
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct FrankGenerator{T} <: UnivariateGenerator
    θ::T
    function FrankGenerator(θ)
        if θ == -Inf
            return WGenerator()
        elseif θ == 0
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
max_monotony(G::FrankGenerator) = G.θ < 0 ? 2 : Inf
ϕ(  G::FrankGenerator, t) = G.θ > 0 ? -LogExpFunctions.log1mexp(LogExpFunctions.log1mexp(-G.θ)-t)/G.θ : -log1p(exp(-t) * expm1(-G.θ))/G.θ
ϕ(  G::FrankGenerator, t::TaylorSeries.Taylor1) = G.θ > 0 ? -log(-expm1(LogExpFunctions.log1mexp(-G.θ)-t))/G.θ : -log1p(exp(-t) * expm1(-G.θ))/G.θ
ϕ⁻¹(G::FrankGenerator, t) = G.θ > 0 ? LogExpFunctions.log1mexp(-G.θ) - LogExpFunctions.log1mexp(-t*G.θ) : -log(expm1(-t*G.θ)/expm1(-G.θ))
ϕ⁽¹⁾(G::FrankGenerator, t) = (one(t) - one(t) / (one(t) + exp(-t)*expm1(-G.θ))) / G.θ  # First derivative of ϕ
# ϕ⁽ᵏ⁾(G::FrankGenerator, k, t) = kth derivative of ϕ
williamson_dist(G::FrankGenerator, d) = G.θ > 0 ?  WilliamsonFromFrailty(Logarithmic(-G.θ), d) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),d)

Debye(x, k::Int=1) = k / x^k * QuadGK.quadgk(t -> t^k/expm1(t), 0, x)[1]
function τ(G::FrankGenerator)
    θ = G.θ
    T = promote_type(typeof(θ),Float64)
    if abs(θ) < sqrt(eps(T))
        # return the taylor approx. 
        return θ/9 * (1 - (θ/10)^2)
    else
        return 1+4(Debye(θ,1)-1)/θ
    end
end
function τ⁻¹(::Type{T},tau) where T<:FrankGenerator
    s,v = sign(tau),abs(tau)
    if v == 0
        return v
    elseif v == 1
        return s * Inf
    else
        return s*Roots.fzero(x -> τ(FrankGenerator(x))-v, 0, Inf)
    end
end
