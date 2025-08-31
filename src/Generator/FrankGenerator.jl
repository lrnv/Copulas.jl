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
struct FrankGenerator{T} <: Generator
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
const FrankCopula{d, T} = ArchimedeanCopula{d, FrankGenerator{T}}
FrankCopula(d, θ) = ArchimedeanCopula(d, FrankGenerator(θ))
Distributions.params(C::FrankCopula) = (C.G.θ,)

max_monotony(G::FrankGenerator) = G.θ < 0 ? 2 : Inf
ϕ(G::FrankGenerator, t) = G.θ > 0 ? -LogExpFunctions.log1mexp(LogExpFunctions.log1mexp(-G.θ)-t)/G.θ : -log1p(exp(-t) * expm1(-G.θ))/G.θ
ϕ⁽¹⁾(G::FrankGenerator, t) = (one(t) - one(t) / (one(t) + exp(-t)*expm1(-G.θ))) / G.θ
ϕ⁻¹⁽¹⁾(G::FrankGenerator, t) = G.θ / (-expm1(G.θ * t))
function ϕ⁽ᵏ⁾(G::FrankGenerator, ::Val{k}, t) where k
    return (-1)^k * (1 / G.θ) * PolyLog.reli(-(k - 1), (1 - exp(-G.θ)) * exp(-t))
end
ϕ⁻¹(G::FrankGenerator, t) = G.θ > 0 ? LogExpFunctions.log1mexp(-G.θ) - LogExpFunctions.log1mexp(-t*G.θ) : -log(expm1(-t*G.θ)/expm1(-G.θ))
williamson_dist(G::FrankGenerator, ::Val{d}) where d = G.θ > 0 ? WilliamsonFromFrailty(Logarithmic(-G.θ), Val{d}()) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),Val{d}())

Debye(x, k::Int=1) = k / x^k * QuadGK.quadgk(t -> t^k/expm1(t), 0, x)[1]
function _frank_tau(θ)
    T = promote_type(typeof(θ),Float64)
    if abs(θ) < sqrt(eps(T))
        # return the taylor approx.
        return θ/9 * (1 - (θ/10)^2)
    else
        return 1+4(Debye(θ,1)-1)/θ
    end
end
τ(G::FrankGenerator) = _frank_tau(G.θ)
function τ⁻¹(::Type{T},tau) where T<:FrankGenerator
    s,v = sign(tau),abs(tau)
    if v == 0
        return v
    elseif v == 1
        return s * Inf
    else
        return s*Roots.fzero(x -> _frank_tau(x)-v, 0, Inf)
    end
end
