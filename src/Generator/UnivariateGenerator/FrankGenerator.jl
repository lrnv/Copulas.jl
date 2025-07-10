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
# generator
function ϕ(G::FrankGenerator, t::Number)
    return if G.θ > 0
        -LogExpFunctions.log1mexp(LogExpFunctions.log1mexp(-G.θ) - t) / G.θ
    else
        -log1p(exp(-t) * expm1(-G.θ)) / G.θ
    end
end
# first generator derivative
function ϕ⁽¹⁾(G::FrankGenerator, t::Real)
    return (one(t) - one(t) / (one(t) + exp(-t) * expm1(-G.θ))) / G.θ
end
# kth generator derivative
function ϕ⁽ᵏ⁾(G::FrankGenerator, k::Integer, t::Real)
    return (-1)^k * (1 / G.θ) * reli(-(k - 1), (1 - exp(-G.θ)) * exp(-t))
end
# inverse generator
function ϕ⁻¹(G::FrankGenerator, t::Real)
    return if G.θ > 0
        LogExpFunctions.log1mexp(-G.θ) - LogExpFunctions.log1mexp(-t * G.θ)
    else
        -log(expm1(-t * G.θ) / expm1(-G.θ))
    end
end
# first inverse generator derivative
function ϕ⁻¹⁽¹⁾(G::FrankGenerator, t::Real)
    return G.θ / (-expm1(G.θ * t))
end
function williamson_dist(G::FrankGenerator, d)
    return if G.θ > 0
        WilliamsonFromFrailty(Logarithmic(-G.θ), d)
    else
        WilliamsonTransforms.𝒲₋₁(t -> ϕ(G, t), d)
    end
end

Debye(x, k::Int=1) = k / x^k * QuadGK.quadgk(t -> t^k / expm1(t), 0, x)[1]
function τ(G::FrankGenerator)
    θ = G.θ
    T = promote_type(typeof(θ), Float64)
    if abs(θ) < sqrt(eps(T))
        # return the taylor approx.
        return θ / 9 * (1 - (θ / 10)^2)
    else
        return 1 + 4(Debye(θ, 1) - 1) / θ
    end
end
function τ⁻¹(::Type{T}, tau) where {T<:FrankGenerator}
    s, v = sign(tau), abs(tau)
    if v == 0
        return v
    elseif v == 1
        return s * Inf
    else
        return s * Roots.fzero(x -> τ(FrankGenerator(x)) - v, 0, Inf)
    end
end
