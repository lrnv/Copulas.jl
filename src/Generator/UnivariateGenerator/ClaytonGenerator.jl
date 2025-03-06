"""
    ClaytonGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    ClaytonGenerator(θ)
    ClaytonCopula(d,θ)

The [Clayton](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1/(d-1),\\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = \\left(1+\\mathrm{sign}(\\theta)*t\\right)^{-1\\frac{1}{\\theta}}
```

It has a few special cases:
- When θ = -1/(d-1), it is the WCopula (Lower Frechet-Hoeffding bound)
- When θ = 0, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct ClaytonGenerator{T} <: UnivariateGenerator
    θ::T
    function ClaytonGenerator(θ)
        if θ < -1
            throw(ArgumentError("Theta must be greater than -1"))
        elseif θ == -1
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
max_monotony(G::ClaytonGenerator) = G.θ >= 0 ? Inf : Int(floor(1 - 1 / G.θ))
# generator
ϕ(G::ClaytonGenerator, t) = (1 + t)^(-1 / G.θ)

# first generator derivative
function ϕ⁽¹⁾(G::ClaytonGenerator, t)
    α = 1 / G.θ
    return -α * (1 + t)^(-1 - α)
end
# kth generator derivative
function ϕ⁽ᵏ⁾(G::ClaytonGenerator, k, t)
    α = 1 / G.θ
    return (-1)^k * prod([0:(k - 1);] .+ α) * (1 + t)^(-(k + α))
end
# inverse generator
function ϕ⁻¹(G::ClaytonGenerator, t)
    return t^(-G.θ) - 1
end
# first inverse generator derivative
function ϕ⁻¹⁽¹⁾(G::ClaytonGenerator, t)
    return -G.θ * t^(-G.θ - 1)
end
τ(G::ClaytonGenerator) = ifelse(isfinite(G.θ), G.θ / (G.θ + 2), 1)
τ⁻¹(::Type{T}, τ) where {T<:ClaytonGenerator} = ifelse(τ == 1, Inf, 2τ / (1 - τ))
function williamson_dist(G::ClaytonGenerator, d)
    return if G.θ >= 0
        WilliamsonFromFrailty(Distributions.Gamma(1 / G.θ, G.θ), d)
    else
        return ClaytonWilliamsonDistribution(G.θ, d)
    end
end
