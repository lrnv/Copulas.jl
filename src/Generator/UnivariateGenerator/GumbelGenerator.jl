"""
    GumbelGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    GumbelGenerator(θ)
    GumbelCopula(d,θ)

The [Gumbel](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = \\exp{-t^{\\frac{1}{θ}}}
```

It has a few special cases: 
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)
"""
struct GumbelGenerator{T} <: UnivariateGenerator
    θ::T
    function GumbelGenerator(θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than or equal to 1"))
        elseif θ == 1
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
max_monotony(G::GumbelGenerator) = Inf
ϕ(  G::GumbelGenerator, t) = exp(-t^(1/G.θ))
ϕ⁻¹(G::GumbelGenerator, t) = (-log(t))^G.θ
# ϕ⁽¹⁾(G::GumbelGenerator, t) =  First derivative of ϕ
# ϕ⁽ᵏ⁾(G::GumbelGenerator, k, t) = kth derivative of ϕ
τ(G::GumbelGenerator) = ifelse(isfinite(G.θ), (G.θ-1)/G.θ, 1)
function τ⁻¹(::Type{T},τ) where T<:GumbelGenerator 
    if τ == 1
        return Inf
    else
        θ = 1/(1-τ)
        if θ < 1
            @warn "GumbelCopula cannot handle negative kendall tau's, returning independence.."
            return 1
        end
        return θ
    end
end
williamson_dist(G::GumbelGenerator, d) = WilliamsonFromFrailty(AlphaStable(α = 1/G.θ, β = 1,scale = cos(π/(2G.θ))^G.θ, location = (G.θ == 1 ? 1 : 0)), d)