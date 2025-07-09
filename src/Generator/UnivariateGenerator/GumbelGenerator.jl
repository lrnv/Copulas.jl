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

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
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
ϕ(G::GumbelGenerator, t::Number) = exp(-t^(1 / G.θ))
ϕ⁻¹(G::GumbelGenerator, t::Real) = (-log(t))^G.θ
function ϕ⁽¹⁾(G::GumbelGenerator, t::Real)
    # first derivative of ϕ
    a = 1 / G.θ
    tam1 = t^(a - 1)
    return -a * tam1 * exp(-tam1 * t)
end
function ϕ⁻¹⁽¹⁾(G::GumbelGenerator, t::Real)
    return -(G.θ * (-log(t))^(G.θ - 1)) / t
end
τ(G::GumbelGenerator) = ifelse(isfinite(G.θ), (G.θ - 1) / G.θ, 1)
function τ⁻¹(::Type{T}, τ) where {T<:GumbelGenerator}
    if τ == 1
        return Inf
    else
        θ = 1 / (1 - τ)
        if θ < 1
            @warn "GumbelCopula cannot handle negative kendall tau's, returning independence.."
            return 1
        end
        return θ
    end
end
function williamson_dist(G::GumbelGenerator, d)
    return WilliamsonFromFrailty(
        AlphaStable(;
            α=1 / G.θ, β=1, scale=cos(π / (2G.θ))^G.θ, location=(G.θ == 1 ? 1 : 0)
        ),
        d,
    )
end

using BigCombinatorics

"""
M. Hofert, M. Mächler, and A. J. McNeil, ‘Likelihood inference for Archimedean copulas in high dimensions under known margins’, Journal of Multivariate Analysis, vol. 110, pp. 133–150, Sep. 2012, doi: 10.1016/j.jmva.2012.02.019.
"""
function ϕ⁽ᵏ⁾(G::GumbelGenerator, d::Integer, t::Real)
    α = 1 / G.θ

    return ϕ(G, t) *
           t^(-d) *
           sum([
               α^j * Stirling1(d, j) * sum([Stirling2(j, k) * (-t^α)^k for k in 1:j]) for
               j in 1:d
           ])
end
