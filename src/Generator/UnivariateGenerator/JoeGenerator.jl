"""
    JoeGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    JoeGenerator(θ)
    JoeCopula(d,θ)

The [Joe](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = 1 - \\left(1 - e^{-t}\\right)^{\\frac{1}{\\theta}}
```

It has a few special cases:
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct JoeGenerator{T} <: UnivariateGenerator
    θ::T
    function JoeGenerator(θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than 1"))
        elseif θ == 1
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
max_monotony(G::JoeGenerator) = Inf
# generator
ϕ(G::JoeGenerator, t::Number) = 1 - (-expm1(-t))^(1 / G.θ)
# first generator derivative
ϕ⁽¹⁾(G::JoeGenerator, t::Real) = (-expm1(-t))^(1 / G.θ) / (G.θ - G.θ * exp(t))
# kth generator derivative
function ϕ⁽ᵏ⁾(G::JoeGenerator, d::Integer, t::Real)
    α = 1 / G.θ
    P_d_α = sum([
        Stirling2(d, k + 1) *
        (SpecialFunctions.gamma(k + 1 - α) / SpecialFunctions.gamma(1 - α)) *
        (exp(-t) / (-expm1(-t)))^k for k in 0:(d - 1)
    ])
    return (-1)^d * α * (exp(-t) / (-expm1(-t))^(1 - α)) * P_d_α
end
# inverse generator
ϕ⁻¹(G::JoeGenerator, t::Real) = -log1p(-(1 - t)^G.θ)
# first inverse generator derivative
function ϕ⁻¹⁽¹⁾(G::JoeGenerator, t::Real)
    return -(G.θ * (1 - t)^(G.θ - 1)) / (1 - (1 - t)^G.θ)
end
τ(G::JoeGenerator) = 1 - 4sum(1 / (k * (2 + k * G.θ) * (G.θ * (k - 1) + 2)) for k in 1:1000) # 446 in R copula.
function τ⁻¹(::Type{T}, tau) where {T<:JoeGenerator}
    if tau == 1
        return Inf
    elseif tau == 0
        return 1
    elseif tau < 0
        @info "JoeCoula cannot handle κ < 0."
        return one(tau)
    else
        return Roots.find_zero(θ -> τ(JoeGenerator(θ)) - tau, (one(tau), tau * Inf))
    end
end
williamson_dist(G::JoeGenerator, d) = WilliamsonFromFrailty(Sibuya(1 / G.θ), d)
