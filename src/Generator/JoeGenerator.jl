"""
    JoeGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    JoeGenerator(θ)
    JoeCopula(d,θ)

The [Joe](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = 1 - \\big(1 - e^{-t}\\big)^{1/\\theta}.
```

It has a few special cases:
- When θ = 1, it is the IndependentCopula
- When θ = ∞, it is the MCopula (Upper Fréchet–Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct JoeGenerator{T} <: AbstractUnivariateFrailtyGenerator
    θ::T
    function JoeGenerator(θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than 1"))
        elseif θ == 1
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const JoeCopula{d, T} = ArchimedeanCopula{d, JoeGenerator{T}}
frailty(G::JoeGenerator) = Sibuya(1/G.θ)
Distributions.params(G::JoeGenerator) = (θ = G.θ,)
_unbound_params(::Type{<:JoeGenerator}, d, θ) = [log(θ.θ - 1)]
_rebound_params(::Type{<:JoeGenerator}, d, α) = (; θ = 1 + exp(α[1]))
_θ_bounds(::Type{<:JoeGenerator}, d) = (1, Inf)

ϕ(  G::JoeGenerator, t) = 1-(-expm1(-t))^(1/G.θ)
ϕ⁻¹(G::JoeGenerator, t) = -log1p(-(1-t)^G.θ)
ϕ⁽¹⁾(G::JoeGenerator, t) = (-expm1(-t))^(1/G.θ) / (G.θ - G.θ * exp(t))
function ϕ⁽ᵏ⁾(G::JoeGenerator, d::Int, t)
    # TODO: test if this ϕ⁽ᵏ⁾ is really more 'efficient' than the default one, 
    # as we already saw that for the Gumbel is wasn't the case. 
    α = 1 / G.θ
    x = exp(-t)
    y = -expm1(-t)
    r = x/y
    P_d_α = sum(Combinatorics.stirlings2(d, k) * (SpecialFunctions.gamma(k - α) / SpecialFunctions.gamma(1 - α)) * r^(k-1) for k in 1:d)
    return (-1)^d * α * (x / y^(1 - α)) * P_d_α
end
function ϕ⁻¹⁽¹⁾(G::JoeGenerator, t)
    return -(G.θ * (1 - t)^(G.θ - 1)) / (1 - (1 - t)^G.θ)
end

_joe_tau(θ) =  1 - 4sum(1/(k*(2+k*θ)*(θ*(k-1)+2)) for k in 1:1000)
τ(G::JoeGenerator) = _joe_tau(G.θ)
function τ⁻¹(::Type{<:JoeGenerator}, τ)
    l, u = one(τ), τ * Inf
    τ ≤ 0 && return l
    τ ≥ 1 && return u
    τ = clamp(τ, 0, 1)
    return Roots.find_zero(θ -> _joe_tau(θ) - τ, (l, u))
end

_rho_joe(θ) = @invoke ρ(JoeCopula(2, θ)::Copula)
ρ(G::JoeGenerator) = _rho_joe(G.θ)
function ρ⁻¹(::Type{<:JoeGenerator}, ρ)
    l, u = one(ρ), ρ * Inf
    ρ ≤ 0 && return l
    ρ ≥ 1 && return u
    return Roots.find_zero(θ -> _rho_joe(θ) - ρ, (1, Inf))
end