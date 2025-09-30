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
    JoeGenerator{T}(θ) where T = JoeGenerator(promote(θ, one(T))[1])
end
const JoeCopula{d, T} = ArchimedeanCopula{d, JoeGenerator{T}}
JoeCopula(d, θ) = ArchimedeanCopula(d, JoeGenerator(θ))
JoeCopula(d; θ::Real) = JoeCopula(d, θ)
frailty(G::JoeGenerator) = Sibuya(1/G.θ)
Distributions.params(G::JoeGenerator) = (θ = G.θ,)
_example(CT::Type{<:JoeCopula}, d) = JoeCopula(d, 1.5)
_example(::Type{ArchimedeanCopula{2, JoeGenerator}}, d) = JoeCopula(d, 1.5)
_unbound_params(::Type{<:JoeCopula}, d, θ) = [log(θ.θ - 1)]
_rebound_params(::Type{<:JoeCopula}, d, α) = (; θ = 1 + exp(α[1]))
_θ_bounds(::Type{<:JoeGenerator}, d) = (1, Inf)

ϕ(  G::JoeGenerator, t) = 1-(-expm1(-t))^(1/G.θ)
ϕ⁻¹(G::JoeGenerator, t) = -log1p(-(1-t)^G.θ)
ϕ⁽¹⁾(G::JoeGenerator, t) = (-expm1(-t))^(1/G.θ) / (G.θ - G.θ * exp(t))
function ϕ⁽ᵏ⁾(G::JoeGenerator, ::Val{d}, t) where d
    # TODO: test if this ϕ⁽ᵏ⁾ is really more 'efficient' than the default one, 
    # as we already saw that for the Gumbel is wasn't the case. 
    α = 1 / G.θ
    P_d_α = sum(
        Float64(BigCombinatorics.Stirling2(d, k + 1)) *
        (SpecialFunctions.gamma(k + 1 - α) / SpecialFunctions.gamma(1 - α)) *
        (exp(-t) / (-expm1(-t)))^k for k in 0:(d - 1)
    )
    return (-1)^d * α * (exp(-t) / (-expm1(-t))^(1 - α)) * P_d_α
end
function ϕ⁻¹⁽¹⁾(G::JoeGenerator, t)
    return -(G.θ * (1 - t)^(G.θ - 1)) / (1 - (1 - t)^G.θ)
end
_joe_tau(θ) =  1 - 4sum(1/(k*(2+k*θ)*(θ*(k-1)+2)) for k in 1:1000) # 446 in R copula.
τ(G::JoeGenerator) = _joe_tau(G.θ)
function τ⁻¹(::Type{<:JoeGenerator}, τ)
    l, u = one(τ), τ * Inf
    τ ≤ 0 && return l
    τ ≥ 1 && return u
    τ = clamp(τ, 0, 1)
    return Roots.find_zero(θ -> _joe_tau(θ) - τ, (l, u))
end

function _rho_joe_via_cdf(θ; rtol=1e-7, atol=1e-9, maxevals=10^6)
    θ = clamp(θ, 1, Inf)
    θ <= 1 && return zero(θ)
    isinf(θ) && return one(θ)
    Cθ   = Copulas.ArchimedeanCopula(2, JoeGenerator(θ))
    f(x) = Distributions.cdf(Cθ, x)
    I = HCubature.hcubature(f, (0.0,0.0), (1.0,1.0); rtol=rtol, atol=atol, maxevals=maxevals)[1]
    return 12I - 3
end

ρ(G::JoeGenerator) = _rho_joe_via_cdf(G.θ)
function ρ⁻¹(::Type{<:JoeGenerator}, ρ)
    l, u = one(ρ), ρ * Inf
    ρ ≤ 0 && return l
    ρ ≥ 1 && return u
    Roots.find_zero(θ -> _rho_joe_via_cdf(θ) - ρ, (l,u))
end