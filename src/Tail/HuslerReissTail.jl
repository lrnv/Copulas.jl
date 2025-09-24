"""
    HuslerReissTail{T}

Fields:
  - θ::Real — dependence parameter, θ ≥ 0

Constructor

    HuslerReissCopula(θ)
    ExtremeValueCopula(2, HuslerReissTail(θ))

The (bivariate) Hüsler-Reiss extreme-value copula is parameterized by ``\\theta \\in [0, \\infty)``.
Its Pickands dependence function is

```math
A(t) = t\\Phi(\\theta^{-1}+\\frac{1}{2}\\theta\\log(\\frac{t}{1-t})) +(1-t)\\Phi(\\theta^{-1}+\\frac{1}{2}\\theta\\log(\\frac{1-t}{t}))
```
where ``\\Phi`` is the standard normal cdf.

Special cases:

* θ = 0   ⇒ IndependentCopula
* θ = ∞   ⇒ MCopula (upper Fréchet-Hoeffding bound)

References:

* [husler1989maxima](@cite) Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between independence and complete dependence. Statistics & Probability Letters, 7(4), 283-286.
"""
struct HuslerReissTail{T} <: AbstractUnivariateTail2
    θ::T
    function HuslerReissTail(θ)
        θ < 0 && throw(ArgumentError("θ must be ≥ 0"))
        θ == 0 && return NoTail()
        isinf(θ) && return MTail()
    return new{typeof(θ)}(θ)
    end
end
const HuslerReissCopula{T} = ExtremeValueCopula{2, HuslerReissTail{T}}
HuslerReissCopula(θ) = ExtremeValueCopula(2, HuslerReissTail(θ))
HuslerReissCopula(d::Integer, θ) = ExtremeValueCopula(2, HuslerReissTail(θ))
Distributions.params(tail::HuslerReissTail) = (θ = tail.θ,)
_θ_bounds(::Type{<:HuslerReissTail}, d) = (0, Inf)
function A(tail::HuslerReissTail, t::Real)
    tt = _safett(t)
    θ  = tail.θ
    θ == 0 && return 1.0
    isinf(θ) && return max(tt, 1-tt)
    Φ = Distributions.cdf
    N = Distributions.Normal()
    term1 = tt * Φ(N, inv(θ) + 0.5*θ*log(tt/(1-tt)))
    term2 = (1-tt) * Φ(N, inv(θ) + 0.5*θ*log((1-tt)/tt))
    return term1 + term2
end

# Fitting helpers for EV copulas using Hüsler–Reiss tail
_example(::Type{<:HuslerReissCopula}, d) = ExtremeValueCopula(2, HuslerReissTail(1.0))
_unbound_params(::Type{<:HuslerReissCopula}, d, θ) = [log(θ.θ)]
_rebound_params(::Type{<:HuslerReissCopula}, d, α) = (; θ = exp(α[1]))

function ℓ(C::ExtremeValueCopula{2,HuslerReissTail{T}}, t) where {T}
    t₁, t₂ = t
    θ = C.tail.θ
    Φ = Distributions.cdf
    N = Distributions.Normal()
    return t₁ * Φ(N, inv(θ) + 0.5*θ*log(t₁/t₂)) + t₂ * Φ(N, inv(θ) + 0.5*θ*log(t₂/t₁))
end

function dA(C::ExtremeValueCopula{2,HuslerReissTail{T}}, t::Real) where {T}
    θ = C.tail.θ
    N = Distributions.Normal()
    Φ = Distributions.cdf
    ϕ = Distributions.pdf

    arg1 = inv(θ) + 0.5*θ*log(t/(1-t))
    arg2 = inv(θ) + 0.5*θ*log((1-t)/t)

    dA_term1 = Φ(N, arg1) + t * ϕ(N, arg1) * (0.5*θ * (1/t + 1/(1-t)))
    dA_term2 = -Φ(N, arg2) + (1-t) * ϕ(N, arg2) * (0.5*θ * (-1/t - 1/(1-t)))

    return dA_term1 + dA_term2
end
function d²A(C::ExtremeValueCopula{2,HuslerReissTail{T}}, t::Real) where {T}
    θ = C.tail.θ
    N  = Distributions.Normal()
    ϕ  = Distributions.pdf
    invθ = inv(θ)
    L   = log(t/(1 - t))
    a1  = invθ + 0.5*θ*L
    a2  = invθ - 0.5*θ*L
    s   = 1/t + 1/(1 - t)
    s2  = -1/t^2 + 1/(1 - t)^2
    a1p = 0.5*θ*s
    a1pp= 0.5*θ*s2
    ϕ1  = ϕ(N, a1)
    ϕ2  = ϕ(N, a2)
    return 2*(ϕ1 + ϕ2)*a1p + t*ϕ1*(a1pp - a1*a1p^2) + (1 - t)*ϕ2*(-a1pp - a2*a1p^2)
end

_tau_HuslerReiss(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : QuadGK.quadgk(t -> d²A(HuslerReissTail(θ),t)*t*(1-t)/max(A(HuslerReissTail(θ),t),_δ(t)), 0, 1; kw...)[1]
_rho_HuslerReiss(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12*QuadGK.quadgk(t -> inv(1+A(HuslerReissTail(θ),t))^2, 0, 1; kw...)[1] - 3

τ(C::HuslerReissCopula) = _tau_HuslerReiss(C.tail.θ)
ρ(C::HuslerReissCopula) = _rho_HuslerReiss(C.tail.θ)
λᵤ(C::HuslerReissCopula) = 2 * (1 - Distributions.cdf(Distributions.Normal(), 1 / C.tail.θ))
β(C::HuslerReissCopula) = 4^(1 - Distributions.cdf(Distributions.Normal(), 1/C.tail.θ)) - 1

τ⁻¹(::Type{<:HuslerReissCopula}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? θmax : _invmono(θ -> _tau_HuslerReiss(θ) - τ; kw...)
ρ⁻¹(::Type{<:HuslerReissCopula}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? θmax : _invmono(θ -> _rho_HuslerReiss(θ) - ρ; kw...)
λᵤ⁻¹(::Type{<:HuslerReissCopula}, λ) = 1 / Distributions.quantile(Distributions.Normal(), 1 - λ/2)
β⁻¹(::Type{<:HuslerReissCopula}, beta) = 1 / Distributions.quantile(Distributions.Normal(), 1 - log(beta + 1) / log(4))
