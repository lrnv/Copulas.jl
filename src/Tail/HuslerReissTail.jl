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
struct HuslerReissTail{T} <: Tail2
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
_example(::Type{ExtremeValueCopula{2, HuslerReissTail{T}}}, d) where {T} = ExtremeValueCopula(2, HuslerReissTail(one(T)))
_example(::Type{ExtremeValueCopula{2, HuslerReissTail}}, d) = ExtremeValueCopula(2, HuslerReissTail(1.0))
_unbound_params(::Type{ExtremeValueCopula{2, HuslerReissTail}}, d, θ) = [log(θ.θ)]
_rebound_params(::Type{ExtremeValueCopula{2, HuslerReissTail}}, d, α) = (; θ = exp(α[1]))

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
