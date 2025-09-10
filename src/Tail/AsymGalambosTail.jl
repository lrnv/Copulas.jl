"""
    AsymGalambosTail{T}

Fields:
  - α::Real          — dependence parameter
  - θ::NTuple{2,Real} — asymmetry weights (length 2)

Constructor

    AsymGalambosCopula(α, θ)         # θ as a 2-vector/tuple
    ExtremeValueCopula(AsymGalambosTail(α, θ))

The (bivariate) asymmetric Galambos extreme–value copula is parameterized by
``\\alpha \\in [0, \\infty) and ``\\theta_1, \\theta_2 \\in [0,1]``. It is an EV copula with Pickands function

```math
A(t) = 1 - \\Big((\\theta_1 t)^{-\\alpha} + (\\theta_2(1-t))^{-\\alpha}\\Big)^{-1/\\alpha},\\quad t\in[0,1].
```

Special cases:

* α = 0 ⇒ IndependentCopula
* θ₁ = θ₂ = 0 ⇒ IndependentCopula
* θ₁ = θ₂ = 1 ⇒ GalambosCopula

References: 

* [Joe1990](@cite) Families of min-stable multivariate exponential and multivariate extreme value distributions. Statist. Probab, 1990.
"""
struct AsymGalambosTail{T} <: Tail2
    α::T                 # α ≥ 0
    θ::NTuple{2,T}       # 0 ≤ θ_i ≤ 1
    function AsymGalambosTail(α, θ)
        (length(θ) == 2) || throw(ArgumentError("θ must have length 2"))
        θt = (θ[1], θ[2])
        (αT ≥ 0) || throw(ArgumentError("α must be ≥ 0"))
        (0 ≤ θt[1] ≤ 1 && 0 ≤ θt[2] ≤ 1) || throw(ArgumentError("each θ[i] must be in [0,1]"))
        α == 0 || (θt[1] == 0 && θt[2] == 0) && return NoTail()
        θt[1] == 1 && θt[2] == 1 && return GalambosTail(α)
        return new{T}(αT, θt)
    end
end

const AsymGalambosCopula{T} = ExtremeValueCopula{2, AsymGalambosTail{T}}
AsymGalambosCopula(α, θ::NTuple{2,Any}) = AsymGalambosCopula(α, collect(θ))
Distributions.params(tail::AsymGalambosTail) = (tail.α, tail.θ[1], tail.θ[2])

function A(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α  = tail.α
    θ1, θ2 = tail.θ

    α == 0 || (θ1 == 0 && θ2 == 0) && return one(tt)
    x1 = -α * log(θ1 * tt)
    x2 = -α * log(θ2 * (1 - tt))
    s  = LogExpFunctions.logaddexp(x1, x2) / α
    return -LogExpFunctions.expm1(-s)
end
