"""
    AsymLogTail{T}

Fields:
  - α::Real  — dependence parameter (α ≥ 1)
  - θ₁::Real — asymmetry weight in [0,1]
  - θ₂::Real — asymmetry weight in [0,1]

Constructor

    AsymLogCopula(α, (θ₁, θ₂))
    ExtremeValueCopula(2, AsymLogTail(α, (θ₁, θ₂)))

The (bivariate) asymmetric logistic extreme–value copula is parameterized by
α ∈ [1, ∞) and θ₁, θ₂ ∈ [0,1]. Its Pickands dependence function is

```math
A(t) = \\Big( \\theta_1^{\\alpha}(1-t)^{\\alpha} + \\theta_2^{\\alpha}t^{\\alpha} \\Big)^{1/\\alpha}
       + (\\theta_1 - \\theta_2)\\,t + 1 - \\theta_1, \\quad t\\in[0,1].
```

Special cases:

* θ₁ = θ₂ = 1 ⇒ symmetric Logistic (Gumbel) copula
* θ₁ = θ₂ = 0 ⇒ independence (A(t) ≡ 1)

References:

* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct AsymLogTail{T} <: Tail2
    α::T
    θ₁::T
    θ₂::T
    function AsymLogTail(α, θ)
        (length(θ) == 2) || throw(ArgumentError("θ must have length 2"))
        T = promote_type(typeof(α), eltype(θ))
        αT = T(α)
        θ₁, θ₂ = T(θ[1]), T(θ[2])
        (αT ≥ 1) || throw(ArgumentError("α must be ≥ 1"))
        (0 ≤ θ₁ ≤ 1 && 0 ≤ θ₂ ≤ 1) || throw(ArgumentError("each θ[i] must be in [0,1]"))
        new{T}(αT, θ₁, θ₂)
    end
end

const AsymLogCopula{T} = ExtremeValueCopula{2, AsymLogTail{T}}
AsymLogCopula(α, θ) =  ExtremeValueCopula(2, AsymLogTail(α, θ))
Distributions.params(tail::AsymLogTail) = (tail.α, tail.θ₁, tail.θ₂)

function A(tail::AsymLogTail, t::Real)
    tt = _safett(t)
    α  = tail.α
    θ₁, θ₂ = tail.θ₁, tail.θ₂
    return ((θ₁^α) * (1-tt)^α + (θ₂^α) * tt^α)^(1/α) + (θ₁ - θ₂)*tt + 1 - θ₁
end
