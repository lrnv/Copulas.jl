"""
    AsymLogTail{T}

Fields:
  - α::Real            — dependence parameter (α ≥ 1)
  - θ::NTuple{2,Real}  — asymmetry weights, each in [0,1]

Constructor

    AsymLogCopula(α, θ::AbstractVector)
    ExtremeValueCopula(AsymLogTail(α, θ))

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
    θ::NTuple{2,T}
    function AsymLogTail(α, θ)
        (length(θ) == 2) || throw(ArgumentError("θ must have length 2"))
        T = promote_type(typeof(α), eltype(θ))
        αT = T(α)
        θT = (T(θ[1]), T(θ[2]))
        (αT ≥ 1) || throw(ArgumentError("α must be ≥ 1"))
        (0 ≤ θT[1] ≤ 1 && 0 ≤ θT[2] ≤ 1) ||
        throw(ArgumentError("each θ[i] must be in [0,1]"))
        new{T}(αT, θT)
    end
end

const AsymLogCopula{T} = ExtremeValueCopula{2, AsymLogTail{T}}
AsymLogCopula(α, θ) =  ExtremeValueCopula(AsymLogTail(α, θ))
Distributions.params(tail::AsymLogTail) = (tail.α, tail.θ[1], tail.θ[2])

function A(E::AsymLogTail, t::Real)
    tt = _safett(t)
    α  = E.α
    θ1, θ2 = E.θ
    return ((θ1^α) * (1-tt)^α + (θ2^α) * tt^α)^(1/α) + (θ1 - θ2)*tt + 1 - θ1
end
