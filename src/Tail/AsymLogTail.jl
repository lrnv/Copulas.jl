"""
    AsymLogTail{T}, AsymLogCopula{T}

Fields:
  - α::Real  — dependence parameter (α ≥ 1)
  - θ₁::Real — asymmetry weight in [0,1]
  - θ₂::Real — asymmetry weight in [0,1]

Constructor

    AsymLogCopula(α, θ₁, θ₂)
    ExtremeValueCopula(2, AsymLogTail(α, θ₁, θ₂))

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
AsymLogTail, AsymLogCopula

struct AsymLogTail{T} <: Tail2
    α::T
    θ₁::T
    θ₂::T
    function AsymLogTail(α, θ₁, θ₂)
        T = promote_type(Float64, typeof(α), typeof(θ₁), typeof(θ₂))
        θ₁, θ₂, αT = T(θ₁), T(θ₂), T(α)
        (αT ≥ 1) || throw(ArgumentError("α must be ≥ 1"))
        (0 ≤ θ₁ ≤ 1 && 0 ≤ θ₂ ≤ 1) || throw(ArgumentError("each θ[i] must be in [0,1]"))
        new{T}(αT, θ₁, θ₂)
    end
end

const AsymLogCopula{T} = ExtremeValueCopula{2, AsymLogTail{T}}
Distributions.params(tail::AsymLogTail) = (α = tail.α, θ₁ = tail.θ₁, θ₂ = tail.θ₂)
_unbound_params(::Type{<:AsymLogTail}, d, θ) = [log(θ.α - 1), log(θ.θ₁) - log1p(-θ.θ₁), log(θ.θ₂) - log1p(-θ.θ₂)]
_rebound_params(::Type{<:AsymLogTail}, d, α) = begin
    σ(x) = 1 / (1 + exp(-x))
    (; α = exp(α[1]) + 1, θ₁ = σ(α[2]), θ₂ = σ(α[3]))
end

function A(tail::AsymLogTail, t::Real)
    tt = _safett(t)
    α  = tail.α
    θ₁, θ₂ = tail.θ₁, tail.θ₂
    return ((θ₁^α) * (1-tt)^α + (θ₂^α) * tt^α)^(1/α) + (θ₁ - θ₂)*tt + 1 - θ₁
end

function dA(tail::AsymLogTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = tail.α, tail.θ₁, tail.θ₂

    (θ₁ == 0 && θ₂ == 0) && return zero(tt)

    a = tt
    b = 1 - tt

    c1 = θ₁^α
    c2 = θ₂^α

    R = c1 * b^α + c2 * a^α
    D = c2 * a^(α - 1) - c1 * b^(α - 1)

    return R^(inv(α) - 1) * D + θ₁ - θ₂
end

function d²A(tail::AsymLogTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = tail.α, tail.θ₁, tail.θ₂

    (θ₁ == 0 && θ₂ == 0) && return zero(tt)
    α == 1 && return zero(tt)

    a = tt
    b = 1 - tt

    c1 = θ₁^α
    c2 = θ₂^α

    R = c1 * b^α + c2 * a^α
    D = c2 * a^(α - 1) - c1 * b^(α - 1)
    E = c2 * a^(α - 2) + c1 * b^(α - 2)

    return (α - 1) * R^(inv(α) - 2) * (R * E - D^2)
end