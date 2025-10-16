"""
    AsymGalambosTail{T}, AsymGalambosCopula{T}

Fields:
  - α::Real          — dependence parameter
  - θ₁::Real — asymmetry weight in [0,1]
  - θ₂::Real — asymmetry weight in [0,1]

Constructor

    AsymGalambosCopula(α, θ₁, θ₂)
    ExtremeValueCopula(2, AsymGalambosTail(α, θ₁, θ₂))

The (bivariate) asymmetric Galambos extreme-value copula is parameterized by
``\\alpha \\in [0, \\infty)`` and ``\\theta_1, \\theta_2 \\in [0,1]``. It is an EV copula with Pickands function

```math
A(t) = 1 - \\Big((\\theta_1 t)^{-\\alpha} + (\\theta_2(1-t))^{-\\alpha}\\Big)^{-1/\\alpha},\\quad t\\in[0,1].
```

Special cases:

* α = 0 ⇒ IndependentCopula
* θ₁ = θ₂ = 0 ⇒ IndependentCopula
* θ₁ = θ₂ = 1 ⇒ GalambosCopula

References: 

* [Joe1990](@cite) Families of min-stable multivariate exponential and multivariate extreme value distributions. Statist. Probab, 1990.
"""
AsymGalambosTail, AsymGalambosCopula

struct AsymGalambosTail{T} <: Tail2
    α::T                 # α ≥ 0
    θ₁::T
    θ₂::T
    function AsymGalambosTail(α, θ₁, θ₂)

        T = promote_type(Float64, typeof(α), typeof(θ₁), typeof(θ₂))
        θ₁, θ₂, αT = T(θ₁), T(θ₂), T(α)

        (αT ≥ 0) || throw(ArgumentError("α must be ≥ 0"))
        (0 ≤ θ₁ ≤ 1 && 0 ≤ θ₂ ≤ 1) || throw(ArgumentError("each θ[i] must be in [0,1]"))
        αT == 0 || (θ₁ == 0 && θ₂ == 0) && return NoTail()
        θ₁ == 1 && θ₂ == 1 && return GalambosTail(α)
        return new{T}(αT, θ₁, θ₂)
    end
end

const AsymGalambosCopula{T} = ExtremeValueCopula{2, AsymGalambosTail{T}}
Distributions.params(tail::AsymGalambosTail) = (α = tail.α, θ₁ = tail.θ₁, θ₂ = tail.θ₂)
_unbound_params(::Type{<:AsymGalambosTail}, d, θ) = [log(θ.α), log(θ.θ₁) - log1p(-θ.θ₁), log(θ.θ₂) - log1p(-θ.θ₂)] 
_rebound_params(::Type{<:AsymGalambosTail}, d, α) = begin 
    σ(x) = 1 / (1 + exp(-x)) 
    (; α = exp(α[1]), θ₁ = σ(α[2]), θ₂ = σ(α[3])) 
end

function A(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂  = tail.α, tail.θ₁, tail.θ₂

    α == 0 || (θ₁ == 0 && θ₂ == 0) && return one(tt)
    x1 = -α * log(θ₁ * tt)
    x2 = -α * log(θ₂ * (1 - tt))
    s  = LogExpFunctions.logaddexp(x1, x2) / α
    return -LogExpFunctions.expm1(-s)
end