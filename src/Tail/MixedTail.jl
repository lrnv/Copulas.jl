"""
    MixedTail{T}, MixedCopula{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    MixedCopula(θ)
    ExtremeValueCopula(2, MixedTail(θ))

The (bivariate) Mixed extreme-value copula is parameterized by ``\\theta \\in [0,1]``.
Its Pickands dependence function is

```math
A(t) = \\theta t^2 - \\theta t + 1, \\quad t \\in [0,1].
```

Special cases:

* θ = 0 ⇒ IndependentCopula

References:

* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
MixedTail, MixedCopula

struct MixedTail{T} <: AbstractUnivariateTail2
    θ::T
    function MixedTail(θ)
        (0 ≤ θ ≤ 1+eps(θ)) || throw(ArgumentError("θ must be in [0,1], provided θ=$θ"))
        θ = clamp(θ, 0, 1)
        θ == 0 && return NoTail()
        return new{typeof(θ)}(θ)
    end
end

const MixedCopula{T} = ExtremeValueCopula{2, MixedTail{T}}
Distributions.params(tail::MixedTail) = (θ = tail.θ,)
_unbound_params(::Type{<:MixedTail}, d, θ) = [log(θ.θ) - log1p(-θ.θ)]
_rebound_params(::Type{<:MixedTail}, d, α) = begin
    θ = 1 / (1 + exp(-α[1]))
    return (; θ)
end
_θ_bounds(::Type{<:MixedTail}, d) = (0, 1)

A(tail::MixedTail, t::Real) = tail.θ * t^2 - tail.θ * t + 1

_tau_Mixed(θ; kw...) = θ ≤ 0 ? 0.0 : θ ≥ 1 ? 1.0 : 1 + 4 * QuadGK.quadgk(t -> ((2θ*t - θ) / (θ*t^2 - θ*t + 1)) * t * (1-t), 0, 1; kw...)[1]
_rho_Mixed(θ; kw...) = θ ≤ 0 ? 0.0 : θ ≥ 1 ? 1.0 : 12 * QuadGK.quadgk(t -> inv((θ*t^2 - θ*t + 1 + 1)^2), 0, 1; kw...)[1] - 3

τ(C::MixedCopula) = 8 / sqrt(C.tail.θ * (4 - C.tail.θ)) * atan( sqrt(C.tail.θ / (4 - C.tail.θ)) ) - 2
ρ(C::MixedCopula) = -3 + 12/(8 - C.tail.θ) + 96 * atan(sqrt(C.tail.θ/(8 - C.tail.θ))) / (sqrt(C.tail.θ) * (8 - C.tail.θ)^(3/2))
β(C::MixedCopula) = 2.0^(C.tail.θ / 2) - 1
λᵤ(C::MixedCopula) = 0

τ⁻¹(::Type{<:MixedCopula}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? 1 : _invmono(θ -> _tau_Mixed(θ) - τ; kw...)
ρ⁻¹(::Type{<:MixedCopula}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? 1 : _invmono(θ -> _rho_Mixed(θ) - ρ; kw...)
β⁻¹(::Type{<:MixedCopula}, beta) = 2 * log2(beta + 1)
λᵤ⁻¹(::Type{<:MixedCopula}, λ) = 2λ

