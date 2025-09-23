"""
    MixedTail{T}

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
struct MixedTail{T} <: AbstractUnivariateTail2
    θ::T
    function MixedTail(θ)
        (0 ≤ θ ≤ 1) || throw(ArgumentError("θ must be in [0,1]"))
        θ == 0 && return NoTail()
        return new{typeof(θ)}(θ)
    end
end

const MixedCopula{T} = ExtremeValueCopula{2, MixedTail{T}}
MixedCopula(θ) = ExtremeValueCopula(2, MixedTail(θ))
MixedCopula(d::Integer, θ) = ExtremeValueCopula(2, MixedTail(θ))
Distributions.params(tail::MixedTail) = (θ = tail.θ,)
_θ_bounds(::Type{<:MixedTail}, d) = (0, 1)
A(tail::MixedTail, t::Real) = tail.θ * t^2 - tail.θ * t + 1
# Fitting helpers for EV copulas using Mixed tail (θ ∈ [0,1])
_example(::Type{<:MixedCopula}, d) = ExtremeValueCopula(2, MixedTail(0.5))
_unbound_params(::Type{<:MixedCopula}, d, θ) = [log(θ.θ) - log1p(-θ.θ)]
_rebound_params(::Type{<:MixedCopula}, d, α) = begin
    p = 1 / (1 + exp(-α[1]))
    (; θ = p)
end
τ(C::MixedCopula{T}) where {T} = 8 / sqrt(C.tail.θ * (4 - C.tail.θ)) * atan( sqrt(C.tail.θ / (4 - C.tail.θ)) ) - 2
ρ(C::MixedCopula{T}) where {T} = -3 + 12/(8 - C.tail.θ) + 96 * atan(sqrt(C.tail.θ/(8 - C.tail.θ))) / (sqrt(C.tail.θ) * (8 - C.tail.θ)^(3/2))
β(C::MixedCopula{T}) where {T} = 2.0^(C.tail.θ / 2) - 1
β⁻¹(C::MixedCopula{T}, beta::Real) where {T} = 2 * log2(beta + 1)
λᵤ(C::MixedCopula{T}) where {T} = C.tail.θ / 2
λᵤ⁻¹(C::MixedCopula{T}, λ::Real) where {T} = 2λ

tau_Mixed(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : QuadGK.quadgk(t -> d²A(MixedTail(θ),t)*t*(1-t)/max(A(MixedTail(θ),t),_δ(t)), 0, 1; kw...)[1]
rho_Mixed(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12*QuadGK.quadgk(t -> inv(1+A(MixedTail(θ),t))^2, 0, 1; kw...)[1] - 3

iTau(::Type{MixedCopula}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? θmax : _invmono(θ -> tau_Mixed(θ) - τ; kw...)
iRho(::Type{MixedCopula}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? θmax : _invmono(θ -> rho_Mixed(θ) - ρ; kw...)

τ⁻¹(C::MixedCopula, τ; kw...) = iTau(MixedCopula, τ; kw...)
ρ⁻¹(C::MixedCopula, ρ; kw...) = iRho(MixedCopula, ρ; kw...)

