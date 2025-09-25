"""
    GalambosTail{T}

Fields:
  - θ::Real — dependence parameter, θ ≥ 0

Constructor

    GalambosCopula(θ)
    ExtremeValueCopula(2, GalambosTail(θ))

The (bivariate) Galambos extreme-value copula is parameterized by ``\\theta \\in [0, \\infty)``.
Its Pickands dependence function is

```math
A(t) = 1 - \\Big( t^{-\\theta} + (1-t)^{-\\theta} \\Big)^{-1/\\theta}, \\quad t \\in (0,1).
```

Special cases:

* θ = 0   ⇒ IndependentCopula
* θ = ∞   ⇒ MCopula (upper Fréchet-Hoeffding bound)

References:

* [galambos1975order](@cite) Galambos, J. (1975). Order statistics of samples from multivariate distributions. Journal of the American Statistical Association, 70(351a), 674-680.
"""
struct GalambosTail{T} <: AbstractUnivariateTail2
    θ::T
    function GalambosTail(θ)
        θ < 0 && throw(ArgumentError("θ must be ≥ 0"))
        θ == 0 && return NoTail()
        isinf(θ) && return MTail()
        new{typeof(float(θ))}(float(θ))
    end
end

const GalambosCopula{T} = ExtremeValueCopula{2, GalambosTail{T}}
GalambosCopula(θ) = ExtremeValueCopula(2, GalambosTail(θ))
GalambosCopula(d::Integer, θ) = ExtremeValueCopula(2, GalambosTail(θ))
Distributions.params(tail::GalambosTail) = (θ = tail.θ,)
_θ_bounds(::Type{<:GalambosTail}, d) = (0, Inf)
needs_binary_search(tail::GalambosTail) = (tail.θ > 19.5)
function A(tail::GalambosTail, t::Real)
    tt = _safett(t)
    θ  = tail.θ
    if θ == 0
        return 1.0
    elseif isinf(θ)
        return max(tt, 1-tt)
    else
        return -LogExpFunctions.expm1(-LogExpFunctions.logaddexp(-θ*log(tt), -θ*log(1-tt)) / θ)
    end
end

# Fitting helpers for EV copulas using Galambos tail
_example(::Type{<:GalambosCopula}, d) = ExtremeValueCopula(2, GalambosTail(1.0))
_example(::Type{ExtremeValueCopula{2, GalambosTail}}, d) = ExtremeValueCopula(2, GalambosTail(1.0))
_unbound_params(::Type{<:GalambosCopula}, d, θ) = [log(θ.θ)]           # θ > 0
_rebound_params(::Type{<:GalambosCopula}, d, α) = (; θ = exp(α[1]))

@inline function d²A(tail::GalambosTail, t::Real)
    tt = _safett(t)
    θ = tail.θ
    if θ == 0
        return 0.0
    elseif isinf(θ)
        return 0.0
    end
    a = tt
    b = 1 - tt
    L1 = -θ*log(a)
    L2 = -θ*log(b)
    M  = max(L1, L2)
    E1 = exp(L1 - M)
    E2 = exp(L2 - M)
    S  = E1 + E2
    # B = (a^-θ + b^-θ)^(-1/θ) with numerically stable rescaling
    B  = exp(-(M/θ)) * S^(-1/θ)

    inva = inv(a); invb = inv(b)
    D    = E2*invb - E1*inva
    term1 = (E2*invb^2 + E1*inva^2) / S
    term2 = (D/S)^2
    return (1 + θ) * B * (term1 - term2)
end

@inline function dA(tail::GalambosTail, t::Real)
    tt = _safett(t)
    θ = tail.θ
    if θ == 0 || isinf(θ)
        return 0.0
    end
    a = tt
    b = 1 - tt
    L1 = -θ*log(a)
    L2 = -θ*log(b)
    M  = max(L1, L2)
    E1 = exp(L1 - M)
    E2 = exp(L2 - M)
    S  = E1 + E2
    B  = exp(-(M/θ)) * S^(-1/θ)
    inva = inv(a); invb = inv(b)
    D    = E2*invb - E1*inva
    # A'(t) = B * (D/S)
    return B * (D / S)
end

_tau_galambos(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : QuadGK.quadgk(t -> d²A(GalambosTail(θ),t)*t*(1-t)/max(A(GalambosTail(θ),t),_δ(t)), 0, 1; kw...)[1]
_rho_galambos(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12*QuadGK.quadgk(t -> inv(1+A(GalambosTail(θ),t))^2, 0, 1; kw...)[1] - 3


τ(C::GalambosCopula) = _tau_galambos(C.tail.θ)
ρ(C::GalambosCopula) = _rho_galambos(C.tail.θ)
β(C::GalambosCopula) = 2.0^( 2.0^(-1.0/C.tail.θ) ) - 1.0
λᵤ(C::GalambosCopula) = 2.0^(-1.0/C.tail.θ)

τ⁻¹(::Type{<:GalambosCopula}, τ; kw...) = τ ≤ 0 ? 0.0 : τ ≥ 1 ? θmax : _invmono(θ -> _tau_galambos(θ) - τ; kw...)
ρ⁻¹(::Type{<:GalambosCopula}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? θmax : _invmono(θ -> _rho_galambos(θ) - ρ; kw...)
β⁻¹(::Type{<:GalambosCopula}, beta) = -1/log2(log2(beta+1))
λᵤ⁻¹(::Type{<:GalambosCopula}, λ) = -1.0 / log2(λ)