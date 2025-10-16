"""
    LogTail{T}, LogCopula{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    LogCopula(θ)
    ExtremeValueCopula(2, LogTail(θ))

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
LogTail, LogCopula

struct LogTail{T} <: AbstractUnivariateTail2
    θ::T
    function LogTail(θ)
        !(1 <= θ) && throw(ArgumentError(" The param θ must be in [1, ∞)"))
        θ == 1 && return NoTail()
        isinf(θ) && return MTail()
        θ, _ = promote(θ, 1.0)
        return new{typeof(θ)}(θ)
    end
end

const LogCopula{T} = ExtremeValueCopula{2, LogTail{T}}
Distributions.params(tail::LogTail) = (θ = tail.θ,)
_unbound_params(::Type{<:LogTail}, d, θ) = [log(θ.θ - 1)]       # θ ≥ 1
_rebound_params(::Type{<:LogTail}, d, α) = (; θ = exp(α[1]) + 1)
_θ_bounds(::Type{<:LogTail}, d) = (1, Inf)


function ℓ(tail::LogTail, t)
    t₁, t₂ = t
    θ = tail.θ
    return (t₁^θ + t₂^θ)^(1/θ)
end
function A(tail::LogTail, t::Real)
    θ = tail.θ
    # log-sum-exp trick: log(t^θ + (1-t)^θ) = logsumexp(θ*log(t), θ*log1p(-t))
    logB = LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t))
    return exp(logB / θ)
end
function dA(tail::LogTail, t::Real)
    θ = tail.θ

    # B = t^θ + (1-t)^θ
    logB = LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t))
    Bpow = exp((1 - θ) / θ * logB)  # B^((1-θ)/θ)

    # D = t^(θ-1) - (1-t)^(θ-1)
    logt = (θ - 1) * log(t)
    log1mt = (θ - 1) * log1p(-t)
    # carrefull for cancellations
    if logt > log1mt
        D = exp(logt) - exp(log1mt)  # no cancellation here. 
    else
        D = exp(log1mt) * (expm1(logt - log1mt))
    end

    return Bpow * D
end
function d²A(tail::LogTail, t::Real)
    θ = tail.θ

    # B = t^θ + (1-t)^θ
    logB = LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t))
    B = exp(logB)

    # D = t^(θ-1) - (1-t)^(θ-1)
    logt = (θ - 1) * log(t)
    log1mt = (θ - 1) * log1p(-t)
    if logt > log1mt
        D = exp(logt) - exp(log1mt)
    else
        D = exp(log1mt) * (expm1(logt - log1mt))
    end

    # B' = θ*D
    Bp = θ * D

    # E = (θ-1)*(t^(θ-2) + (1-t)^(θ-2))
    logt2 = (θ - 2) * log(t)
    log1mt2 = (θ - 2) * log1p(-t)
    # lets avoid unstable additions
    if logt2 > log1mt2
        E = (θ - 1) * (exp(logt2) + exp(log1mt2))
    else
        E = (θ - 1) * (exp(log1mt2) * (1 + exp(logt2 - log1mt2)))
    end

    term1 = ((1 - θ) / θ) * exp((1 - 2θ) / θ * logB) * Bp * D
    term2 = exp((1 - θ) / θ * logB) * E

    return term1 + term2
end

_rho_Log(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12*QuadGK.quadgk(t -> inv(1+A(LogTail(θ),t))^2, 0, 1; kw...)[1] - 3

τ(C::LogCopula) = 1 - inv(C.tail.θ)
ρ(C::LogCopula) = _rho_Log(C.tail.θ)
β(C::LogCopula) = 4 * 2^(-2^(1 / C.tail.θ)) - 1
λᵤ(C::LogCopula) = 2 - 2^(1 / C.tail.θ)


τ⁻¹(::Type{<:LogCopula}, tau) = 1 / (1 - tau)
ρ⁻¹(::Type{<:LogCopula}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? θmax : _invmono(θ -> _rho_Log(θ) - ρ; a=1.0, b=2.0)
β⁻¹(::Type{<:LogCopula}, beta) = 1 / log2(-log2((beta + 1) / 4))
λᵤ⁻¹(::Type{<:LogCopula}, λ) = 1 / log2(2 - λ)