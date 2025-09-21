"""
    LogTail{T}

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
struct LogTail{T} <: Tail2
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
LogCopula(θ) = ExtremeValueCopula(2, LogTail(θ))
LogCopula(d::Integer, θ) = ExtremeValueCopula(2, LogTail(θ))
Distributions.params(tail::LogTail) = (θ = tail.θ,)

function ℓ(tail::LogTail, t)
    t₁, t₂ = t
    θ = tail.θ
    return (t₁^θ + t₂^θ)^(1/θ)
end

# Fitting helpers for EV copulas using Log tail
_example(::Type{ExtremeValueCopula{2, LogTail{T}}}, d) where {T} = ExtremeValueCopula(2, LogTail(one(T)+one(T)))
_example(::Type{ExtremeValueCopula{2, LogTail}}, d) = ExtremeValueCopula(2, LogTail(2.0))
_unbound_params(::Type{ExtremeValueCopula{2, LogTail}}, d, θ) = [log(θ.θ - 1)]       # θ ≥ 1
_rebound_params(::Type{ExtremeValueCopula{2, LogTail}}, d, α) = (; θ = exp(α[1]) + 1)

# A(t) for LogCopula (avec log-exp pour la stabilité)
function A(tail::LogTail, t::Real)
    θ = tail.θ
    # log-sum-exp trick: log(t^θ + (1-t)^θ) = logsumexp(θ*log(t), θ*log1p(-t))
    logB = LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t))
    return exp(logB / θ)
end

# Première dérivée dA/dt (stable numériquement)
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

# Seconde dérivée d²A/dt² (stable numériquement)
function d2A(tail::LogTail, t::Real)
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