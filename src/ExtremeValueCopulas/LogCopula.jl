"""
    LogCopula{P}

Fields:

    - θ::Real - parameter
    
Constructor

    LogCopula(θ)

The bivariate Logistic copula (or Gumbel Copula) is parameterized by ``\\theta \\in [1,\\infty)``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = (t^{\\theta}+(1-t)^{\\theta})^{\\frac{1}{\\theta}}
```

It has a few special cases: 
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)

References:
* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct LogCopula{P} <: ExtremeValueCopula{P}
    θ::P  # Copula parameter
    function LogCopula(θ)
        if !(1 <= θ)
            throw(ArgumentError(" The param θ must be in [1, ∞)"))
        elseif θ == 1
            return IndependentCopula(2)
        elseif θ == Inf
            return MCopula(2)
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
Distributions.params(C::LogCopula) = (C.θ)
# #  specific ℓ funcion of LogCopula
function ℓ(G::LogCopula, t₁, t₂)
    θ = G.θ
    return (t₁^θ + t₂^θ)^(1/θ)
end
# # #  specific A funcion of LogCopula
# A(C::LogCopula, t::Real) = exp(LogExpFunctions.logaddexp(C.θ*log(t),C.θ*log(1-t))/C.θ) 

# A(t) pour la LogCopula (avec log-exp pour la stabilité)
function A(C::LogCopula, t::Real)
    θ = C.θ
    # log-sum-exp trick: log(t^θ + (1-t)^θ) = logsumexp(θ*log(t), θ*log1p(-t))
    logB = LogExpFunctions.logaddexp(θ*log(t), θ*log1p(-t))
    return exp(logB / θ)
end

# Première dérivée dA/dt (stable numériquement)
function dA(C::LogCopula, t::Real)
    θ = C.θ

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
function d2A(C::LogCopula, t::Real)
    θ = C.θ

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