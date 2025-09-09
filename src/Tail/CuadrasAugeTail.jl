"""
    CuadrasAugeTail{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    CuadrasAugeCopula(θ)
    ExtremeValueCopula(CuadrasAugeTail(θ))

The (bivariate) Cuadras-Augé extreme-value copula is parameterized by ``\\theta \\in [0,1]``.
Its Pickands dependence function is

```math
A(t) = \\max\\{t, 1-t\\} + (1-\\theta)\\min\\{t,1-t\\}, \\quad t \\in [0,1].
```

Special cases:

* θ = 0 ⇒ IndependentCopula
* θ = 1 ⇒ MCopula (comonotone copula)

References:

* Mai & Scherer (2012). Simulating copulas: stochastic models, sampling algorithms, and applications. World Scientific.
  """
struct CuadrasAugeTail{T} <: Tail{2}
    θ::T
    function CuadrasAugeTail(θ)
        (0 ≤ θ ≤ 1) || throw(ArgumentError("θ must be in [0,1]"))
        T = promote_type(typeof(θ))
        new{T}(T(θ))
    end
end
const CuadrasAugeCopula{T} = ExtremeValueCopula{2, CuadrasAugeTail{T}}
Distributions.params(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}) where {T} = (C.E.θ,)

function A(E::CuadrasAugeTail, t::Real)
    tt = _safett(t)
    θ = E.θ
    return max(tt, 1-tt) + (1-θ) * min(tt, 1-tt)
end

function CuadrasAugeCopula(θ)
    if θ == 0
        return IndependentCopula(2)
    elseif θ == 1
        return MCopula(2)
    else
        return ExtremeValueCopula(CuadrasAugeTail(θ))
    end
end

dA(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}, t::Real) where {T} = (t <= 0.5 ? -C.E.θ : C.E.θ)

τ(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}) where {T} = C.E.θ / (2 - C.E.θ)
ρ(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}) where {T} = (3 * C.E.θ) / (4 - C.E.θ)

ℓ(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}, t₁, t₂) where {T} = max(t₁, t₂) + (1 - C.E.θ) * min(t₁, t₂)

function Distributions._rand!(rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{2, CuadrasAugeTail{T}},
    x::AbstractVector{S}) where {T,S<:Real}
    θ = C.E.θ
    E₁, E₂ = rand(rng, Distributions.Exponential(θ/(1-θ)), 2)
    E₁₂ = rand(rng, Distributions.Exponential())
    x[1] = exp(-(1/θ) * min(E₁, E₁₂))
    x[2] = exp(-(1/θ) * min(E₂, E₁₂))
    return x
end

function Distributions.logcdf(D::BivEVDistortion{<:ExtremeValueCopula{2,CuadrasAugeTail{T}}, S}, z::Real) where {T,S}
    θ = D.C.E.θ
    # bounds and degeneracies
    z ≤ 0    && return S(-Inf)
    z ≥ 1    && return S(0)
    D.uⱼ ≤ 0 && return S(-Inf)
    D.uⱼ ≥ 1 && return S(log(z))

    z ≥ D.uⱼ && return (1-θ) * log(z)
    return log1p(-θ) + log(z) - θ * log(D.uⱼ)

end

function Distributions.quantile(D::BivEVDistortion{<:ExtremeValueCopula{2,CuadrasAugeTail{T}}}, α::Real) where {T}
    θ = D.C.E.θ
    α ≤ 0 && return 0.0
    α ≥ 1 && return 1.0
    D.uⱼ ≤ 0 && return 0.0
    D.uⱼ ≥ 1 && return α

    la = log(α)
    lu = log(D.uⱼ)
    lt = log1p(-θ)

    if la < lt + (1-θ)*lu
        return exp(la - lt + θ*lu)
    elseif la ≤ (1-θ)*lu
        return D.uⱼ
    else
        return exp(la / (1 - θ))
    end
end