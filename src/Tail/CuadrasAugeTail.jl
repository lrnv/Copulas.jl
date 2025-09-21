"""
    CuadrasAugeTail{T}

Fields:
  - θ::Real — dependence parameter, θ ∈ [0,1]

Constructor

    CuadrasAugeCopula(θ)
    ExtremeValueCopula(2, CuadrasAugeTail(θ))

The (bivariate) Cuadras-Augé extreme-value copula is parameterized by ``\\theta \\in [0,1]``.
Its Pickands dependence function is

```math
A(t) = \\max\\{t, 1-t\\} + (1-\\theta)\\min\\{t,1-t\\}, \\quad t \\in [0,1].
```

Special cases:

* θ = 0 ⇒ IndependentCopula
* θ = 1 ⇒ MCopula (comonotone copula)

References:

* [mai2012simulating](@cite) Mai, J. F., & Scherer, M. (2012). Simulating copulas: stochastic models, sampling algorithms, and applications (Vol. 4). World Scientific.
"""
struct CuadrasAugeTail{T} <: Tail2
    θ::T
    function CuadrasAugeTail(θ)
        (0 ≤ θ ≤ 1) || throw(ArgumentError("θ must be in [0,1]"))
        θ == 0 && return NoTail()
        θ == 1 && return MTail() 
        θf = float(θ)
        new{typeof(θf)}(θf)
    end
end

const CuadrasAugeCopula{T} = ExtremeValueCopula{2, CuadrasAugeTail{T}}
CuadrasAugeCopula(θ) = ExtremeValueCopula(2, CuadrasAugeTail(θ))
CuadrasAugeCopula(d::Integer, θ) = ExtremeValueCopula(2, CuadrasAugeTail(θ))
Distributions.params(tail::CuadrasAugeTail) = (θ = tail.θ,)

function A(tail::CuadrasAugeTail, t::Real)
    tt = _safett(t)
    θ = tail.θ
    return max(tt, 1-tt) + (1-θ) * min(tt, 1-tt)
end

# Fitting helpers for EV copulas using Cuadras–Augé tail (θ ∈ [0,1])
_example(::Type{ExtremeValueCopula{2, CuadrasAugeTail{T}}}, d) where {T} = ExtremeValueCopula(2, CuadrasAugeTail(T(0.5)))
_example(::Type{ExtremeValueCopula{2, CuadrasAugeTail}}, d) = ExtremeValueCopula(2, CuadrasAugeTail(0.5))
_unbound_params(::Type{ExtremeValueCopula{2, CuadrasAugeTail}}, d, θ) = [log(θ.θ) - log1p(-θ.θ)]
_rebound_params(::Type{ExtremeValueCopula{2, CuadrasAugeTail}}, d, α) = begin
    p = 1 / (1 + exp(-α[1]))
    (; θ = p)
end

dA(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}, t::Real) where {T} = (t <= 0.5 ? -tail.θ : C.tail.θ)

τ(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}) where {T} = C.tail.θ / (2 - C.tail.θ)
ρ(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}) where {T} = (3 * C.tail.θ) / (4 - C.tail.θ)

ℓ(C::ExtremeValueCopula{2, CuadrasAugeTail{T}}, t) where {T} = max(t[1], t[2]) + (1 - C.tail.θ) * min(t[1], t[2])

function Distributions._rand!(rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{2, CuadrasAugeTail{T}},
    x::AbstractVector{S}) where {T,S<:Real}
    θ = C.tail.θ
    E₁, E₂ = rand(rng, Distributions.Exponential(θ/(1-θ)), 2)
    E₁₂ = rand(rng, Distributions.Exponential())
    x[1] = exp(-(1/θ) * min(E₁, E₁₂))
    x[2] = exp(-(1/θ) * min(E₂, E₁₂))
    return x
end

function Distributions.logcdf(D::BivEVDistortion{CuadrasAugeTail{T}, S}, z::Real) where {T,S}
    θ = D.tail.θ
    # bounds and degeneracies
    z ≤ 0    && return S(-Inf)
    z ≥ 1    && return S(0)
    D.uⱼ ≤ 0 && return S(-Inf)
    D.uⱼ ≥ 1 && return S(log(z))

    z ≥ D.uⱼ && return (1-θ) * log(z)
    return log1p(-θ) + log(z) - θ * log(D.uⱼ)

end

function Distributions.quantile(D::BivEVDistortion{CuadrasAugeTail{T}, S}, α::Real) where {T, S}
    θ = D.tail.θ
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