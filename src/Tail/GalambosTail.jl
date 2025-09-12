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
struct GalambosTail{T} <: Tail2
    θ::T
    function GalambosTail(θ)
        θ < 0 && throw(ArgumentError("θ must be ≥ 0"))
        θ == 0 && return NoTail()
        isinf(θ) && return MTail()
        new{typeof(float(θ))}(float(θ))
    end
end

const GalambosCopula{T} = ExtremeValueCopula{2, GalambosTail{T}}
GalambosCopula(θ) =ExtremeValueCopula(2, GalambosTail(θ))
Distributions.params(tail::GalambosTail) = (tail.θ,)
_is_valid_in_dim(::GalambosTail, d::Int) = (d >= 2)

function A(tail::GalambosTail, ω::NTuple{d,<:Real}) where {d}
    θ = tail.θ
    θ == 0 && return 1.0
    isinf(θ) && return maximum(ω)
    return -LogExpFunctions.expm1(-LogExpFunctions.logsumexp(-θ .* log.(ω))/θ)  # 1 - (∑ ω_i^{-θ})^{-1/θ}
end

#### Special bindings for dimension d == 2
needs_binary_search(tail::GalambosTail) = (tail.θ > 19.5)
function A(tail::GalambosTail, t::Real)
    tt = _safett(t)
    tail.θ == 0 && return 1.0
    isinf(tail.θ) && return max(tt, 1-tt)
    return -LogExpFunctions.expm1(-LogExpFunctions.logaddexp(-tail.θ*log(tt), -tail.θ*log(1-tt)) / tail.θ)
end