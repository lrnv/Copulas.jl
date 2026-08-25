"""
    LogTail{T}, LogCopula{d,T}

    LogCopula{d}(θ)
    LogCopula(d, θ)

Logistic (Gumbel-Hougaard) extreme-value copula in dimension `d ≥ 2`, with
`θ ∈ [1, ∞]`. Its stable tail dependence function is

```math
\\ell(x_1,\\ldots,x_d)
=
\\left(\\sum_{i=1}^d x_i^\\theta\\right)^{1/\\theta}.
```

For `d = 2` this is the usual logistic extreme-value model and is equivalent
to `GumbelCopula(2, θ)`. The same mathematical tail is used in every supported
dimension, while dimension two retains specialized analytic kernels.

Special cases:

* `θ = 1` returns `IndependentCopula(d)`.
* `θ = ∞` returns `MCopula(d)`.

References:

* [tawn1988bivariate](@cite) Tawn, J. A. (1988). Bivariate extreme value theory: models and estimation. Biometrika, 75(3), 397-415.
"""
LogTail, LogCopula

struct LogTail{T} <: OneParameterPickandsTail
    θ::T
    function LogTail(θ)
        !(1 <= θ) && throw(ArgumentError(" The param θ must be in [1, ∞)"))
        θ == 1 && return NoTail()
        isinf(θ) && return MTail()
        θ, _ = promote(θ, 1.0)
        return new{typeof(θ)}(θ)
    end
end

const LogCopula{d,T} = ExtremeValueCopula{d, LogTail{T}}
_is_valid_in_dim(::LogTail, d::Int) = d >= 2
Distributions.params(tail::LogTail) = (θ = tail.θ,)
_unbound_params(::Type{<:LogTail}, d, θ) = [log(θ.θ - 1)]       # θ ≥ 1
_rebound_params(::Type{<:LogTail}, d, α) = (; θ = exp(α[1]) + 1)
_θ_bounds(::Type{<:LogTail}, d) = (1, Inf)


function ℓ(tail::LogTail, x)
    m = maximum(x)
    isinf(m) && return m
    iszero(m) && return zero(m * one(tail.θ))
    s = sum((xi / m)^tail.θ for xi in x)
    return m * s^inv(tail.θ)
end
A(tail::LogTail, t::Real) = ℓ(tail, (t, one(t) - t))
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
_pickands_left_slope(::LogTail, prototype::Real) = -one(prototype)
_pickands_right_slope(::LogTail, prototype::Real) = one(prototype)

function d²A(tail::LogTail, t::Real)
    tt = _safett(t)
    θ = tail.θ
    logB = LogExpFunctions.logaddexp(θ * log(tt), θ * log1p(-tt))
    # (θ-1) * [t(1-t)]^(θ-2) * (t^θ + (1-t)^θ)^(1/θ-2)
    logA2 = log(θ - 1) + (θ - 2) * (log(tt) + log1p(-tt)) +
            (inv(θ) - 2) * logB
    return exp(logA2)
end

# Closed forms for the logistic model avoid cancellation in the generic
# Ghoudi auxiliary distribution when θ is large.
function Distributions.cdf(d::ExtremeDist{<:LogTail}, z::Real)
    z <= zero(z) && return zero(float(z))
    z >= one(z) && return one(float(z))
    θ = d.tail.θ
    x = θ * LogExpFunctions.logit(z)
    return inv(one(x) + exp(-x))
end

function Distributions.logpdf(d::ExtremeDist{<:LogTail}, z::Real)
    (z <= zero(z) || z >= one(z)) && return oftype(float(z), -Inf)
    θ = d.tail.θ
    logS = LogExpFunctions.logaddexp(θ * log(z), θ * log1p(-z))
    return log(θ) + (θ - 1) * (log(z) + log1p(-z)) - 2 * logS
end
Distributions.pdf(d::ExtremeDist{<:LogTail}, z::Real) = exp(Distributions.logpdf(d, z))

function Distributions.quantile(d::ExtremeDist{<:LogTail}, p::Real)
    T = float(promote_type(typeof(p), typeof(d.tail.θ)))
    zero(T) <= p <= one(T) || throw(ArgumentError("p must be between 0 and 1"))
    p == zero(T) && return zero(T)
    p == one(T) && return one(T)
    x = LogExpFunctions.logit(T(p)) / T(d.tail.θ)
    return inv(one(T) + exp(-x))
end

_ghoudi_mixture_probability(tail::LogTail, ::Real) = (tail.θ - one(tail.θ)) / tail.θ

# Stable closed-form bivariate density for the logistic model.
#
# For ℓ(x,y) = (x^θ + y^θ)^(1/θ),
#
#   ℓ₁ℓ₂ - ℓ₁₂
#   = x^(θ-1) y^(θ-1) ℓ^(1-2θ) (ℓ + θ - 1).
#
# Evaluating the logarithm of this expression directly avoids the cancellation
# that can affect the generic Pickands derivative kernel under strong
# dependence, while avoiding the overhead of delegating to GumbelCopula.
function Distributions._logpdf(C::ExtremeValueCopula{2,<:LogTail}, u)
    u1, u2 = u
    (zero(u1) < u1 <= one(u1) && zero(u2) < u2 <= one(u2)) ||
        return oftype(float(u1 + u2), -Inf)
    (isone(u1) || isone(u2)) && return oftype(float(u1 + u2), -Inf)

    x, y = -log(u1), -log(u2)
    θ = C.tail.θ
    val = ℓ(C.tail, (x, y))
    oneθ = one(θ)

    return -val + x + y +
           (θ - oneθ) * (log(x) + log(y)) +
           (oneθ - 2θ) * log(val) +
           log(val + θ - oneθ)
end

function _ellpartial_signlog(tail::LogTail, x, I::Tuple{Vararg{Int}})
    k = length(I)
    θ = tail.θ
    logabs = (one(θ) - k * θ) * log(ℓ(tail, x))
    logabs += (θ - one(θ)) * sum(log(x[i]) for i in I)
    k > 1 && (logabs += sum(log(j * θ - one(θ)) for j in 1:k-1))
    return isodd(k - 1) ? -1 : 1, logabs
end


function Distributions.logpdf(D::BivEVDistortion{<:LogTail}, z::Real)
    T = float(promote_type(typeof(z), typeof(D.uⱼ), typeof(D.tail.θ)))
    z <= zero(z) && return T(-Inf)
    z >= one(z) && return T(-Inf)
    D.uⱼ <= zero(D.uⱼ) && return _biv_ev_endpoint_logpdf(D, z, true, T)
    D.uⱼ >= one(D.uⱼ) && return _biv_ev_endpoint_logpdf(D, z, false, T)
    u = D.j == 2 ? (T(z), T(D.uⱼ)) : (T(D.uⱼ), T(z))
    return Distributions._logpdf(GumbelCopula(2, T(D.tail.θ)), u)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:LogTail}, X::AbstractMatrix{T},) where {T<:Real,d}
    size(X, 1) == d || throw(DimensionMismatch("output dimension does not match copula dimension",))
    return Distributions._rand!(rng, GumbelCopula(d, C.tail.θ), X)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2,<:LogTail}, X::AbstractMatrix{T},) where {T<:Real}
    signature = Tuple{
        Distributions.AbstractRNG,
        ExtremeValueCopula{2,<:BivariatePickandsTail},
        AbstractMatrix{T},
    }
    return invoke(Distributions._rand!, signature, rng, C, X)
end

# LogCopula is the bivariate Gumbel copula, so its conditional quantile can use
# the same closed-form inverse of the first generator derivative.
function Distributions.quantile(D::BivEVDistortion{<:LogTail}, α::Real)
    T = float(promote_type(typeof(α), typeof(D.tail.θ), typeof(D.uⱼ)))
    D.uⱼ ≤ 0 && return _biv_ev_endpoint_quantile(D, α, true, T)
    D.uⱼ ≥ 1 && return _biv_ev_endpoint_quantile(D, α, false, T)
    G = GumbelGenerator(T(D.tail.θ))
    sJ = ϕ⁻¹(G, T(D.uⱼ))
    den = ϕ⁽¹⁾(G, sJ)
    return Distributions.quantile(ArchimedeanDistortion(G, 1, sJ, den), T(α))
end

_rho_Log(θ; kw...) = θ == 0 ? 0.0 : !isfinite(θ) ? 1.0 : 12*QuadGK.quadgk(t -> inv(1+A(LogTail(θ),t))^2, 0, 1; kw...)[1] - 3

τ(C::ExtremeValueCopula{2,<:LogTail}) = 1 - inv(C.tail.θ)
ρ(C::ExtremeValueCopula{2,<:LogTail}) = _rho_Log(C.tail.θ)
β(C::ExtremeValueCopula{2,<:LogTail}) = 4 * 2^(-2^(1 / C.tail.θ)) - 1
λᵤ(C::ExtremeValueCopula{2,<:LogTail}) = 2 - 2^(1 / C.tail.θ)


τ⁻¹(::Type{<:ExtremeValueCopula{D,<:LogTail} where D}, tau) = 1 / (1 - tau)
τ⁻¹(::Type{<:LogTail}, tau) = 1 / (1 - tau)
ρ⁻¹(::Type{<:ExtremeValueCopula{D,<:LogTail} where D}, ρ; kw...) = ρ ≤ 0 ? 0.0 : ρ ≥ 1 ? θmax : _invmono(θ -> _rho_Log(θ) - ρ; a=1.0, b=2.0)
β⁻¹(::Type{<:ExtremeValueCopula{D,<:LogTail} where D}, beta) = 1 / log2(-log2((beta + 1) / 4))
λᵤ⁻¹(::Type{<:ExtremeValueCopula{D,<:LogTail} where D}, λ) = 1 / log2(2 - λ)
