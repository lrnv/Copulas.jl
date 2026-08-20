"""
    LogTail{T}, LogCopula{d,T}

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
    x = θ * (log(z) - log1p(-z))
    return inv(one(x) + exp(-x))
end

function Distributions.logpdf(d::ExtremeDist{<:LogTail}, z::Real)
    (z <= zero(z) || z >= one(z)) && return oftype(float(z), -Inf)
    θ = d.tail.θ
    logS = LogExpFunctions.logaddexp(θ * log(z), θ * log1p(-z))
    return log(θ) + (θ - 1) * (log(z) + log1p(-z)) - 2 * logS
end
_pdf(d::ExtremeDist{<:LogTail}, z::Real) = exp(Distributions.logpdf(d, z))

function Distributions.quantile(d::ExtremeDist{<:LogTail}, p::Real)
    T = float(promote_type(typeof(p), typeof(d.tail.θ)))
    zero(T) <= p <= one(T) || throw(ArgumentError("p must be between 0 and 1"))
    p == zero(T) && return zero(T)
    p == one(T) && return one(T)
    x = (log(T(p)) - log1p(-T(p))) / T(d.tail.θ)
    return inv(one(T) + exp(-x))
end

_probability_z(tail::LogTail, ::Real) = (tail.θ - one(tail.θ)) / tail.θ

# The bivariate logistic EV copula is exactly Gumbel.
Distributions._logpdf(C::ExtremeValueCopula{2,<:LogTail}, u) =
    Distributions._logpdf(GumbelCopula(2, C.tail.θ), u)

function _ellpartial_signlog(tail::LogTail, x, I)
    k = length(I)
    θ = tail.θ
    logabs = (one(θ) - k * θ) * log(ℓ(tail, x))
    logabs += (θ - one(θ)) * sum(log(x[i]) for i in I)
    k > 1 && (logabs += sum(log(j * θ - one(θ)) for j in 1:k-1))
    return isodd(k - 1) ? -1 : 1, logabs
end

function ellpartial(tail::LogTail, x, I::Tuple{Vararg{Int}})
    isempty(I) && return ℓ(tail, x)
    sgn, logabs = _ellpartial_signlog(tail, x, I)
    return sgn * exp(logabs)
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

_rand_ev_multivariate!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:LogTail}, X::AbstractMatrix{T}) where {T<:Real,d} =
    Distributions._rand!(rng, GumbelCopula(d, C.tail.θ), X)

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
