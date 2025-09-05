"""
    BB4Copula{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB4Copula(θ, δ)

The BB4 copula in dimension ``d = 2`` is parameterized by ``\\theta \\in [0,\\infty)`` and ``\\delta \\in (0,\\infty). It is an Archimedean copula with generator :

```math
\\phi(t) = \\exp(-[\\delta^{-1}\\log(1 + t)]^{\\frac{1}{\\theta}}),
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.197-198
"""
struct BB4Copula{T} <: Copula{2}
    θ::T
    δ::T
    function BB4Copula(θ, δ)
        (θ ≥ 0) || throw(ArgumentError("θ must be ≥ 0"))
        (δ > 0) || throw(ArgumentError("δ must be > 0"))
        new{typeof(θ)}(θ, δ)
    end
end

Distributions.params(C::BB4Copula) = (C.θ, C.δ)

# ---------- CDF ----------
function _cdf(C::BB4Copula, u)
    θ, δ = C.θ, C.δ
    u1, u2 = u
    if θ == 0
        return u1*u2
    end
    uθ = exp(-θ*log(u1))
    vθ = exp(-θ*log(u2))
    a  = expm1(-θ*log(u1))              # = u1^{-θ} - 1  ≥ 0
    b  = expm1(-θ*log(u2))              # = u2^{-θ} - 1  ≥ 0
    x  = a^(-δ)
    y  = b^(-δ)
    s  = (x + y)^(-1/δ)
    T  = uθ + vθ - 1 - s                # [1 + a + b - (x+y)^{-1/δ}]
    return T^(-1/θ)
end

# ---------- log-PDF ----------
function Distributions._logpdf(C::BB4Copula, u)
    Tret = promote_type(Float64, eltype(u))
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return Tret(-Inf)

    θ, δ = C.θ, C.δ
    if θ == 0
        return Tret(log(u1) + log(u2))
    end

    uθ = exp(-θ*log(u1))
    vθ = exp(-θ*log(u2))
    a  = expm1(-θ*log(u1))              # u1^{-θ} - 1
    b  = expm1(-θ*log(u2))              # u2^{-θ} - 1
    x  = a^(-δ)
    y  = b^(-δ)
    S  = x + y
    sS = S^(-1/δ)                       # (x+y)^{-1/δ}
    Tm = uθ + vθ - 1 - sS
    (Tm > 0) || return Tret(-Inf)

    invδ = inv(δ)
    log_fac1 = (-1/θ - 2) * log(Tm)

    log_fac2 = (1 + invδ) * (log(x) + log(y)) + (-θ - 1) * (log(u1) + log(u2))

    invx, invy, invS = inv(x), inv(y), inv(S)
    p = a*invx - sS*invS
    q = b*invy - sS*invS
    term1 = (θ + 1) * p * q
    term2 = θ * (1 + δ) * (Tm) * S^(-invδ - 2)
    bracket = term1 + term2
    (bracket > 0) || return Tret(-Inf)

    logc = log_fac1 + log_fac2 + log(bracket)
    return Tret(logc)
end

archimax_view(C::BB4Copula) = ArchimaxCopula(ClaytonCopula(2, C.θ), GalambosCopula(C.δ))

function Distributions._rand!(rng::Distributions.AbstractRNG, C::BB4Copula, x::AbstractVector{T}) where {T<:Real}
    return Distributions._rand!(rng, archimax_view(C), x)
end

τ(C::BB4Copula) = τ(archimax_view(C))