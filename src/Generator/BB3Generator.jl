"""
    BB3Generator{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB3Generator(θ, δ)
    BB3Copula(d, θ, δ)

The BB3 copula is parameterized by ``\\theta \\in [1,\\infty)`` and ``\\delta \\in (0,\\infty). It is an Archimedean copula with generator :

```math
\\phi(t) = \\exp(-[\\delta^{-1}\\log(1 + t)]^{\\frac{1}{\\theta}}),
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.195-196
"""
struct BB3Generator{T} <: Generator
    θ::T
    δ::T
    function BB3Generator(θ, δ)
        (θ ≥ 1) || throw(ArgumentError("θ must be ≥ 1"))
        (δ > 0) || throw(ArgumentError("δ must be > 0"))
        new{typeof(θ)}(θ, δ)
    end
end

const BB3Copula{d, T} = ArchimedeanCopula{d, BB3Generator{T}}
BB3Copula(d, θ, δ) = ArchimedeanCopula(d, BB3Generator(θ, δ))
Distributions.params(C::BB3Copula) = (C.G.θ, C.G.δ)
max_monotony(::BB3Generator) = Inf

ϕ(  G::BB3Generator, s) = exp(-exp(log(log1p(s)/G.δ)/G.θ))

ϕ⁻¹(G::BB3Generator, t) = expm1(G.δ * exp(G.θ * log(-log(t))))

function ϕ⁽¹⁾(G::BB3Generator, s)
    a  = inv(G.δ);  pw = inv(G.θ)
    A  = a * log1p(s)
    B = exp((pw-1)*log(A))
    return -(pw*a) * B * inv(1+s) * ϕ(G,s)
end

function ϕ⁽ᵏ⁾(G::BB3Generator, ::Val{2}, s)
    a  = inv(G.δ);  pw = inv(G.θ)
    A  = a * log1p(s);  inv1p = inv(1+s)
    B = exp((pw-1)*log(A))
    C = exp((pw-2)*log(A))
    φ  = ϕ(G,s)
    # K(s) = (pw*a) A^{pw-1} /(1+s);  ψ'' = ψ (K^2 - K')
    K   = (pw*a) * B * inv1p
    K′  = (pw*a) * inv1p^2 * ((pw-1)*a*C - B)
    return φ * (K^2 - K′)
end
ϕ⁻¹⁽¹⁾(G::BB3Generator, t) = -(G.δ*G.θ) * inv(t) * exp(G.δ * exp(G.θ * log(-log(t)))) * (-log(t))^(G.θ - 1)
function _f_for_BB3_ϕ⁽¹⁾⁻¹(lt, a, δ, lny)
    t = exp(lt)
    return (a-1)*lt - δ*t - exp(a*lt) - lny
end
function ϕ⁽ᵏ⁾⁻¹(G::BB3Generator, ::Val{1}, x; start_at=x)
    # compute the inverse of ϕ⁽¹⁾
    θ, δ = G.θ, G.δ
    a = 1/θ
    lny = log(θ)+log(δ)+log(abs(x))
    lt0 = log(log1p(abs(start_at))/δ)
    lt_opt = Roots.find_zero(lt -> _f_for_BB3_ϕ⁽¹⁾⁻¹(lt, a, δ, lny), lt0)
    return expm1(exp(lt_opt)*δ)
end

# Frailty: M = S_{1/δ} * Gamma_{1/θ}^{δ}
williamson_dist(G::BB3Generator, ::Val{d}) where d = WilliamsonFromFrailty(PosStableStoppedGamma(G.θ, G.δ), Val{d}())

@inline function _clip_u_bb3(u::Real, θ::Real, δ::Real)
    # δ(-log u)^θ ≤ LOGMAX - MARGIN = log(floatmax(Float64)) - 8.0
    tmax = (log(floatmax(Float64)) - 8.0) / δ
    ϵθδ  = exp(-exp(log(tmax)*inv(θ)))
    ϵ    = max(1e-12, ϵθδ)
    return clamp(float(u), ϵ, 1 - 1e-12)
end

@inline function _abpair_bb3(u1::Real, u2::Real, θ::Real, δ::Real)
    u1c = _clip_u_bb3(u1, θ, δ)
    u2c = _clip_u_bb3(u2, θ, δ)
    a   = δ * exp(θ*log(-log(u1c)))      # = log(1 + φ⁻¹(u1))  (porque 1+φ⁻¹(u)=e^a)
    b   = δ * exp(θ*log(-log(u2c)))
    return a, b, u1c, u2c
end

# ------------------ stable CDF (d = 2) ------------------
function _cdf(C::ArchimedeanCopula{2,G}, u::AbstractVector{<:Real}) where {G<:BB3Generator}
    u1, u2 = float(u[1]), float(u[2])

    if u1 <= 0 || u2 <= 0
        return 0.0
    elseif u1 >= 1 && u2 >= 1
        return 1.0
    elseif u1 >= 1
        return u2
    elseif u2 >= 1
        return u1
    end

    θ, δ = C.G.θ, C.G.δ
    a, b, _, _ = _abpair_bb3(u1, u2, θ, δ)

    ℓ    = LogExpFunctions.logaddexp(a, b)   # log(e^a+e^b)
    L    = ℓ + LogExpFunctions.log1mexp(-ℓ)  # log(e^a+e^b - 1)
    logA = log(L) - log(δ)                   # A = L/δ

    # C = exp( - A^(1/θ) ) = exp( -exp( (1/θ)*logA ) )
    z = inv(θ) * logA
    if z ≥ log(floatmax(Float64))                          # A^(1/θ) overflows ⇒ C≈0
        return 0.0
    else
        return exp( -exp(z) )
    end
end

# ---------------- stable logpdf (d = 2) -----------------
function Distributions._logpdf(C::ArchimedeanCopula{2,G},
                               u::AbstractVector{<:Real}) where {G<:BB3Generator}
    u1, u2 = float(u[1]), float(u[2])
    (0.0 < u1 < 1.0 && 0.0 < u2 < 1.0) || return -Inf

    θ, δ = C.G.θ, C.G.δ
    a, b, u1c, u2c = _abpair_bb3(u1, u2, θ, δ)

    ℓ    = LogExpFunctions.logaddexp(a, b)
    L    = ℓ + LogExpFunctions.log1mexp(-ℓ)      # = log(e^a + e^b - 1)
    logA = log(L) - log(δ)                       # A = L/δ
    pw   = inv(θ);  acoef = inv(δ)

    logϕinv′u = log(δ*θ) - log(u1c) + (θ-1)*log(-log(u1c)) + a
    logϕinv′v = log(δ*θ) - log(u2c) + (θ-1)*log(-log(u2c)) + b

    T1 = 2*log(pw*acoef) + pw*logA          # log( (pw*acoef)^2 * A^pw )
    T2 = logA                                # log(A)
    T3 = log(1 - pw) + log(acoef)            # log( (1-pw)*acoef )
    log_bracket = LogExpFunctions.logaddexp(T1, LogExpFunctions.logaddexp(T2, T3))

    z = pw*logA
    logφpp = -2L + (pw-2)*logA + log_bracket - exp(z)   # = log φ''

    lp = logφpp + logϕinv′u + logϕinv′v
    return lp < -745 ? -Inf : lp # (optional, for controlled underflow)
end

function τ(G::Copulas.BB3Generator{T}; rtol=1e-8, atol=1e-12) where {T}
    θ = float(G.θ); δ = float(G.δ)

    # f(t) = (t/(δθ)) * expm1(-δ*(-log(t))^θ) / (-log(t))^(θ-1),  t∈(0,1)
    f(t) = (t <= 0.0 || t >= 1.0) ? 0.0 :
           (t / (δ*θ)) * LogExpFunctions.expm1(-δ * (-log(t))^θ) / ((-log(t))^(θ - 1))

    I, _ = QuadGK.quadgk(f, 0.0, 1.0; rtol=rtol, atol=atol)
    return 1 + 4I
end