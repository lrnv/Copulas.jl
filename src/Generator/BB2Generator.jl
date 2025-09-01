"""
    BB2Generator{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB2Generator(θ, δ)
    BB2Copula(d, θ, δ)

The BB2 copula is parameterized by ``\\theta, \\delta \\in (0,\\infty)`. It is an Archimedean copula with generator :

```math
\\phi(t) = [1 + \\delta^{-1}log(1 + t)]^{-\\frac{1}{\\theta}},
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.193-194
"""
struct BB2Generator{T} <: Generator
    θ::T
    δ::T
    function BB2Generator(θ, δ)
        (θ > 0) || throw(ArgumentError("θ must be > 0"))
        (δ > 0) || throw(ArgumentError("δ must be > 0"))
        new{typeof(θ)}(θ, δ)
    end
end
const BB2Copula{d, T} = ArchimedeanCopula{d, BB2Generator{T}}
BB2Copula(d, θ, δ) = ArchimedeanCopula(d, BB2Generator(θ, δ))

Distributions.params(C::BB2Copula) = (C.G.θ, C.G.δ)

max_monotony(::BB2Generator) = Inf

ϕ(  G::BB2Generator, s) = exp(-log1p(log1p(s)/G.δ)/G.θ)
ϕ⁻¹(G::BB2Generator, t) = expm1(G.δ*expm1(-G.θ*log(t)))
function ϕ⁽¹⁾(G::BB2Generator, s)
    θ, δ = G.θ, G.δ
    u = log1p(s)
    v = (1+1/θ) * log1p(u/δ) + log(θ) + log(δ) + u
    return -exp(-v)
end
function ϕ⁽ᵏ⁾(G::BB2Generator, ::Val{2}, s)
    θ, δ = G.θ, G.δ
    logA = log1p(log1p(s)/δ)
    inv1p = inv(1+s)
    term1 = exp(-(1+1/θ) * logA)
    term2 = ((1/θ) + 1) * exp(-(2+1/θ) * logA) / δ
    return (1/(θ*δ)) * inv1p^2 * (term1 + term2)
end
function ϕ⁻¹⁽¹⁾(G::BB2Generator, t)
    lt = log(t)
    A = G.δ * expm1(-G.θ*lt)
    B = G.δ * exp(-(1+G.θ)*lt)
    return - G.θ * B * exp(A)
end
function ϕ⁽ᵏ⁾⁻¹(G::BB2Generator, ::Val{1}, x; start_at=x)
    # compute the inverse of ϕ⁽¹⁾
    θ, δ = G.θ, G.δ
    a = 1 + 1/θ          # a > 0
    logv = -log(a) + (δ + (a - 1) * log(δ) - log(- θ * x)) / a
    w = LambertW.lambertw(exp(logv))
    return expm1(a * w - δ)
end

# Frailty: M = S_{1/δ} * Gamma_{1/θ}^{δ}
williamson_dist(G::BB2Generator, ::Val{d}) where d = WilliamsonFromFrailty(GammaStoppedGamma(G.θ, G.δ), Val{d}())

# a,b must be log(1+w_i) = δ(u^{-θ}-1); also returns u1c,u2c
@inline function _abpair_robust(u1::Real, u2::Real, θ::Real, δ::Real)
    u1c = clamp(float(u1), 1e-15, 1-1e-15)
    u2c = clamp(float(u2), 1e-15, 1-1e-15)
    r1  = -θ * log(u1c)
    r2  = -θ * log(u2c)
    a   = δ * expm1(r1)               # = δ(u1^{-θ}-1)
    b   = δ * expm1(r2)
    return a, b, u1c, u2c
end

# ------------------ stable CDF (d = 2) ------------------
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB2Generator}
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
    a, b, _, _ = _abpair_robust(u1, u2, θ, δ)

    ℓ    = LogExpFunctions.logaddexp(a, b)             # = log(e^a + e^b)
    L    = ℓ + LogExpFunctions.log1mexp(-ℓ)            # = log(e^ℓ - 1) = log(e^a + e^b - 1)
    logA = log1p(L/δ)                                  # A = 1 + L/δ

    return exp(-inv(θ) * logA)
end

# ---------------- stable logpdf (d = 2) -----------------
function _logpdf(C::ArchimedeanCopula{2,G}, u::AbstractVector{<:Real}) where {G<:BB2Generator}
    u1, u2 = float(u[1]), float(u[2])
    (0.0 < u1 < 1.0 && 0.0 < u2 < 1.0) || return -Inf

    θ, δ = C.G.θ, C.G.δ
    a, b, u1c, u2c = _abpair_robust(u1, u2, θ, δ)

    # ℓ = log(e^a+e^b),  L = log(e^ℓ-1) = log(1+w1+w2)
    ℓ    = LogExpFunctions.logaddexp(a, b)
    L    = ℓ + LogExpFunctions.log1mexp(-ℓ)

    # A = 1 + (1/δ) L
    logA  = log1p(L/δ)
    # log(A + (1 + 1/θ)/δ)
    logAp = log1p((L + 1 + inv(θ))/δ)

    return  (log(δ*θ)                          # θδ
          - 2L                                # (x+y+1)^{-2}
          + (-inv(θ) - 2)*logA                # A^{-1/θ - 2}
          + logAp                             # A + (1+1/θ)/δ
          + a + b                             # (x+1)(y+1)
          + (-θ - 1)*(log(u1c) + log(u2c)))    # (uv)^{-θ-1}
end

function τ(G::Copulas.BB2Generator{T}; rtol=1e-10, atol=1e-12) where {T}
    θ = float(G.θ); δ = float(G.δ)

    invδθ = 1/(δ*θ)
    #   φ⁻¹/ (φ⁻¹)' = (t^(θ+1))/(δθ) * expm1(-δ*(t^(-θ) - 1))
    f(t) = (t<=0 || t>=1) ? 0.0 : (t^(θ+1)) * invδθ * LogExpFunctions.expm1(-δ*(t^(-θ) - 1))
    I, _ = QuadGK.quadgk(f, 0.0, 1.0; rtol=rtol, atol=atol)
    return 1 + 4I
end

function λᵤ(C::BB2Generator{T}; tsmall::Float64=1e-10) where {T}
    G = C.G
    r = ϕ⁽¹⁾(G, 2tsmall) / ϕ⁽¹⁾(G, tsmall)
    return 2 - 2*r
end

function λₗ(C::BB2Generator{T}; tlarge::Float64=1e6) where {T}
    G = C.G
    r = ϕ⁽¹⁾(G, 2tlarge) / ϕ⁽¹⁾(G, tlarge)
    return 2*r
end