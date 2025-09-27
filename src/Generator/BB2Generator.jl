"""
    BB2Generator{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB2Generator(θ, δ)
    BB2Copula(d, θ, δ)

The BB2 copula has parameters ``\\theta, \\delta \\in (0,\\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = [1 + \\delta^{-1}log(1 + t)]^{-\\frac{1}{\\theta}},
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.193-194
"""
struct BB2Generator{T} <: AbstractFrailtyGenerator
    θ::T
    δ::T
    function BB2Generator(θ, δ)
        (θ > 0) || throw(ArgumentError("θ must be > 0"))
        (δ > 0) || throw(ArgumentError("δ must be > 0"))
        θ, δ, _ = promote(θ, δ, 1.0)
        new{typeof(θ)}(θ, δ)
    end
end
const BB2Copula{d, T} = ArchimedeanCopula{d, BB2Generator{T}}
Distributions.params(G::BB2Generator) = (θ = G.θ, δ = G.δ)
_unbound_params(::Type{<:BB2Generator}, d, θ) = [log(θ.θ), log(θ.δ)]
_rebound_params(::Type{<:BB2Generator}, d, α) = (; θ = exp(α[1]), δ = exp(α[2]))

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
frailty(G::BB2Generator) = GammaStoppedGamma(G.θ, G.δ)
function _abpair_robust(u1::Real, u2::Real, θ::Real, δ::Real)
    u1c = clamp(float(u1), 1e-15, 1-1e-15)
    u2c = clamp(float(u2), 1e-15, 1-1e-15)
    A   = δ * LogExpFunctions.expm1(-θ*log(u1c))   # = δ(u1^{-θ}-1) = log(1+x)
    B   = δ * LogExpFunctions.expm1(-θ*log(u2c))
    return A, B, u1c, u2c
end

function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB2Generator}
    u1, u2 = u[1], u[2]
    if u1 ≤ 0 || u2 ≤ 0; return 0.0
    elseif u1 ≥ 1 && u2 ≥ 1; return 1.0
    elseif u1 ≥ 1; return float(u2)
    elseif u2 ≥ 1; return float(u1)
    end
    θ, δ = C.G.θ, C.G.δ
    A, B, _, _ = _abpair_robust(u1, u2, θ, δ)
    ℓ = LogExpFunctions.logaddexp(A, B)          # log(e^A + e^B)
    L = LogExpFunctions.logexpm1(ℓ)              # log(e^A + e^B - 1)
    logg = LogExpFunctions.log1p(L/δ)                            # log(1 + L/δ)
    return exp(-inv(θ)*logg)                     # g^{-1/θ}
end

function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB2Generator}
    u1, u2 = u[1], u[2]
    (0.0 < u1 < 1.0 && 0.0 < u2 < 1.0) || return -Inf
    θ, δ = C.G.θ, C.G.δ
    A, B, u1c, u2c = _abpair_robust(u1, u2, θ, δ)
    a, b = inv(δ), inv(θ)
    ℓ = LogExpFunctions.logaddexp(A, B)
    L = LogExpFunctions.logexpm1(ℓ)
    logg = LogExpFunctions.log1p(a*L)

    logφpp = log(a) + log(b) - 2L + (-b-1)*logg + LogExpFunctions.log1p((b+1)*a*exp(-logg))
    logd1  = A + log(δ*θ) - (θ+1)*log(u1c)
    logd2  = B + log(δ*θ) - (θ+1)*log(u2c)
    return logφpp + logd1 + logd2
end

function τ(G::Copulas.BB2Generator{T}; rtol=1e-10, atol=1e-12) where {T}
    # Closed-form for Kendall's tau using a change of variable
    # Let a = 2 + 2/θ. Then
    #   τ = 1 + 4/(δ θ^2) * ( 1/(a-1) - e^δ δ^(a-1) Γ(1-a, δ) )
    # where Γ(·,·) is the upper incomplete gamma. This comes from
    # u = t^{-θ} substitution in the standard Archimedean integral for τ.
    θ = float(G.θ); δ = float(G.δ)
    a = 2 + 2/θ
    term_gamma = exp(δ) * (δ^(a-1)) * SpecialFunctions.gamma(1 - a, δ)
    τval = 1 + 4 * ((1/(a - 1)) - term_gamma) / (δ * θ^2)
    if isfinite(τval)
        return τval
    end
    # Fallback to numerical quadrature if needed (extreme parameter regimes)
    invδθ = 1/(δ*θ)
    f(t) = (t<=0 || t>=1) ? 0.0 : (t^(θ+1)) * invδθ * LogExpFunctions.expm1(-δ*(t^(-θ) - 1))
    I, _ = QuadGK.quadgk(f, 0.0, 1.0; rtol=rtol, atol=atol)
    return 1 + 4I
end

# Spearman's rho via a stable single-integral formulation
function ρ(G::Copulas.BB2Generator{T}; rtol=1e-7, atol=1e-9) where {T}
    # Use J = ∫_0^∞ ϕ(s) [∫_0^s g(x) g(s-x) dx] ds with g = -ϕ'(·)
    # Outer: map s = t/(1-t) for t∈(0,1) to control tails
    gfun(s) = -ϕ⁽¹⁾(G, s)
    inner(s) = s <= 0 ? 0.0 : begin
        innerf(z) = gfun(s*z) * gfun(s*(1 - z))
        val, _ = QuadGK.quadgk(innerf, 0.0, 1.0; rtol=sqrt(rtol), atol=sqrt(atol))
        s * val
    end
    outerf(t) = (t <= 0 || t >= 1) ? 0.0 : begin
        s = t/(1 - t)
        jac = 1/(1 - t)^2
        ϕ(G, s) * inner(s) * jac
    end
    I, _ = QuadGK.quadgk(outerf, 0.0, 1.0; rtol=rtol, atol=atol)
    return 12I - 3
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