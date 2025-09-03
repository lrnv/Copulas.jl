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

@inline function _clipu_bb3(u::Real, θ::Real, δ::Real)
    tmax = (log(floatmax(Float64)) - 8.0) / δ
    ϵθδ = exp(-tmax^(inv(θ)))
    ϵ = max(1e-12, ϵθδ)
    return clamp(float(u), ϵ, 1 - 1e-12)
end

@inline function _abpair_bb3(u1::Real, u2::Real, θ::Real, δ::Real)
    u1c = _clipu_bb3(u1, θ, δ)
    u2c = _clipu_bb3(u2, θ, δ)
    a = δ * (-log(u1c))^θ
    b = δ * (-log(u2c))^θ
    return a, b, u1c, u2c
end

function _cdf(C::ArchimedeanCopula{2,G}, u::AbstractVector{<:Real}) where {G<:BB3Generator}
    u1, u2 = u[1], u[2]

    if u1 ≤ 0 || u2 ≤ 0; return zero(float(u1 + u2)) end
    if u1 ≥ 1 && u2 ≥ 1; return one(float(u1 + u2)) end
    if u1 ≥ 1; return float(u2) end
    if u2 ≥ 1; return float(u1) end

    θ, δ = C.G.θ, C.G.δ
    a, b, _, _ = _abpair_bb3(u1, u2, θ, δ)

    ℓ = LogExpFunctions.logaddexp(a, b)
    L = LogExpFunctions.logexpm1(ℓ)

    # r = (L/δ)^(1/θ)
    r = exp(inv(θ) * (log(L) - log(δ)))

    return exp(-r)  # C(u,v)
end

@inline function _log1pexpm1(a::Real, b::Real)
    T = promote_type(typeof(a), typeof(b))
    if max(a,b) < T(30)
        return log1p(expm1(a) + expm1(b))
    else
        ℓ = LogExpFunctions.logaddexp(a,b)
        return ℓ + log1p(-exp(-ℓ))
    end
end

function Distributions._logpdf(C::ArchimedeanCopula{2,G},
                               u::AbstractVector{<:Real}) where {G<:BB3Generator}
    u1, u2 = u[1], u[2]
    (0.0 < u1 < 1.0 && 0.0 < u2 < 1.0) || return -Inf

    θ, δ = C.G.θ, C.G.δ
    pw   = inv(θ)
    logδ = log(δ)

    # a = δ(-log u)^θ, b = δ(-log v)^θ  (con tu clipping)
    a, b, u1c, u2c = _abpair_bb3(u1, u2, θ, δ)
    t1, t2 = -log(u1c), -log(u2c)

    # L = log( e^a + e^b - 1 ),  (1+s)=e^L
    L    = _log1pexpm1(a, b)
    logL = log(L)
    eL   = exp(L)

    # r = (L/δ)^(1/θ)
    r = exp(pw*(logL - logδ))
    # g'(s) = δ^{-pw} * pw * L^{pw-1} / (1+s) = δ^{-pw} * pw * L^{pw-1} * e^{-L}
    g1 = (δ^(-pw)) * pw * (L^(pw - 1)) / eL

    # g''(s) = δ^{-pw} * pw * [ (pw-1)L^{pw-2} - L^{pw-1} ] / (1+s)^2
    g2 = (δ^(-pw)) * pw * ( (pw - 1)*L^(pw - 2) - L^(pw - 1) ) / (eL^2)

    φdd = exp(-r) * (g1^2 - g2)
    if !(φdd > 0) || !isfinite(φdd)
        return -Inf
    end
    logφdd = log(φdd)

    # |∂x/∂u| = δθ e^{a} t1^{θ-1} / u  (x+1 = e^a)
    logSu = logδ + log(θ) + a + (θ - 1)*log(t1) - log(u1c)
    logSv = logδ + log(θ) + b + (θ - 1)*log(t2) - log(u2c)

    return logφdd + logSu + logSv
end

function Distributions.pdf(C::ArchimedeanCopula{2,G},
                           u::AbstractVector{<:Real}) where {G<:BB3Generator}
    lp = Distributions._logpdf(C, u)
    return (lp < -745) ? 0.0 : exp(lp)
end