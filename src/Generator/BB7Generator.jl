"""
    BB7Generator{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB7Generator(θ, δ)
    BB7Copula(d, θ, δ)

The BB7 copula is parameterized by ``\\theta \\in [1,\\infty)`` and ``\\delta \\in (0, \\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = 1 - \\Big[ 1 - (1 + t)^{-1/\\delta} \\Big]^{1/\\theta}.
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.202-203
"""
struct BB7Generator{T} <: AbstractFrailtyGenerator
    θ::T
    δ::T
    function BB7Generator(θ, δ)
        (θ ≥ 1) || throw(ArgumentError("θ must be ≥ 1"))
        (δ > 0) || throw(ArgumentError("δ must be > 0"))
        if θ == 1
            return ClaytonGenerator(δ)
        end
        θ, δ, _ = promote(θ, δ, 1.0)
        return new{typeof(θ)}(θ, δ)
    end
end

const BB7Copula{d, T} = ArchimedeanCopula{d, BB7Generator{T}}
Distributions.params(G::BB7Generator) = (θ = G.θ, δ = G.δ)
_unbound_params(::Type{<:BB7Generator}, d, θ) = [log(θ.θ - 1), log(θ.δ)]
_rebound_params(::Type{<:BB7Generator}, d, α) = (; θ = 1 + exp(α[1]), δ = exp(α[2]))

ϕ(  G::BB7Generator, s) = begin
    a = exp( -inv(G.δ)*log1p(s) )  
    return 1 - (1 - a)^(inv(G.θ))
end

ϕ⁻¹(G::BB7Generator, t) = begin
    w = -expm1(G.θ*log1p(-t))                # 1 - (1-t)^θ 
    return exp(-G.δ*log(w)) - 1              # w^(-δ) - 1
end

function ϕ⁽¹⁾(G::BB7Generator, s)
    return -(1/(G.θ*G.δ)) * (1 - exp(-inv(G.δ)*log1p(s)))^(inv(G.θ)-1) * (1+s)^(-inv(G.δ)-1)
end

function ϕ⁽ᵏ⁾(G::BB7Generator, ::Val{2}, s)
    θ, δ = G.θ, G.δ
    invθ, invδ = inv(θ), inv(δ)
    a   = exp(-invδ * log1p(s))                 # (1+s)^(-1/δ)
    fac = exp(-(invδ + 2) * log1p(s))           # a/(1+s)^2  = (1+s)^(-1/δ - 2)
    return (invθ*invδ) * fac * (1 - a)^(invθ - 2) *
           ( (1 + invδ) - (1 + invθ*invδ)*a )
end

ϕ⁻¹⁽¹⁾(G::BB7Generator, u) = begin
    θ, δ = G.θ, G.δ
    m = exp(θ*log1p(-u))          # (1-u)^θ
    w = 1 - m                     # 1 - (1-u)^θ
    -δ*θ * (1-u)^(θ-1) * w^(-δ-1) # **negativo**
end

frailty(G::BB7Generator) = SibuyaStoppedGamma(G.θ, G.δ)
# --------------- CDF y log-PDF (d = 2) ----------------

function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB7Generator}
    θ, δ = C.G.θ, C.G.δ
    u1, u2 = u

    log1mu1 = log1p(-u1)
    log1mu2 = log1p(-u2)

    w1 = -expm1(θ*log1mu1)                   # 1 - (1-u1)^θ
    w2 = -expm1(θ*log1mu2)

    log_xp1 = -δ*log(w1)                     # log(x+1)
    log_yp1 = -δ*log(w2)

    L  = exp(log_xp1) + exp(log_yp1) - 1.0
    t  = exp( (-1/δ)*log(L) )                # L^(-1/δ)
    return 1 - exp( (1/θ)*log1p(-t) )
end

function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB7Generator}
    Tret = promote_type(Float64, eltype(u))
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return Tret(-Inf)

    θ, δ = C.G.θ, C.G.δ
    log1mu1 = log1p(-u1)
    log1mu2 = log1p(-u2)

    w1 = -expm1(θ*log1mu1)                   # 1 - (1-u1)^θ
    w2 = -expm1(θ*log1mu2)

    log_xp1 = -δ*log(w1)                     # log(x+1)
    log_yp1 = -δ*log(w2)

    L   = exp(log_xp1) + exp(log_yp1) - 1.0
    logL = log(L)
    t    = exp( (-1/δ)*logL )                # L^(-1/δ)
    logA = log1p(-t)                         # log(1 - L^(-1/δ))

    # factores
    log_fac =
        (1/θ - 2)*logA +
        (-1/δ - 2)*logL +
        (1 + 1/δ)*(log_xp1 + log_yp1) +
        (θ - 1)*(log1mu1 + log1mu2)

    B = θ*(δ+1) - (θ*δ + 1)*t
    (B > 0) || return Tret(-Inf)

    return Tret(log_fac + log(B))
end

function _τ_bb7(θ::Real, δ::Real; tol::Real=1e-12, maxiter::Int=10^6, θtol::Real=1e-8)
    θ == 1 && return δ/(δ+2)

    if abs(θ - 2) ≤ θtol
        return 1 + (1 - Base.MathConstants.eulergamma - SpecialFunctions.digamma(δ + 2))/δ
    elseif 1 < θ && θ < 2
        a = 2/θ - 1
        return 1 - 2/(δ*(2-θ)) + (4/(δ*θ^2)) * exp(SpecialFunctions.logbeta(δ+2, a))
    else
        a  = 2/θ
        p  = 1.0
        S  = p/(a*(a+1))
        for i in 0:maxiter-2
            p *= (i+1 - δ)/(i+2)
            a += 1
            term = p/(a*(a+1))
            S += term
            if abs(term) ≤ tol*max(1.0, abs(S))
                return 1 - 4/(θ^2) * S
            end
        end
        error("_τ_bb7: series did not converge in $maxiter iterations (θ=$θ, δ=$δ)")
    end
end

τ(G::BB7Generator) = _τ_bb7(G.θ, G.δ)

λᵤ(G::BB7Generator) = 2 - 2^(inv(G.θ))
λₗ(G::BB7Generator) = 2^(-inv(G.δ))

function β(G::BB7Generator)
    θ, δ = G.θ, G.δ
    log2 = log(2.0)

    # a = 1 - 2^{-θ}  (estable)
    a = -expm1(-θ * log2)                # ∈ (0,1)

    # B = 2 * a^{-δ} - 1  (usar expm1 para estabilidad)
    # x = log(2) - δ*log(a)  =>  B = expm1(x)
    x  = log2 - δ * log(a)
    B  = expm1(x)                        # > 0

    # C = B^{-1/δ}  ⇒ logC = -(1/δ)*log(B)
    logC = -(1/δ) * log(B)

    # D = 1 - C  ⇒ logD = log(1 - e^{logC}) = log1p(-exp(logC))
    # (estable incluso cuando C≈1)
    logD = log1p(-exp(logC))             # < 0

    # β* = 1 - D^{1/θ}  ⇒ 1 - exp((1/θ)*logD)
    beta_star = 1 - exp((1/θ) * logD)

    βval = 4*beta_star - 1
    return clamp(βval, -1.0, 1.0)        # recorte defensivo
end
