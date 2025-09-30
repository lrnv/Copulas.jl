"""
    BB6Generator{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB6Generator(θ, δ)
    BB6Copula(d, θ, δ)

The BB6 copula has parameters ``\\theta, \\delta \\in [1,\\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = 1 - [1 - \\exp(-t^{\\frac{1}{\\delta}})]^{\\frac{1}{\\theta}}
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.200-201
"""
struct BB6Generator{T} <: AbstractFrailtyGenerator
    θ::T
    δ::T
    function BB6Generator(θ, δ)
        (θ ≥ 1) || throw(ArgumentError("θ must be ≥ 1"))
        (δ ≥ 1) || throw(ArgumentError("δ must be ≥ 1"))
        if δ == 1
            return JoeGenerator(θ)
        elseif θ == 1
            return GumbelGenerator(δ)
        else
            θ, δ, _ = promote(θ, δ, 1.0)
            return new{typeof(θ)}(θ, δ)
        end
    end
    BB6Generator{T}(θ, δ) where T = BB6Generator(promote(θ, δ, one(T))[1:2]...)
end

const BB6Copula{d, T} = ArchimedeanCopula{d, BB6Generator{T}}
BB6Copula(d, θ, δ) = ArchimedeanCopula(d, BB6Generator(θ, δ))
BB6Copula(d; θ::Real, δ::Real) = BB6Copula(d, θ, δ)
Distributions.params(G::BB6Generator) = (θ = G.θ, δ = G.δ)
_example(CT::Type{<:BB6Copula}, d) = BB6Copula(d, 1.5, 1.5)
_unbound_params(::Type{<:BB6Copula}, d, θ) = [log(θ.θ - 1), log(θ.δ - 1)]
_rebound_params(::Type{<:BB6Copula}, d, α) = (; θ = 1 + exp(α[1]), δ = 1 + exp(α[2]))

ϕ(  G::BB6Generator, s) = 1 - (1 - exp(-s^(inv(G.δ))))^(inv(G.θ))
ϕ⁻¹(G::BB6Generator, t) = (-log1p(- (1 - t)^(G.θ)))^(G.δ)

function ϕ⁽¹⁾(G::BB6Generator, s)
    a = inv(G.θ); b = inv(G.δ)
    r = s^b
    E = exp(-r)
    H = 1 - E
    return -(a*b) * s^(b-1) * E * H^(a-1)
end

function ϕ⁽ᵏ⁾(G::BB6Generator, ::Val{2}, s)
    a = inv(G.θ); b = inv(G.δ)
    r = s^b
    E = exp(-r)
    H = 1 - E
    term = (b - 1) * s^(b - 2) - b * s^(2b - 2) + (a - 1) * b * s^(2b - 2) * (E / H)
    return -a * b * E * H^(a - 1) * term 
end
function ϕ⁻¹⁽¹⁾(G::BB6Generator, u::Real)
    θ, δ = G.θ, G.δ
    h  = 1 - (1 - u)^θ                  # ∈ (0,1]
    j  = -log(h)
    return -δ*θ * j^(δ - 1) * (1 - u)^(θ - 1) / h
end

frailty(G::BB6Generator) = SibuyaStoppedPosStable(G.θ, G.δ)
# ------------------ CDF (d = 2) ------------------
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB6Generator}
    θ, δ = C.G.θ, C.G.δ
    ū = 1 - u[1];  v̄ = 1 - u[2]
    x = -log1p(- ū^θ)         # x = -log(1 - (1-u)^θ)
    y = -log1p(- v̄^θ)
    sδ  = exp(LogExpFunctions.logaddexp(δ*log(x), δ*log(y)))   # x^δ + y^δ
    s1d = exp((1/δ)*log(sδ))
    w = exp(-s1d)
    return 1 - (1 - w)^(inv(θ))
end

# ------------------ log-PDF (d = 2) ------------------
function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB6Generator}
    Tret = promote_type(Float64, eltype(u))
    (0.0 < u[1] ≤ 1.0 && 0.0 < u[2] ≤ 1.0) || return Tret(-Inf)

    θ, δ = C.G.θ, C.G.δ
    ū = 1 - u[1];  v̄ = 1 - u[2]

    # x = -log(1 - (1-u)^θ)
    x = -log1p(- ū^θ)
    y = -log1p(- v̄^θ)
    (x>0 && y>0) || return Tret(-Inf)

    log_sδ = LogExpFunctions.logaddexp(δ*log(x), δ*log(y))
    sδ  = exp(log_sδ)
    s1d = exp((1/δ)*log_sδ)
    w   = exp(-s1d)

    log_fac1 = (inv(θ) - 2) * log1p(-w) + log(w)
    log_fac2 = (inv(δ) - 2) * log_sδ
    B = (θ - w)*s1d + θ*(δ - 1)*(1 - w)  
    B > 0 || return Tret(-Inf)
    log_fac3 = log(B)
    log_fac4 = (δ - 1)*(log(x) + log(y))
    log_fac5 = -( log1p(- ū^θ) + log1p(- v̄^θ) ) 
    log_fac6 = (θ - 1)*(log(ū) + log(v̄))

    return Tret(log_fac1 + log_fac2 + log_fac3 + log_fac4 + log_fac5 + log_fac6)
end

# --- Tail dependence (BB6) ---
λₗ(G::BB6Generator) = 0.0
λᵤ(G::BB6Generator) = 2 - 2.0^(1.0/(G.θ*G.δ)) # 2 - 2^{1/(θδ)}

# --- Blomqvist beta (BB6) ---
function β(G::BB6Generator)
    θ, δ = G.θ, G.δ
    # 1 - 2^{-θ} con estabilidad
    one_minus_two_pow_negθ = -expm1(-θ * log(2))        # = 1 - 2^{-θ}
    # (1 - 2^{-θ})^{2^{1/δ}}
    pow_inner = exp( exp2(1/δ) * log(one_minus_two_pow_negθ) )
    # 1 - (...)^{1/θ}
    beta_star = 1 - exp( (1/θ) * log1p(-pow_inner) )
    return 4*beta_star - 1
end

# --- Kendall tau (BB6) — vía integral para la CÓPULA ---
# τ(C) = 1 + 4 ∫ φ(t)/φ'(t) dt, con φ = ϕ⁻¹ del generador de C
function τ(C::BB6Copula; rtol=1e-8)
    G = C.G
    φ(t)  = ϕ⁻¹(G, t)
    φ′(t) = ϕ⁻¹⁽¹⁾(G, t)        # ya lo tienes; si no, diferencia/AD
    f(t)  = φ(t) / φ′(t)
    a = eps(Float64); b = 1 - eps(Float64)
    val = QuadGK.quadgk(f, a, b; rtol=rtol)[1]   # o tu cuadratura preferida
    return 1 + 4*val
end
