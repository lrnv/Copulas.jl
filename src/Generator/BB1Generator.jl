"""
    BB1Generator{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB1Generator(θ, δ)
    BB1Copula(d, θ, δ)

The BB1 copula is parameterized by ``\\theta \\in (0,\\infty)`` and ``\\delta \\in [1, \\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = (1 + t^{\\frac{1}{δ}})^{\\frac{-1}{θ}},
```

It has a few special cases:
- When δ = 1, it is the ClaytonCopula with parameter `\\theta`. 

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.190-192
"""
struct BB1Generator{T} <: Generator
    θ::T
    δ::T
    function BB1Generator(θ, δ)
        (θ > 0) || throw(ArgumentError("θ must be > 0"))
        (δ ≥ 1) || throw(ArgumentError("δ must be ≥ 1"))
        if δ == 1
            return ClaytonGenerator(θ)
        else
            θ, δ, _ = promote(θ, δ, 1.0)
            return new{typeof(θ)}(θ, δ)
        end
    end
end
const BB1Copula{d, T} = ArchimedeanCopula{d, BB1Generator{T}}
BB1Copula(d, θ, δ) = ArchimedeanCopula(d, BB1Generator(θ, δ))

Distributions.params(G::BB1Generator) = (G.θ, G.δ)

max_monotony(::BB1Generator) = Inf   # Maybe extendable for dimension d???

# --- generator ψ and inversa ψ^{-1} ---
ϕ(G::BB1Generator, s) = exp(-inv(G.θ) * log1p(s^(inv(G.δ))))
ϕ⁻¹(G::BB1Generator, t) = (expm1(-G.θ * log(t)))^(G.δ)  # evita t^(-θ)-1
function ϕ⁽¹⁾(G::BB1Generator, s)
    a = inv(G.δ)                                   # a = 1/δ
    return -(a/G.θ) * s^(a-1) * (1 + s^a)^(-inv(G.θ)-1)
end
function ϕ⁽ᵏ⁾(G::BB1Generator, ::Val{2}, s) # only d=2 case, other cases are not implemented. 
    a = inv(G.δ)
    return (a/G.θ) * s^(a-2) * (1 + s^a)^(-inv(G.θ)-2) *
           ( (1 + a/G.θ)*s^a - (a - 1) )
end
ϕ⁻¹⁽¹⁾(G::BB1Generator, t) = -G.δ*G.θ * t^(-G.θ-1) * (t^(-G.θ) - 1)^(G.δ-1)

# Frailty: M = S_{1/δ} * Gamma_{1/θ}^{δ}
williamson_dist(G::BB1Generator, ::Val{d}) where d = WilliamsonFromFrailty(GammaStoppedPositiveStable(inv(G.δ), inv(G.θ)), Val{d}())
frailty(G::BB1Generator) = GammaStoppedPositiveStable(inv(G.δ), inv(G.θ))
# --- CDF and logpdf (d=2), numeric stable version ---
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB1Generator}
    θ, δ = C.G.θ, C.G.δ
    w1 = expm1(-θ*log(u[1]))                 # u1^{-θ} - 1
    w2 = expm1(-θ*log(u[2]))                 # u2^{-θ} - 1
    s  = w1^δ + w2^δ
    sa = s^(1/δ)
    return exp( - (1/θ) * log1p(sa) )
end

function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB1Generator}
    T = promote_type(Float64, eltype(u))
    (0.0 < u[1] ≤ 1.0 && 0.0 < u[2] ≤ 1.0) || return T(-Inf)

    θ, δ = C.G.θ, C.G.δ
    a    = inv(δ)
    w1   = expm1(-θ*log(u[1]))               # u1^{-θ} - 1
    w2   = expm1(-θ*log(u[2]))               # u2^{-θ} - 1
    s    = w1^δ + w2^δ
    sa   = s^a

    # log c(u,v) = log ψ''(s) + log (ψ^{-1})'(u) + log (ψ^{-1})'(v)
    logc = log(δ*θ) +                         # (= log(a/θ) + 2log(δθ))
           (a - 2)*log(s) +
           (-inv(θ) - 2)*log1p(sa) +
           log((1 + a/θ)*sa - (a - 1)) +
           (δ - 1)*(log(w1) + log(w2)) +
           (-θ - 1)*(log(u[1]) + log(u[2]))

    return T(logc)
end

# === Momentos teóricos (generator y copula) ================================

# Kendall tau
τ(G::BB1Generator)  = 1 - 2 / (G.δ * (G.θ + 2))

# Tail dependence
λₗ(G::BB1Generator) = 2^(-inv(G.θ * G.δ))          # 2^{-1/(θδ)}
λᵤ(G::BB1Generator) = 2 - 2^(inv(G.δ))             # 2 - 2^{1/δ}

# Blomqvist β
# β* = { 1 + 2^{1/δ}(2^θ - 1) }^{-1/θ},   β = 4β* - 1
β(G::BB1Generator)  = begin
    θ, δ  = G.θ, G.δ
    βstar = (1 + 2.0^(1/δ) * (2.0^θ - 1.0))^(-1/θ)
    4βstar - 1
end

