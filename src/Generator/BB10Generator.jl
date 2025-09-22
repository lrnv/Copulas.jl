"""
    BB10Generator{T}

Fields:
  - θ::Real - parameter
  - δ::Real - parameter

Constructor

    BB10Generator(θ, δ)
    BB10Copula(d, θ, δ)

The BB10 copula has parameters ``\\theta \\in (0,\\infty)`` and ``\\delta \\in [0, 1]``. It is an Archimedean copula with generator:

```math
\\phi(t) = \\Big(\\tfrac{1-\\delta}{e^{t}-\\delta}\\Big)^{1/\\theta},
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.206-207
"""
struct BB10Generator{T} <: AbstractFrailtyGenerator
    θ::T          # θ > 0
    δ::T          # 0 ≤ δ ≤ 1
    function BB10Generator(θ, δ)
        (θ > 0) || throw(ArgumentError("θ must be > 0"))
        (0 ≤ δ ≤ 1) || throw(ArgumentError("δ must be in [0,1]"))
        if θ == 1
            return AMHGenerator(δ)
        else 
            θ, δ, _ = promote(θ, δ, 1.0)
            return new{typeof(θ)}(θ, δ)
        end
    end
end

const BB10Copula{d, T} = ArchimedeanCopula{d, BB10Generator{T}}
BB10Copula(d, θ, δ) = ArchimedeanCopula(d, BB10Generator(θ, δ))
BB10Copula(d; θ::Real, δ::Real) = BB10Copula(d, θ, δ)
Distributions.params(G::BB10Generator) = (θ = G.θ, δ = G.δ)
_example(CT::Type{<:BB10Copula}, d) = BB10Copula(d, 2.0, 0.4)
_unbound_params(::Type{<:BB10Copula}, d, θ) = [log(θ.θ), log(θ.δ) - log1p(-θ.δ)]  # logit(δ)
_rebound_params(::Type{<:BB10Copula}, d, α) = (; θ = exp(α[1]), δ = 1 / (1 + exp(-α[2])))

ϕ(G::BB10Generator, s) = begin
    θ, δ = G.θ, G.δ
    exp( (1/θ) * (log1p(-δ) - log(expm1(s) + (1 - δ))) )
end

ϕ⁻¹(G::BB10Generator, t) = begin
    θ, δ = G.θ, G.δ
    log(δ + (1 - δ) * exp(-θ * log(t)))
end

function ϕ⁽¹⁾(G::BB10Generator, s)
    θ, δ = G.θ, G.δ
    es = exp(s)
    ψ  = ϕ(G, s)
    return -(1/θ) * es/(es - δ) * ψ
end
function ϕ⁽ᵏ⁾(G::BB10Generator, ::Val{2}, s)
    θ, δ = G.θ, G.δ
    es = exp(s)
    ψ  = ϕ(G, s)                    # ya usa forma estable con log1p/expm1
    den = es - δ
    return ψ * (es / (den^2)) * (es/θ^2 + δ/θ)
end

ϕ⁻¹⁽¹⁾(G::BB10Generator, t) = begin
    θ, δ = G.θ, G.δ
    num = -θ * (1 - δ) * exp(-(θ + 1) * log(t))
    den =  δ + (1 - δ) * exp(-θ * log(t))
    num/den
end

frailty(G::BB10Generator) = ShiftedNegBin(inv(G.θ), 1 - G.δ)
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB10Generator}
    θ, δ = C.G.θ, C.G.δ
    uθ, vθ = u[1]^θ, u[2]^θ
    D = 1 - δ*(1 - uθ)*(1 - vθ)
    return exp( log(u[1]) + log(u[2]) - (1/θ)*log(D) )
end

# --- log-density
function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB10Generator}
    T = promote_type(Float64, eltype(u))
    (0.0 < u[1] ≤ 1.0 && 0.0 < u[2] ≤ 1.0) || return T(-Inf)

    θ, δ = C.G.θ, C.G.δ
    uθ, vθ = u[1]^θ, u[2]^θ

    D = 1 - δ + δ*(uθ + vθ) - δ*uθ*vθ
    K = (1 - δ)^2 + δ*(1 - δ)*(uθ + vθ) + δ*(θ + δ)*uθ*vθ

    (D > 0 && K > 0) || return T(-Inf)

    logc = (-1/θ - 2)*log(D) + log(K)
    return T(logc)
end