"""
    BB8Generator{T}

Fields:
  - ϑ::Real - parameter
  - δ::Real - parameter

Constructor

    BB8Generator(ϑ, δ)
    BB8Copula(d, ϑ, δ)

The BB8 copula is parameterized by ``\\vartheta, \\in [1,\\infty)`` and ``\\delta \\in (0, 1]``. It is an Archimedean copula with generator :

```math
\\phi(t) = \\delta^{-1}[1 - (1 - \\eta \\exp(-t))^{\\frac{1}{\\vartheta}}],
```

where ``\\eta = 1 - (1 - \\delta)^{\\vartheta}.``

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.204-205
"""
struct BB8Generator{T} <: Generator
    ϑ::T
    δ::T
    function BB8Generator(ϑ, δ)
        (ϑ ≥ 1) || throw(ArgumentError("ϑ must be ≥ 1"))
        (0 < δ ≤ 1) || throw(ArgumentError("δ must be in (0,1]"))
        if δ == 1
            return JoeGenerator(ϑ)
        else
            ϑ, δ, _ = promote(ϑ, δ, 1.0)
            return new{typeof(ϑ)}(ϑ, δ)
        end
    end
end

const BB8Copula{d, T} = ArchimedeanCopula{d, BB8Generator{T}}
BB8Copula(d, ϑ, δ) = ArchimedeanCopula(d, BB8Generator(ϑ, δ))
Distributions.params(G::BB8Generator) = (G.ϑ, G.δ)
max_monotony(::BB8Generator) = Inf 

@inline _η(G::BB8Generator) = -expm1(G.ϑ * log1p(-G.δ))

ϕ(G::BB8Generator, s)  = (1/G.δ) * (1 - (1 - _η(G)*exp(-s))^(inv(G.ϑ)))
ϕ⁻¹(G::BB8Generator, t) = -log((1 - (1 - G.δ*t)^G.ϑ)/_η(G))
ϕ⁽¹⁾(G::BB8Generator, s) = -(_η(G)/(G.δ*G.ϑ)) * exp(-s) * (1 - _η(G)*exp(-s))^(inv(G.ϑ)-1)

function ϕ⁽ᵏ⁾(G::BB8Generator, ::Val{2}, s)
    δ, ϑ = G.δ, G.ϑ
    α, β = inv(δ), inv(ϑ)
    ηv   = _η(G)
    u    = exp(-s)
    b    = 1 - ηv*u
    return (α*β*ηv) * u * b^(β - 2) * (1 - β*ηv*u)
end
ϕ⁻¹⁽¹⁾(G::BB8Generator, t) = -G.ϑ*G.δ * (1 - G.δ*t)^(G.ϑ - 1) / (1 - (1 - G.δ*t)^G.ϑ)

williamson_dist(G::BB8Generator, ::Val{d}) where d = WilliamsonFromFrailty(GeneralizedSibuya(G.ϑ, G.δ), Val{d}())

function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB8Generator}
    ϑ, δ = C.G.ϑ, C.G.δ
    η = -expm1(ϑ*log1p(-δ))
    x = -expm1(ϑ*log1p(-δ*u[1]))
    y = -expm1(ϑ*log1p(-δ*u[2]))
    z = (x*y)/η
    t = exp((1/ϑ)*log1p(-z))           
    return (1/δ) * (1 - t)
end

function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB8Generator}
    Tret = promote_type(Float64, eltype(u))
    u1, u2 = u
    (0.0 < u1 ≤ 1.0 && 0.0 < u2 ≤ 1.0) || return Tret(-Inf)

    ϑ, δ = C.G.ϑ, C.G.δ
    η = -expm1(ϑ*log1p(-δ))

    log1mδu1 = log1p(-δ*u1)
    log1mδu2 = log1p(-δ*u2)
    x = -expm1(ϑ*log1mδu1)
    y = -expm1(ϑ*log1mδu2)

    z = (x*y)/η
    (0.0 ≤ z < 1.0) || return Tret(-Inf)

    log1mz = log1p(-z)
    θminus = ϑ - z
    (θminus > 0) || return Tret(-Inf)

    logc =  log(δ) - log(η) +
            (1/ϑ - 2)*log1mz +
            log(θminus) +
            (ϑ - 1)*(log1mδu1 + log1mδu2)

    return Tret(logc)
end