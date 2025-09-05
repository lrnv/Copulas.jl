"""
    BB9Generator{T}

Fields:
  - ϑ::Real - parameter
  - δ::Real - parameter

Constructor

    BB9Generator(ϑ, δ)
    BB9Copula(d, ϑ, δ)

The BB9 copula is parameterized by ``\\vartheta, \\in [1,\\infty)`` and ``\\delta \\in (0, \\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = \\exp(-(\\delta^{-\\vartheta} + t)^{\\frac{1}{\\vartheta}} + \\delta^{-1}),
```

References:
* [joe2014](@cite) Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.205-206
"""
struct BB9Generator{T} <: Generator
    θ::T
    δ::T
    function BB9Generator(θ, δ)
        (θ ≥ 1) || throw(ArgumentError("θ must be ≥ 1"))
        (δ > 0) || throw(ArgumentError("δ must be > 0"))
        θ, δ, _ = promote(θ, δ, 1.0)
        new{typeof(θ)}(θ, δ)
    end
end

const BB9Copula{d, T} = ArchimedeanCopula{d, BB9Generator{T}}
BB9Copula(d, θ, δ) = ArchimedeanCopula(d, BB9Generator(θ, δ))
Distributions.params(G::BB9Generator) = (G.θ, G.δ)
max_monotony(::BB9Generator) = Inf

ϕ(  G::BB9Generator, s) = begin
    a  = inv(G.θ)
    c  = G.δ^(-G.θ)
    exp(inv(G.δ) - (s + c)^a)
end
ϕ⁻¹(G::BB9Generator, t) = (inv(G.δ) - log(t))^(G.θ) - G.δ^(-G.θ)

function ϕ⁽¹⁾(G::BB9Generator, s)
    a  = inv(G.θ);  c = G.δ^(-G.θ)
    ϕ(G,s) * ( -a * (s + c)^(a-1) )
end
function ϕ⁽ᵏ⁾(G::BB9Generator, ::Val{2}, s)
    a  = inv(G.θ);  c = G.δ^(-G.θ)
    φ  = ϕ(G,s)
    t  = s + c
    φ * ( a^2 * t^(2a-2) - a*(a-1) * t^(a-2) )
end

ϕ⁻¹⁽¹⁾(G::BB9Generator, t) = -G.θ * (inv(G.δ) - log(t))^(G.θ - 1) / t

williamson_dist(G::BB9Generator, ::Val{d}) where d = WilliamsonFromFrailty(TiltedPositiveStable(inv(G.θ), G.δ^(-G.θ)), Val{d}())
frailty_dist(G::BB9Generator) =  TiltedPositiveStable(inv(G.θ), G.δ^(-G.θ))
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB9Generator}
    θ, δ = C.G.θ, C.G.δ
    x = inv(δ) - log(u[1])
    y = inv(δ) - log(u[2])
    c = δ^(-θ)
    A = (x^θ + y^θ - c)^(1/θ)
    return exp(inv(δ) - A)
end

function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:BB9Generator}
    T = promote_type(Float64, eltype(u))
    (0.0 < u[1] ≤ 1.0 && 0.0 < u[2] ≤ 1.0) || return T(-Inf)

    θ, δ = C.G.θ, C.G.δ
    x = inv(δ) - log(u[1])
    y = inv(δ) - log(u[2])
    S = x^θ + y^θ - δ^(-θ)
    S ≤ 0 && return T(-Inf)

    A = S^(1/θ)
    logGbar = inv(δ) - A
    logc = logGbar +
           (1/θ - 2)*log(S) +
           log(A + θ - 1) +
           (θ - 1)*(log(x) + log(y)) -
           (log(u[1]) + log(u[2]))

    return T(logc)
end