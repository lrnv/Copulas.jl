"""
    JoeGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    JoeGenerator(θ)
    JoeCopula(d,θ)

The [Joe](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = 1 - \\left(1 - e^{-t}\\right)^{\\frac{1}{\\theta}}
```

It has a few special cases:
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct JoeGenerator{T} <: Generator
    θ::T
    function JoeGenerator(θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than 1"))
        elseif θ == 1
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            return new{typeof(θ)}(θ)
        end
    end
end
const JoeCopula{d, T} = ArchimedeanCopula{d, JoeGenerator{T}}
JoeCopula(d, θ) = ArchimedeanCopula(d, JoeGenerator(θ))

Distributions.params(C::JoeCopula) = (C.G.θ)

max_monotony(::JoeGenerator) = Inf
ϕ(  G::JoeGenerator, t) = 1-(-expm1(-t))^(1/G.θ)
ϕ⁻¹(G::JoeGenerator, t) = -log1p(-(1-t)^G.θ)
ϕ⁽¹⁾(G::JoeGenerator, t) = (-expm1(-t))^(1/G.θ) / (G.θ - G.θ * exp(t))
function ϕ⁽ᵏ⁾(G::JoeGenerator, ::Val{d}, t) where d
    α = 1 / G.θ
    P_d_α = sum(
        BigCombinatorics.Stirling2(d, k + 1) *
        (SpecialFunctions.gamma(k + 1 - α) / SpecialFunctions.gamma(1 - α)) *
        (exp(-t) / (-expm1(-t)))^k for k in 0:(d - 1)
    )
    return (-1)^d * α * (exp(-t) / (-expm1(-t))^(1 - α)) * P_d_α
end
ϕ⁽ᵏ⁾(G::JoeGenerator, ::Val{0}, t) = ϕ(G, t)
function ϕ⁻¹⁽¹⁾(G::JoeGenerator, t)
    return -(G.θ * (1 - t)^(G.θ - 1)) / (1 - (1 - t)^G.θ)
end
_joe_tau(θ) =  1 - 4sum(1/(k*(2+k*θ)*(θ*(k-1)+2)) for k in 1:1000) # 446 in R copula.
τ(G::JoeGenerator) = _joe_tau(G.θ)
function τ⁻¹(::Type{T},tau) where T<:JoeGenerator
    if tau == 1
        return Inf
    elseif tau == 0
        return 1
    elseif tau < 0
        @info "JoeCoula cannot handle κ < 0."
        return one(tau)
    else
        return Roots.find_zero(θ -> _joe_tau(θ) - tau, (one(tau),tau*Inf))
    end
end
williamson_dist(G::JoeGenerator, ::Val{d}) where d = WilliamsonFromFrailty(Sibuya(1/G.θ), Val{d}())
frailty_dist(G::JoeGenerator) = Sibuya(1/G.θ)
function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:JoeGenerator}
    δ = C.G.θ
    uu = clamp(u[1], eps(), 1 - eps())
    vv = clamp(u[2], eps(), 1 - eps())

    a  = δ * log1p(-uu)                  # log A
    b  = δ * log1p(-vv)                  # log B
    # log(1-A) y log(1-B) con estabilidad
    l1mA = log1p(-exp(a))
    l1mB = log1p(-exp(b))
    # log S = log(1 - (1-A)(1-B)) = log1p( -exp(l1mA + l1mB) )
    logS = log1p(-exp(l1mA + l1mB))

    return 1 - exp((1/δ) * logS)
end

# ---------------------------------------------------------------------------
# 2) ρ(θ) integrating ρ = 12 ∫∫ C(u,v) du dv − 3, with C = _cdf(Cθ,(u,v))
# ---------------------------------------------------------------------------
function _rho_joe_via_cdf(θ; rtol=1e-7, atol=1e-9, maxevals=10^6)
    θ ≥ 1 || throw(ArgumentError("Joe requiere θ ≥ 1."))
    θeff = ifelse(θ == 1, 1 + 1e-12, θ)
    Cθ   = Copulas.ArchimedeanCopula(2, JoeGenerator(θeff))
    f(x) = _cdf(Cθ, (x[1], x[2]))
    I = HCubature.hcubature(f, (0.0,0.0), (1.0,1.0);
                            rtol=rtol, atol=atol, maxevals=maxevals)[1]
    return 12I - 3
end

ρ(G::JoeGenerator; rtol=1e-7, atol=1e-9, maxevals=10^6) =
    _rho_joe_via_cdf(G.θ; rtol=rtol, atol=atol, maxevals=maxevals)

    # ---------------------------------------------------------------------------
# 3) ρ⁻¹(·) by Brent, also based on TU _cdf (via _rho_joe_via_cdf)
# ---------------------------------------------------------------------------
function ρ⁻¹(::Type{JoeGenerator}, ρ̂::Real; xatol::Real=1e-8)
    ρc = clamp(ρ̂, nextfloat(0.0), prevfloat(1.0))
    f(θ) = _rho_joe_via_cdf(θ) - ρc

    a = 1 + 1e-6
    b = 5.0
    fa, fb = f(a), f(b)
    k = 0
    while signbit(fa) == signbit(fb) && b < 1e6
        b *= 2
        fb = f(b)
        k += 1
        k > 20 && break
    end

    if signbit(fa) != signbit(fb)
        return Roots.find_zero(f, (a, b), Roots.Brent(); xatol=xatol, rtol=0.0)
    else
        θ0 = 1 + 4ρc/(1 - ρc + eps())
        θ  = Roots.find_zero(f, θ0, Roots.Order1(); xatol=xatol)
        return min(θ, 1e6)
    end
end
