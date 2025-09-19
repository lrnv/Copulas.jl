"""
    JoeGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    JoeGenerator(θ)
    JoeCopula(d,θ)

The [Joe](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator:

```math
\\phi(t) = 1 - \\big(1 - e^{-t}\\big)^{1/\\theta}.
```

It has a few special cases:
- When θ = 1, it is the IndependentCopula
- When θ = ∞, it is the MCopula (Upper Fréchet–Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct JoeGenerator{T} <: AbstractFrailtyGenerator
    θ::T
    function JoeGenerator(θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than 1"))
        elseif θ == 1
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const JoeCopula{d, T} = ArchimedeanCopula{d, JoeGenerator{T}}
JoeCopula(d, θ) = ArchimedeanCopula(d, JoeGenerator(θ))
Distributions.params(G::JoeGenerator) = (G.θ,)
frailty(G::JoeGenerator) = Sibuya(1/G.θ)

ϕ(  G::JoeGenerator, t) = 1-(-expm1(-t))^(1/G.θ)
ϕ⁻¹(G::JoeGenerator, t) = -log1p(-(1-t)^G.θ)
ϕ⁽¹⁾(G::JoeGenerator, t) = (-expm1(-t))^(1/G.θ) / (G.θ - G.θ * exp(t))
function ϕ⁽ᵏ⁾(G::JoeGenerator, ::Val{d}, t) where d
    # TODO: test if this ϕ⁽ᵏ⁾ is really more 'efficient' than the default one, 
    # as we already saw that for the Gumbel is wasn't the case. 
    α = 1 / G.θ
    P_d_α = sum(
        BigCombinatorics.Stirling2(d, k + 1) *
        (SpecialFunctions.gamma(k + 1 - α) / SpecialFunctions.gamma(1 - α)) *
        (exp(-t) / (-expm1(-t)))^k for k in 0:(d - 1)
    )
    return (-1)^d * α * (exp(-t) / (-expm1(-t))^(1 - α)) * P_d_α
end
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
        @info "JoeCopula cannot handle κ < 0."
        return one(tau)
    else
        return Roots.find_zero(θ -> _joe_tau(θ) - tau, (one(tau),tau*Inf))
    end
end

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