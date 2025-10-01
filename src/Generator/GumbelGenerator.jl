"""
    GumbelGenerator{T}

Fields:
  - θ::Real - parameter

Constructor

    GumbelGenerator(θ)
    GumbelCopula(d,θ)

The [Gumbel](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [1,\\infty)``. It is an Archimedean copula with generator :

```math
\\phi(t) = \\exp\\!\\big( - t^{1/\\theta} \\big).
```

It has a few special cases:
- When θ = 1, it is the IndependentCopula
- When θ → ∞, it is the MCopula (Upper Fréchet–Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct GumbelGenerator{T} <: AbstractUnivariateFrailtyGenerator
    θ::T
    function GumbelGenerator(θ)
        if θ < 1
            throw(ArgumentError("Theta must be greater than or equal to 1"))
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
const GumbelCopula{d, T} = ArchimedeanCopula{d, GumbelGenerator{T}}
frailty(G::GumbelGenerator) = AlphaStable(α = 1/G.θ, β = 1,scale = cos(π/(2G.θ))^G.θ, location = (G.θ == 1 ? 1 : 0))
Distributions.params(G::GumbelGenerator) = (θ = G.θ,)
_unbound_params(::Type{<:GumbelGenerator}, d, θ) = [log(θ.θ - 1)]                # θ ≥ 1
_rebound_params(::Type{<:GumbelGenerator}, d, α) = (; θ = 1 + exp(α[1]))
_θ_bounds(::Type{<:GumbelGenerator}, d) = (1, Inf)

ϕ(  G::GumbelGenerator, t) = exp(-exp(log(t)/G.θ))
ϕ⁻¹(G::GumbelGenerator, t) = exp(log(-log(t))*G.θ)
function ϕ⁽¹⁾(G::GumbelGenerator, t)
    # first derivative of ϕ
    a = 1/G.θ
    tam1 = exp((a-1)*log(t))
    return - a * tam1 * exp(-tam1*t)
end

# The folliwng function got commented because it does WORSE in term of runtime than the 
# corredponsing generic :)

function ϕ⁽ᵏ⁾(G::GumbelGenerator, ::Val{d}, t) where d
    α = 1 / G.θ
    return eltype(t)(ϕ(G, t) * t^(-d) * sum(
        α^j * Float64(BigCombinatorics.Stirling1(d, j)) * sum(Float64(BigCombinatorics.Stirling2(j, k)) * (-t^α)^k for k in 1:j) for j in 1:d
    ))
end
ϕ⁻¹⁽¹⁾(G::GumbelGenerator, t) = -(G.θ * exp(log(-log(t))*(G.θ - 1))) / t
τ(G::GumbelGenerator) = ifelse(isfinite(G.θ), (G.θ-1)/G.θ, 1)
function τ⁻¹(::Type{<:GumbelGenerator}, τ)
    τ ≥ 1 && return Inf
    τ ≤ 0 && return one(τ)
    return 1/(1-τ)
end

function _cdf(C::ArchimedeanCopula{2,G}, u) where {G<:GumbelGenerator}
    θ = C.G.θ
    x₁, x₂ = -log(u[1]), -log(u[2])
    lx₁, lx₂ = log(x₁), log(x₂)
    return 1 - LogExpFunctions.cexpexp(LogExpFunctions.logaddexp(θ * lx₁, θ * lx₂) / θ)
end
function Distributions._logpdf(C::ArchimedeanCopula{2,G}, u) where {G<:GumbelGenerator}
    T = promote_type(Float64, eltype(u))
    !all(0 .< u .<= 1) && return T(-Inf) # if not in range return -Inf

    θ = C.G.θ
    x₁, x₂ = -log(u[1]), -log(u[2])
    lx₁, lx₂ = log(x₁), log(x₂)
    A = LogExpFunctions.logaddexp(θ * lx₁, θ * lx₂)
    B = exp(A/θ)
    return - B + x₁ + x₂ + (θ-1) * (lx₁ + lx₂) + A/θ - 2A + log(B + θ - 1)
end

function _rho_gumbel_via_cdf(θ; rtol=1e-7, atol=1e-9, maxevals=10^6)
    θeff = clamp(θ, 1+1e-12, Inf)
    Cθ   = Copulas.ArchimedeanCopula(2, GumbelGenerator(θeff))
    f(x) = _cdf(Cθ, (x[1], x[2]))  # <- tu _cdf
    I = HCubature.hcubature(f, (0.0,0.0), (1.0,1.0);
                            rtol=rtol, atol=atol, maxevals=maxevals)[1]
    return 12I - 3
end

ρ(G::GumbelGenerator; rtol=1e-7, atol=1e-9, maxevals=10^6) =
    _rho_gumbel_via_cdf(G.θ; rtol=rtol, atol=atol, maxevals=maxevals)

function ρ⁻¹(::Type{<:GumbelGenerator}, ρ̂; xatol=1e-8)
    # Rango de Spearman para Gumbel: [0, 1)
    ρc = clamp(ρ̂, nextfloat(0.0), prevfloat(1.0))
    f(θ) = _rho_gumbel_via_cdf(θ) - ρc

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
        # Last resort without bracketing (difficult if _rho is fine)
        θ0 = 1 + 4ρc/(1 - ρc + eps())
        θ  = Roots.find_zero(f, θ0, Roots.Order1(); xatol=xatol)
        return min(θ, 1e6)  # practical cota...  
    end
end