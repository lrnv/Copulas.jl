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
_available_fitting_methods(::Type{<:ArchimedeanCopula{d,<:GumbelGenerator} where {d}}, d) = (:mle, :itau, :ibeta, :irho)

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

function ϕ⁽ᵏ⁾(G::GumbelGenerator, d::Int, t)
    α = 1 / G.θ
    ntα = -t^α
    return ϕ(G, t) * t^(-d) * sum(
        α^j * Combinatorics.stirlings1(d, j, true) * sum(
            Combinatorics.stirlings2(j, k) * ntα^k for k in 1:j
        ) for j in 1:d
    )
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
function Distributions._logpdf(C::ArchimedeanCopula{2,GumbelGenerator{TF}}, u) where {TF}
    T = promote_type(TF, eltype(u))
    !all(0 .< u .<= 1) && return T(-Inf) # if not in range return -Inf

    θ = C.G.θ
    x₁, x₂ = -log(u[1]), -log(u[2])
    lx₁, lx₂ = log(x₁), log(x₂)
    A = LogExpFunctions.logaddexp(θ * lx₁, θ * lx₂)
    B = exp(A/θ)
    return - B + x₁ + x₂ + (θ-1) * (lx₁ + lx₂) + A/θ - 2A + log(B + θ - 1)
end

_rho_gumbel(θ) = @invoke ρ(GumbelCopula(2, θ)::Copula)
ρ(G::GumbelGenerator) = _rho_gumbel(G.θ)
function ρ⁻¹(::Type{<:GumbelGenerator}, ρ)
    l, u = one(ρ), ρ * Inf
    ρ ≤ 0 && return l
    ρ ≥ 1 && return u
    return Roots.find_zero(θ -> _rho_gumbel(θ) - ρ, (1, Inf))
end
