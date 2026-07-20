"""
    FrankGenerator{T}, FrankCopula{d, T}

Fields:
  - θ::Real - parameter

Constructor

    FrankGenerator(θ)
    FrankCopula(d,θ)

The [Frank](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in (-\\infty,\\infty)`` (with independence as the limit ``\\theta\\to 0``). It is an Archimedean copula with generator

```math
\\phi(t) = -\\tfrac{1}{\\theta} \\log\\big( 1 - (1 - e^{-\\theta}) e^{-t} \\big).
```

Special cases:
- When ``\\theta \\to -\\infty``, it is the WCopula (Lower Fréchet–Hoeffding bound)
- When ``\\theta \\to 0``, it is the IndependentCopula
- When ``\\theta \\to \\infty``, it is the MCopula (Upper Fréchet–Hoeffding bound)

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
FrankGenerator, FrankCopula

struct FrankGenerator{T} <: AbstractUnivariateGenerator
    θ::T
    function FrankGenerator(θ)
        if θ == -Inf
            return WGenerator()
        elseif θ == 0
            return IndependentGenerator()
        elseif θ == Inf
            return MGenerator()
        else
            θ, _ = promote(θ, 1.0)
            return new{typeof(θ)}(θ)
        end
    end
end
const FrankCopula{d, T} = ArchimedeanCopula{d, FrankGenerator{T}}
max_monotony(G::FrankGenerator) = G.θ < 0 ? 2 : Inf
Distributions.params(G::FrankGenerator) = (θ = G.θ,)
_unbound_params(::Type{<:FrankGenerator}, d, θ) = d == 2 ? [θ.θ] : [log(θ.θ)]
_rebound_params(::Type{<:FrankGenerator}, d, α) = d==2 ? (; θ = α[1]) : (; θ = exp(α[1]))
_θ_bounds(::Type{<:FrankGenerator}, d) = d==2 ? (-Inf, Inf) : (0, Inf)


ϕ(G::FrankGenerator, t) = G.θ > 0 ? -LogExpFunctions.log1mexp(LogExpFunctions.log1mexp(-G.θ)-t)/G.θ : -log1p(exp(-t) * expm1(-G.θ))/G.θ
ϕ⁽¹⁾(G::FrankGenerator, t) = (1 - 1 / (1 + exp(-t)*expm1(-G.θ))) / G.θ
ϕ⁻¹⁽¹⁾(G::FrankGenerator, t) = G.θ / (-expm1(G.θ * t))
function ϕ⁽ᵏ⁾(G::FrankGenerator, k::Int, t)
    return (-1)^k * (1 / G.θ) * PolyLog.reli(-(k - 1), -expm1(-G.θ) * exp(-t))
end
function ϕ⁽ᵏ⁾⁻¹(G::FrankGenerator, k::Int, t; start_at=t)
    k == 1 || return @invoke ϕ⁽ᵏ⁾⁻¹(G::Generator, k, t; start_at=start_at)
    T = float(promote_type(typeof(t), typeof(G.θ)))
    target = T(t)
    iszero(target) && return T(Inf)
    θ = T(G.θ)
    z = θ * target
    return log(abs(expm1(-θ))) + log1p(-z) - log(abs(z))
end
ϕ⁻¹(G::FrankGenerator, t) = G.θ > 0 ? LogExpFunctions.log1mexp(-G.θ) - LogExpFunctions.log1mexp(-t*G.θ) : -log(expm1(-t*G.θ)/expm1(-G.θ))

# Taylor1-compatible ϕ / ϕ⁻¹: the θ>0 scalar forms above use `log1mexp`, which has
# no `TaylorSeries.Taylor1` method, so the nested direct edge composition (a jet
# over ϕ⁻¹_outer ∘ ϕ_inner) would fail when Frank appears in the tree. These add
# the equivalent θ≠0 closed forms built only from `expm1`/`log1p` (all Taylor1-
# compatible) — exactly the θ<0 branch forms, on a Taylor1 argument; the scalar
# path is untouched. (A future TaylorSeries⊕LogExpFunctions extension giving
# `log1mexp(::Taylor1)` would make these unnecessary; the implicit edge-composition
# method already avoids them, using only the closed-form ϕ⁽ᵏ⁾.)
ϕ(G::FrankGenerator, t::TaylorSeries.Taylor1{T}) where {T} = -log1p(exp(-t) * expm1(-T(G.θ))) / T(G.θ)
ϕ⁻¹(G::FrankGenerator, t::TaylorSeries.Taylor1{T}) where {T} = -log(expm1(-T(G.θ) * t) / expm1(-T(G.θ)))
𝒲₋₁(G::FrankGenerator, d::Int) = G.θ > 0 ? WilliamsonFromFrailty(Logarithmic(-G.θ), d) : @invoke 𝒲₋₁(G::Generator, d)
frailty(G::FrankGenerator) = G.θ > 0 ? Logarithmic(-G.θ) : throw("The frank copula has no frailty when θ < 0")

Debye(x, k::Int=1) = k / x^k * QuadGK.quadgk(t -> t^k/expm1(t), 0, x)[1]

function _frank_tau(θ)
    T = promote_type(typeof(θ),Float64)
    abs(θ) < sqrt(eps(T)) && return θ/9 * (1 - (θ/10)^2)
    return 1+4(Debye(θ,1)-1)/θ
end
τ(G::FrankGenerator) = _frank_tau(G.θ)
function τ⁻¹(::Type{<:FrankGenerator}, tau)
    s, v = sign(tau), abs(tau)
    v == 0 && return v
    v == 1 && return s * Inf
    emθ = Roots.fzero(x -> _frank_tau(-log(x))-v, 0, 1)
    return - s * log(emθ)
end
function _frank_rho(θ)
    T = promote_type(typeof(θ),Float64)
    abs(θ) < sqrt(eps(T)) && return θ/6
    return 1 + 12*(Debye(θ,2) - Debye(θ,1))/θ
end
ρ(G::FrankGenerator) = _frank_rho(G.θ)
function ρ⁻¹(::Type{<:FrankGenerator}, rho)
    s, v = sign(rho), abs(rho)
    v == 0 && return v
    v == 1 && return s * Inf
    emθ = Roots.fzero(x -> _frank_rho(-log(x))-v, 0, 1)
    return - s * log(emθ)
end
