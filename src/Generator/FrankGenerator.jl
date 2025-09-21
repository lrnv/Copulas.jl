"""
    FrankGenerator{T}

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
FrankCopula(d, θ) = ArchimedeanCopula(d, FrankGenerator(θ))
FrankCopula(d; θ::Real) = FrankCopula(d, θ)
max_monotony(G::FrankGenerator) = G.θ < 0 ? 2 : Inf

Distributions.params(G::FrankGenerator) = (θ = G.θ,)
_example(CT::Type{<:FrankCopula}, d) = FrankCopula(d, 1.0)
_unbound_params(::Type{<:FrankCopula}, d, θ) = d == 2 ? [θ.θ] : [log(θ.θ)]
_rebound_params(::Type{<:FrankCopula}, d, α) = d==2 ? (; θ = α[1]) : (; θ = exp(α[1]))
_θ_bounds(::Type{<:FrankGenerator}, d) = d==2 ? (-Inf, Inf) : (0, Inf)


ϕ(G::FrankGenerator, t) = G.θ > 0 ? -LogExpFunctions.log1mexp(LogExpFunctions.log1mexp(-G.θ)-t)/G.θ : -log1p(exp(-t) * expm1(-G.θ))/G.θ
ϕ⁽¹⁾(G::FrankGenerator, t) = (1 - 1 / (1 + exp(-t)*expm1(-G.θ))) / G.θ
ϕ⁻¹⁽¹⁾(G::FrankGenerator, t) = G.θ / (-expm1(G.θ * t))
function ϕ⁽ᵏ⁾(G::FrankGenerator, ::Val{k}, t) where k
    return (-1)^k * (1 / G.θ) * PolyLog.reli(-(k - 1), -expm1(-G.θ) * exp(-t))
end
ϕ⁻¹(G::FrankGenerator, t) = G.θ > 0 ? LogExpFunctions.log1mexp(-G.θ) - LogExpFunctions.log1mexp(-t*G.θ) : -log(expm1(-t*G.θ)/expm1(-G.θ))
williamson_dist(G::FrankGenerator, ::Val{d}) where d = G.θ > 0 ? WilliamsonFromFrailty(Logarithmic(-G.θ), Val{d}()) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),Val{d}())
frailty(G::FrankGenerator) = G.θ > 0 ? Logarithmic(-G.θ) : throw("The frank copula has no frailty when θ < 0")
Debye(x, k::Int=1) = k / x^k * QuadGK.quadgk(t -> t^k/expm1(t), 0, x)[1]
function _frank_tau(θ)
    T = promote_type(typeof(θ),Float64)
    if abs(θ) < sqrt(eps(T))
        # return the taylor approx.
        return θ/9 * (1 - (θ/10)^2)
    else
        return 1+4(Debye(θ,1)-1)/θ
    end
end
τ(G::FrankGenerator) = _frank_tau(G.θ)
function τ⁻¹(::Type{T},tau) where T<:FrankGenerator
    s,v = sign(tau),abs(tau)
    if v == 0
        return v
    elseif v == 1
        return s * Inf
    else
        return s*Roots.fzero(x -> _frank_tau(x)-v, 0, Inf)
    end
end

function ρ(G::FrankGenerator)
    θ = G.θ
    (-Inf < θ < Inf) || throw(ArgumentError("Frank definido para θ∈ℝ\\{0}"))
    abs(θ) < 1e-8 && return θ/6
    return 1 + 12*(Debye(θ,2) - Debye(θ,1))/θ
end

function ρ⁻¹(::Type{FrankGenerator}, ρ̂::Real; tol::Real=1e-10)
    ρc = clamp(ρ̂, -1+1e-12, 1-1e-12)

    f(θ) = ρ(FrankGenerator(θ)) - ρc

    # bracketing adaptative
    # for ρ>0 search θ>0, for ρ<0 θ<0
    if ρc > 0
        a, b = 1e-6, 50.0
        while f(a)*f(b) > 0 && b < 1e6
            b *= 2
        end
    else
        a, b = -50.0, -1e-6
        while f(a)*f(b) > 0 && a > -1e6
            a *= 2
        end
    end

    return Roots.find_zero(f, (a,b), Roots.Brent(); xatol=tol, rtol=0)
end
