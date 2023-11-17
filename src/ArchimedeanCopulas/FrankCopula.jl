"""
    FrankCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    FrankCopula(d, θ)

The [Frank](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-\\infty,\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = -\\frac{\\log\\left(1+e^{-t}(e^{-\\theta-1})\\right)}{\theta}
```

It has a few special cases: 
- When θ = -∞, it is the WCopula (Lower Frechet-Hoeffding bound)
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)
"""
struct FrankCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function FrankCopula(d,θ)
        if d > 2 && θ < 0
            throw(ArgumentError("Negatively dependent Frank copulas cannot exists in dimensions > 2"))
        end
        if θ == -Inf
            return WCopula(d)
        elseif θ == 0
            return IndependentCopula(d)
        elseif θ == Inf
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::FrankCopula,       t) = C.θ > 0 ? -LogExpFunctions.log1mexp(LogExpFunctions.log1mexp(-C.θ)-t)/C.θ : -log1p(exp(-t) * expm1(-C.θ))/C.θ
ϕ⁻¹(C::FrankCopula,       t) = C.θ > 0 ? LogExpFunctions.log1mexp(-C.θ) - LogExpFunctions.log1mexp(-t*C.θ) : -log(expm1(-t*C.θ)/expm1(-C.θ))

# A bit of type piracy but should be OK : 
# LogExpFunctions.log1mexp(t::TaylorSeries.Taylor1) = log(-expm1(t))
# Avoid type piracy by defiing it myself: 
ϕ(  C::FrankCopula,       t::TaylorSeries.Taylor1) = C.θ > 0 ? -log(-expm1(LogExpFunctions.log1mexp(-C.θ)-t))/C.θ : -log1p(exp(-t) * expm1(-C.θ))/C.θ

D₁ = GSL.sf_debye_1 # sadly, this is C code.
# could be replaced by : 
# using QuadGK
# D₁(x) = quadgk(t -> t/(exp(t)-1), 0, x)[1]/x
# to make it more general. but once gain, it requires changing the integrator at each evlauation, 
# which is problematic. 
# Better option is to try to include this function into SpecialFunctions.jl. 


τ(C::FrankCopula) = 1+4(D₁(C.θ)-1)/C.θ
function τ⁻¹(::Type{FrankCopula},τ)
    if τ == zero(τ)
        return τ
    end
    if abs(τ==1)
        return Inf * τ
    end
    x₀ = (1-τ)/4
    return Roots.fzero(x -> (1-D₁(x))/x - x₀, 1e-4, Inf)
end
    
williamson_dist(C::FrankCopula{d,T}) where {d,T} = C.θ > 0 ?  WilliamsonFromFrailty(Logarithmic(-C.θ), d) : WilliamsonTransforms.𝒲₋₁(t -> ϕ(C,t),d)

