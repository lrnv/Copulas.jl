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
const FrankCopula{d,T} = ArchimedeanCopula{d,FrankGenerator{T}}
FrankCopula(d,θ) = ArchimedeanCopula(d,FrankGenerator(θ))

D₁ = GSL.sf_debye_1 # sadly, this is C code.
# could be replaced by : 
# using QuadGK
# D₁(x) = quadgk(t -> t/(exp(t)-1), 0, x)[1]/x
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