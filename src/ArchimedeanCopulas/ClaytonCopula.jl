"""
    ClaytonCopula{d,T}

Fields:
  - θ::Real - parameter

Constructor

    ClaytonCopula(d, θ)

The [Clayton](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) copula in dimension ``d`` is parameterized by ``\\theta \\in [-1/(d-1),\\infty)``. It is an Archimedean copula with generator : 

```math
\\phi(t) = \\left(1+\\mathrm{sign}(\\theta)*t\\right)^{-1\\frac{1}{\\theta}}
```

It has a few special cases: 
- When θ = -1/(d-1), it is the WCopula (Lower Frechet-Hoeffding bound)
- When θ = 0, it is the IndependentCopula
- When θ = ∞, is is the MCopula (Upper Frechet-Hoeffding bound)
"""
const ClaytonCopula{d,T} = ArchimedeanCopula{d,ClaytonGenerator{T}}
ClaytonCopula(d,θ) = ArchimedeanCopula(d,ClaytonGenerator(θ))
τ(C::ClaytonCopula) = ifelse(isfinite(C.θ), C.θ/(C.θ+2), 1)
τ⁻¹(::Type{ClaytonCopula},τ) = ifelse(τ == 1,Inf,2τ/(1-τ))