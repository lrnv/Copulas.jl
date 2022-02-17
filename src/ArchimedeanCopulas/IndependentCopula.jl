"""
IndependentCopula{d,T}

Constructor

    IndependentCopula(d, θ)

The [Independant Copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) in dimension ``d`` is
the simplest copula, that has the form : 

```math
C(\\mathbf{x}) = \\prod_{i=1}^d x_i.
```

It happends to be an Archimedean Copula, with generator : 

```math
\\phi(t) = \\exp{-t}
```
"""
struct IndependentCopula{d} <: ArchimedeanCopula{d} end
IndependentCopula(d) = IndependentCopula{d}
function Distributions._logpdf(::IndependentCopula{d},u) where d
    return all(0 .<= u .<= 1) ? 1 : 0
end
function Distributions.cdf(::IndependentCopula{d},u) where d
    return all(0 .<= u .<= 1) ? prod(u) : 0
end
ϕ(::IndependentCopula,t) = exp(-t)
ϕ⁻¹(::IndependentCopula,t) = -log(t)
τ(::IndependentCopula) = 0
