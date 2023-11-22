"""
    IndependentGenerator

Constructor

    IndependentGenerator()
    IndependentCopula(d)

The [Independent Copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Most_important_Archimedean_copulas) in dimension ``d`` is
the simplest copula, that has the form : 

```math
C(\\mathbf{x}) = \\prod_{i=1}^d x_i.
```

It happends to be an Archimedean Copula, with generator : 

```math
\\phi(t) = \\exp{-t}
```
"""
struct IndependentGenerator <: ZeroVariateGenerator end
max_monotony(::IndependentGenerator) = Inf
ϕ(::IndependentGenerator,t) = exp(-t)
ϕ⁻¹(::IndependentGenerator,t) = -log(t)
τ(::IndependentGenerator) = 0