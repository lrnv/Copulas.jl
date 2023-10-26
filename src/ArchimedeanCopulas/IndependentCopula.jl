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
IndependentCopula(d) = IndependentCopula{d}()
function Distributions._logpdf(C::IndependentCopula{d},u) where d
    return all(0 .<= u .<= 1) ? zero(eltype(u)) : -Inf
end
function Distributions.cdf(C::IndependentCopula{d},u) where d
    return prod(u)
end
ϕ(::IndependentCopula,t) = exp(-t)
ϕ⁻¹(::IndependentCopula,t) = -log(t)
τ(::IndependentCopula) = 0

# Exceptionally we overload the functions as we dont want to take the slow route of archemedean copulas for the independant copula. 

function Distributions._rand!(rng::Distributions.AbstractRNG, C::IndependentCopula{d}, x::AbstractVector{T}) where {T<:Real,d}
    Random.rand!(rng,x)
end