"""
    IndependentCopula{d}

Constructor

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

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct IndependentCopula{d} <: Copula{d} 
    IndependentCopula(d) = new{d}()
end

_cdf(::IndependentCopula{d}, u) where d = prod(u)
Distributions._logpdf(::IndependentCopula{d}, u) where {d} = all(0 .<= u .<= 1) ? zero(eltype(u)) : eltype(u)(-Inf)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::IndependentCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    Random.rand!(rng,x)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::IndependentCopula{d}, A::DenseMatrix{T}) where {T<:Real, d}
    Random.rand!(rng,A)
    return A
end
rosenblatt(::IndependentCopula{d}, u::AbstractMatrix{<:Real}) where {d} = u
inverse_rosenblatt(::IndependentCopula{d}, u::AbstractMatrix{<:Real}) where {d} = u
τ(::IndependentCopula) = 0
ρ(::IndependentCopula) = 0
StatsBase.corkendall(::IndependentCopula{d}) where d = one(zeros(d,d))
StatsBase.corspearman(::IndependentCopula{d}) where d = one(zeros(d,d))