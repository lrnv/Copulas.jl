"""
    MCopula(d)

The [Upper Frechet-Hoeffding bound](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds) is the copula with the greatest value among all copulas. It correspond to comonotone random vectors. 

For any copula ``C``, if ``W`` and ``M`` are (respectively) the lower and uppder Frechet-Hoeffding bounds, we have that for all ``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u})
```

The two Frechet-Hoeffding bounds are also Archimedean copulas.

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct MCopula{d} <: Copula{d} 
    MCopula(d) = new{d}()
end
Distributions._logpdf(::MCopula{d}, u) where {d} = all(u == u[1]) ? zero(eltype(u)) : eltype(u)(-Inf)
_cdf(::MCopula{d}, u) where {d} = minimum(u)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::MCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    x .= rand(rng)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::MCopula{d}, A::DenseMatrix{T}) where {T<:Real, d}
    A[1,:] .= rand(rng,size(A,2))
    for i in axes(A,1)
        A[i,:] .= A[1,:]
    end
    return A
end
τ(::MCopula) = 1
ρ(::MCopula) = 1
StatsBase.corkendall(::MCopula{d}) where d = ones(d,d)
StatsBase.corspearman(::MCopula{d}) where d = ones(d,d)