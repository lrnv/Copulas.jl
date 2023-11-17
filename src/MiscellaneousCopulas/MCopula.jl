"""
    MCopula{d}

Constructor

    MCopula(d)

The [Upper Frechet-Hoeffding bound](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds) is the copula with the greatest value among all copulas. It correspond to comonotone random vectors. 

For any copula ``C``, if ``W`` and ``M`` are (respectively) the lower and uppder Frechet-Hoeffding bounds, we have that for all ``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u})
```

The two Frechet-Hoeffding bounds are also Archimedean copulas, although this link is not represetned by the hierachy of types in the package for preformances reasons. 
"""
struct MCopula{d} <: Copula{d} end
MCopula(d) = MCopula{d}()
_cdf(::MCopula{d},u) where {d} = minimum(u)
function Distributions._logpdf(C::MCopula, u)
    return all(u == u[1]) ? zero(eltype(u)) : eltype(u)(-Inf)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::MCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    x .= rand(rng)
end
τ(::MCopula) = 1