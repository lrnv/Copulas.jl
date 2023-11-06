"""
    WCopula{d}

Constructor

    WCopula(d)

The [Lower Frechet-Hoeffding bound](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds) is the copula with the lowest value among all copulas. Note that ``W`` is only a proper copula when ``d=1``, in greater dimensions it is still the (pointwise) lower bound, but not a copula anymore. For any copula ``C``, if ``W`` and ``M`` are (respectively) the lower and uppder Frechet-Hoeffding bounds, we have that for all ``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u})
```

The two Frechet-Hoeffding bounds are also Archimedean copulas, although this link is not represetned by the hierachy of types in the package for preformances reasons. 
"""
struct WCopula{d} <: Copula{d} end
WCopula(d) = WCopula{d}()
Distributions.cdf(::WCopula{d},u) where {d} = max(1 + sum(u)-d,0)
function Distributions._rand!(rng::Distributions.AbstractRNG, ::WCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    @assert d==2
    x[1] = rand(rng)
    x[2] = 1-x[1] 
end
τ(::WCopula{d}) where d = -1/(d-1)