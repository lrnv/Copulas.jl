"""
    WGenerator

Constructor

    WGenerator()
    WCopula(d)

The [Lower Frechet-Hoeffding bound](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds) is the copula with the lowest value among all copulas. Note that ``W`` is only a proper copula when ``d=2``, in greater dimensions it is still the (pointwise) lower bound, but not a copula anymore. For any copula ``C``, if ``W`` and ``M`` are (respectively) the lower and uppder Frechet-Hoeffding bounds, we have that for all ``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u})
```

The two Frechet-Hoeffding bounds are also Archimedean copulas.
"""
struct WGenerator <: ZeroVariateGenerator end
max_monotony(G::WGenerator) = 2
τ(::WGenerator) = -1
ϕ(::WGenerator,t) = throw(ArgumentError("WGenerator cannot have a ϕ function")) 
ϕ⁻¹(::WGenerator,t) = throw(ArgumentError("WGenerator cannot have a ϕ⁻¹ function")) 
