"""
    MGenerator

Constructor

    MGenerator()
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
struct MGenerator <: ZeroVariateGenerator end
max_monotony(::MGenerator) = Inf
τ(::MGenerator) = 1
ϕ(::MGenerator,t) = throw(ArgumentError("MGenerator cannot have a ϕ function")) 
ϕ⁻¹(::MGenerator,t) = throw(ArgumentError("MGenerator cannot have a ϕ⁻¹ function")) 