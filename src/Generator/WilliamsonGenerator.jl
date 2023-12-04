"""
    WilliamsonGenerator{TX}
    i𝒲{TX}

Fields:
* `X::TX` -- a random variable that represents its williamson d-transform
* `d::Int` -- the dimension of the transformation. 

Constructor

    WilliamsonGenerator(X::Distributions.UnivariateDistribution, d)
    i𝒲(X::Distributions.UnivariateDistribution,d)

The `WilliamsonGenerator` (alias `i𝒲`) allows to construct a d-monotonous archimedean generator from a positive random variable `X::Distributions.UnivariateDistribution`. The transformation, wich is called the inverse williamson transformation, is implemented in [WilliamsonTransforms.jl](https://www.github.com/lrnv/WilliamsonTransforms.jl). 

For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and an integer ``d\\ge 2``, the Williamson-d-transform of ``X`` is the real function supported on ``[0,\\infty[`` given by:

```math
\\phi(t) = 𝒲_{d}(X)(t) = \\int_{t}^{\\infty} \\left(1 - \\frac{t}{x}\\right)^{d-1} dF(x) = \\mathbb E\\left( (1 - \\frac{t}{X})^{d-1}_+\\right) \\mathbb 1_{t > 0} + \\left(1 - F(0)\\right)\\mathbb 1_{t <0}
```

This function has several properties: 
- We have that ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

These properties makes this function what is called a *d-monotone archimedean generator*, able to generate *archimedean copulas* in dimensions up to ``d``. Our implementation provides this through the `Generator` interface: the function ``\\phi`` can be accessed by 

    G = WilliamsonGenerator(X, d)
    ϕ(G,t)

Note that you'll always have:

    max_monotony(WilliamsonGenerator(X,d)) === d

References: 
* [williamson1955multiply](@cite) Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
* [mcneil2009](@cite) McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct WilliamsonGenerator{TX} <: Generator
    X::TX
    d::Int
    function WilliamsonGenerator(X,transform_dimension)
        # check that X is indeed a positively supported random variable... 
        return new{typeof(X)}(X,transform_dimension)
    end
end
const i𝒲 = WilliamsonGenerator
max_monotony(G::WilliamsonGenerator) = G.d
function williamson_dist(G::WilliamsonGenerator, d)
    if d == G.d 
        return G.X
    end
    # what about d < G.d ? Mayeb we can do some frailty stuff ? 
    return WilliamsonTransforms.𝒲₋₁(t -> ϕ(G,t),d)
end
ϕ(G::WilliamsonGenerator, t) = WilliamsonTransforms.𝒲(G.X,G.d)(t)