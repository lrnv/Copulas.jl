"""
    FrailtyGenerator{TX}

Fields:
* `X::TX` -- a random variable that represents the frailty distribution.

Constructor

    FrailtyGenerator(X::Distributions.UnivariateDistribution)

The `FrailtyGenerator` allows to construct a completely monotonous archimedean generator from a positive random variable `X::Distributions.UnivariateDistribution`, assuming the distribution has a completely monotonous laplace transform (which will be used as the generator).

References: 
* [williamson1955multiply](@cite) Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
* [mcneil2009](@cite) McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
* [mcneil2008](@cite) M
"""
struct FrailtyGenerator{TX} <: Generator
    X::TX
    function FrailtyGenerator(X)
        # Check that X is indeed a positively supported random variable
        @assert minimum(X) ≥ 0 
        return new{typeof(X)}(X)
    end
end
max_monotony(::FrailtyGenerator) = Inf
ϕ(G::FrailtyGenerator, t) = Distributions.mgf(G.X,t)
williamson_dist(G::FrailtyGenerator, d) = WilliamsonFromFrailty(G.X,d)

