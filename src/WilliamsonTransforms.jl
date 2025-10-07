"""
    𝒲(X,d)(x)

Computes the Williamson d-transform of the random variable X, taken at point x.

For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and an integer ``d\\ge 2``, the Williamson-d-transform of ``X`` is the real function supported on ``[0,\\infty[`` given by:

```math
\\phi(t) = 𝒲_{d}(X)(t) = \\int_{t}^{\\infty} \\left(1 - \\frac{t}{x}\\right)^{d-1} dF(x) = \\mathbb E\\left( (1 - \\frac{t}{X})^{d-1}_+\\right) \\mathbb 1_{t > 0} + \\left(1 - F(0)\\right)\\mathbb 1_{t <0}
```

This function has several properties: 
    - We have that ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
    - ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
    - ``\\phi^{(d-2)}`` is convex.

These properties makes this function what is called an *archimedean generator*, able to generate *archimedean copulas* in dimensions up to ``d``. 

References: 
- Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
- McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct 𝒲{TX, d}
    X::TX
    function 𝒲(X::TX, d::Int) where {TX<:Distributions.UnivariateDistribution}
        @assert Base.minimum(X) ≥ 0 && Base.maximum(X) ≤ Inf 
        @assert d ≥ 2 && isinteger(d) 
        return new{typeof(X), d}(X)
    end
end
function (ϕ::𝒲{TX, d})(x) where {TX,d}
    x <= 0 && return 1 - Distributions.cdf(ϕ.X,0)
    return Distributions.expectation(y -> (1 - x/y)^(d-1) * (y > x), ϕ.X)
end

"""
    𝒲₋₁(ϕ,d)


Computes the inverse Williamson d-transform of the d-monotone archimedean generator ϕ. 

A ``d``-monotone archimedean generator is a function ``\\phi`` on ``\\mathbb R_+`` that has these three properties:
- ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

For such a function ``\\phi``, the inverse Williamson-d-transform of ``\\phi`` is the cumulative distribution function ``F`` of a non-negative random variable ``X``, defined by : 

```math
F(x) = 𝒲_{d}^{-1}(\\phi)(x) = 1 - \\frac{(-x)^{d-1} \\phi_+^{(d-1)}(x)}{k!} - \\sum_{k=0}^{d-2} \\frac{(-x)^k \\phi^{(k)}(x)}{k!}
```

We return this cumulative distribution function in the form of the corresponding random variable `<:Distributions.ContinuousUnivariateDistribution` from `Distributions.jl`. You may then compute : 
    - The cdf via `Distributions.cdf`
    - The pdf via `Distributions.pdf` and the logpdf via `Distributions.logpdf`
    - Samples from the distribution via `rand(X,n)`

References: 
    - Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
    - McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.
"""
struct 𝒲₋₁{Tϕ, d} <: Distributions.ContinuousUnivariateDistribution
    # Woul dprobably be much more efficient if it took the generator and not the function itself. 
    ϕ::Tϕ
    function 𝒲₋₁(ϕ, d::Int)
        @assert ϕ(0.0) == 1.0
        @assert ϕ(float(Inf)) == 0.0
        @assert isinteger(d)
        return new{typeof(ϕ),d}(ϕ)
    end
end
function Distributions.cdf(dist::𝒲₋₁{Tϕ, d}, x) where {Tϕ, d}
    x ≤ 0 && return zero(x)
    rez, x_pow = zero(x), one(x)
    c = taylor(dist.ϕ, x, d-1)
    for k in 1:d
        rez += iszero(c[k]) ? 0 : x_pow * c[k]
        x_pow *= -x
    end
    return isnan(rez) ? one(x) : 1 - rez
end

Distributions.logpdf(dist::𝒲₋₁{Tϕ, d}, x) where {Tϕ, d} = log(max(0, taylor(x -> Distributions.cdf(dist,x), x, 1)[end]))
_quantile(dist::𝒲₋₁, p) = Roots.find_zero(x -> (Distributions.cdf(dist, x) - p), (0.0, Inf))
Distributions.rand(rng::Distributions.AbstractRNG, dist::𝒲₋₁) = _quantile(dist, rand(rng))
Base.minimum(::𝒲₋₁) = 0.0
Base.maximum(::𝒲₋₁) = Inf
function Distributions.quantile(dist::𝒲₋₁, p::Real)
    # Validate that p is in the range [0, 1]
    @assert 0 <= p <= 1
    return _quantile(dist, p)
end


