"""
    WilliamsonCopula{d,Tϕ,TX}

Fields:
    - ϕ::Tϕ -- a function representing the archimedean generator.
    - X::TX -- a random variable that represents its williamson d-transform

Constructors

    WilliamsonCopula(X::Distributions.UnivariateDistribution, d)
    WilliamsonCopula(ϕ::Function, d)
    WilliamsonCopula(ϕ::Function, X::Distributions.UnivariateDistribution, d)

The WilliamsonCopula is the barebone Archimedean Copula that directly leverages the Williamson transform and inverse transform (in their greatest generalities), that are implemented in [WilliamsonTransforms.jl](https://www.github.com/lrnv/WilliamsonTransforms.jl). You can construct it by providing the Williamson-d-tranform as a (non-negative) random variable `X::Distributions.UnivariateDistribution`, or by providing the ``d``-monotone generator `ϕ::Function` as a function from ``\\mathbb R_+`` to ``[0,1]``, decreasing and d-monotone. The other component will be computed automatically. You also have the option to provide both, which will probably be faster if you have an analytical expression for both. 

About `WilliamsonCopula(X::Distributions.UnivariateDistribution, d)`: For a univariate non-negative random variable ``X``, with cumulative distribution function ``F`` and an integer ``d\\ge 2``, the Williamson-d-transform of ``X`` is the real function supported on ``[0,\\infty[`` given by:

```math
\\phi(t) = 𝒲_{d}(X)(t) = \\int_{t}^{\\infty} \\left(1 - \\frac{t}{x}\\right)^{d-1} dF(x) = \\mathbb E\\left( (1 - \\frac{t}{X})^{d-1}_+\\right) \\mathbb 1_{t > 0} + \\left(1 - F(0)\\right)\\mathbb 1_{t <0}
```

This function has several properties: 
- We have that ``\\phi(0) = 1`` and ``\\phi(Inf) = 0``
- ``\\phi`` is ``d-2`` times derivable, and the signs of its derivatives alternates : ``\\forall k \\in 0,...,d-2, (-1)^k \\phi^{(k)} \\ge 0``.
- ``\\phi^{(d-2)}`` is convex.

These properties makes this function what is called an *archimedean generator*, able to generate *archimedean copulas* in dimensions up to ``d``. 

About `WilliamsonCopula(ϕ::Function, d)`: On the other hand, `WilliamsonCopula(ϕ::Function, d)` Computes the inverse Williamson d-transform of the d-monotone archimedean generator ϕ. 

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
    Williamson, R. E. (1956). Multiply monotone functions and their Laplace transforms. Duke Math. J. 23 189–207. MR0077581
    McNeil, Alexander J., and Johanna Nešlehová. "Multivariate Archimedean copulas, d-monotone functions and ℓ 1-norm symmetric distributions." (2009): 3059-3097.


"""
struct WilliamsonCopula{d,Tϕ,TX} <: ArchimedeanCopula{d}
    ϕ::Tϕ
    X::TX
end
function WilliamsonCopula(X::Distributions.UnivariateDistribution, d)
    ϕ = WilliamsonTransforms.𝒲(X,d)
    return WilliamsonCopula{d,typeof(ϕ),typeof(X)}(ϕ,X)
end
function WilliamsonCopula(ϕ::Function, d)
    X = WilliamsonTransforms.𝒲₋₁(ϕ,d)
    return WilliamsonCopula{d,typeof(ϕ),typeof(X)}(ϕ,X)
end
function WilliamsonCopula(ϕ::Function, X::Distributions.UnivariateDistribution, d)
    return WilliamsonCopula{d,typeof(ϕ),typeof(X)}(ϕ,X)
end
williamson_dist(C::WilliamsonCopula) = C.X
ϕ(C::WilliamsonCopula, t) = C.ϕ(t)