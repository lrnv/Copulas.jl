"""
    PlackettCopula{P}

Fields:
    - θ::Real - parameter

Constructor

    PlackettCopula(θ)

Parameterized by ``\\theta > 0`` The [Plackett](https://www.cambridge.org/core/books/abs/copulas-and-their-applications-in-water-resources-engineering/plackett-copula/2D407DAB691623AB52CF74044B42C61F) copula is

```math
\\C_{\\theta}(u,v) = \\frac{\\left [1+(\\theta-1)(u+v)\\right]- \\sqrt{[1+(\\theta-1)(u+v)]^2-4uv\\theta(\\theta-1)}}{2(\\theta-1)}
```
and for ``\\theta = 1``
    ```math
\\C_{1}(u,v) = uv ```

It has a few special cases: 
- When θ = 0, is is the MCopula (Upper Frechet-Hoeffding bound)
- When θ = 1, it is the IndependentCopula
- When θ = ∞, is is the WCopula (Lower Frechet-Hoeffding bound)

More details about Plackett copulation are found in Joe, H. (2014). Dependence modeling with copulas. CRC press, Page.164

Details about the algorithm to generate copula samples can be seen in the following references:
    
    Johnson, Mark E. Multivariate statistical simulation:
    A guide to selecting and generating continuous multivariate distributions.
    Vol. 192. John Wiley & Sons, 1987. Page 193.
    Nelsen, Roger B. An introduction to copulas. Springer, 2006. Exercise 3.38.
"""
struct PlackettCopula{P} <: Copula{2} # since it is only bivariate.
    θ::P  # Copula parameter

    function PlackettCopula(θ)
        if θ < 0
            throw(ArgumentError("Theta must be non-negative"))
        elseif θ == 0
            return MCopula(2)
        elseif θ == 1
            return IndependentCopula(2)
        elseif θ == Inf
            return WCopula(2)
        else
            return new{typeof(θ)}(θ)
        end
    end
end

Base.eltype(S::PlackettCopula{P}) where {P} = P # this shuold be P. 

# CDF calculation for bivariate Plackett Copula
function Distributions.cdf(S::PlackettCopula{P}, uv) where {P}
    u, v = uv
    η = S.θ - 1
    term1 = 1 + η * (u + v)
    term2 = sqrt(term1^2 - 4 * S.θ * η * u * v)
    return 0.5 * η^(-1) * (term1 - term2)
end

# PDF calculation for bivariate Plackett Copula
function Distributions._logpdf(S::PlackettCopula{P}, uv) where {P}
    u, v = uv
    η = S.θ - 1
    term1 = S.θ * (1 + η * (u + v - 2 * u * v))
    term2 = (1+η*(u+v))^2-4*(S.θ)*η*u*v
    return log(term1) - 3 * log(term2)/2 # since we are supposed to return the logpdf. 
end
import Random

function Distributions._rand!(rng::Distributions.AbstractRNG, C::CT, x::AbstractVector{T}) where {T<:Real, CT<:PlackettCopula}
    u = rand(rng)
    t = rand(rng)
    a = t * (1 - t)
    b = C.θ + a * (C.θ - 1)^2
    cc = 2a * (u * C.θ^2 + 1 - u) + C.θ * (1 - 2a)
    d = sqrt(C.θ) * sqrt(C.θ + 4a * u * (1 - u) * (1 - C.θ)^2)
    v = (cc - (1 - 2t) * d) / (2b)
    x[1] = u
    x[2] = v
    return x
end

# Calculate Spearman's rho based on the PlackettCopula parameters
function ρ(c::PlackettCopula{P}) where P
    return (c.θ+1)/(c.θ-1)-(2*c.θ*log(c.θ)/(c.θ-1)^2)
end