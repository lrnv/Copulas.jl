"""
    GalambosCopula{P}

Fields:

    - θ::Real - parameter
    
Constructor

    GalambosCopula(θ)

The bivariate Galambos copula is parameterized by ``\\alpha \\in [0,\\infty)``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = 1 - (t^{-\\theta}+(1-t)^{-\\theta})^{-\\frac{1}{\\theta}}
```

It has a few special cases:

- When θ = 0, it is the Independent Copula
- When θ = ∞, it is the M Copula (Upper Frechet-Hoeffding bound)

References:
* Order statistics of samples from multivariate distributions. J. Amer. Statist Assoc. 1975.
"""
struct GalambosCopula{P} <: ExtremeValueCopula{P}
    θ::P  # Copula parameter
    function GalambosCopula(θ)
        if θ < 0
            throw(ArgumentError("Theta must be >= 0"))
        elseif θ == 0
            return IndependentCopula(2)
        elseif θ == Inf
            return MCopula(2)
        else
            return new{typeof(θ)}(θ)
        end
    end
end

𝘈(C::GalambosCopula, t::Real) = 1 - (t^(-C.θ) + (1 - t)^(-C.θ))^(-1/C.θ)