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
* [galambos1975order](@cite) Galambos, J. (1975). Order statistics of samples from multivariate distributions. Journal of the American Statistical Association, 70(351a), 674-680.
"""
struct GalambosCopula{P} <: ExtremeValueCopula{P}
    θ::P  # Copula parameter
    function GalambosCopula(θ)
        if θ > 19.5
            @info "GalambosCopula(θ=$(θ)): Such large θ may lead to numerical issues."
        end
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

A(C::GalambosCopula, t::Real) = -expm1(-LogExpFunctions.logaddexp(-C.θ*log(t),-C.θ*log(1-t))/C.θ)
# This auxiliary function helps determine if we need binary search or not in the generation of random samples
function needs_binary_search(C::GalambosCopula)
    return C.θ > 19.5
end