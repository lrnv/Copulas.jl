"""
    HuslerReissCopula{P}

Fields:

    - θ::Real - parameter
    
Constructor

    HuslerReissCopula(θ)

The bivariate Husler-Reiss copula is parameterized by ``\\theta \\in [0,\\infty)``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = t\\Phi(\\theta^{-1}+\\frac{1}{2}\\theta\\log(\\frac{t}{1-t})) +(1-t)\\Phi(\\theta^{-1}+\\frac{1}{2}\\theta\\log(\\frac{1-t}{t}))
```
Where ``\\Phi``is the cumulative distribution function (CDF) of the standard normal distribution.

It has a few special cases:

- When θ = 0, it is the Independent Copula
- When θ = ∞, it is the M Copula (Upper Frechet-Hoeffding bound)

References:
* [husler1989maxima](@cite) Hüsler, J., & Reiss, R. D. (1989). Maxima of normal random vectors: between independence and complete dependence. Statistics & Probability Letters, 7(4), 283-286.
"""
struct HuslerReissCopula{P} <: ExtremeValueCopula{P}
    θ::P # Copula parameter
    function HuslerReissCopula(θ)
        if θ < 0
            throw(ArgumentError("Theta must be ≥ 0"))
        elseif θ == 0
            return IndependentCopula(2)
        elseif θ == Inf
            return MCopula(2)
        else
            return new{typeof(θ)}(θ)
        end
    end
end

Distributions.params(C::HuslerReissCopula) = (C.θ)
#  specific ℓ funcion of HuslerReissCopula
function ℓ(H::HuslerReissCopula, t₁, t₂)
    θ = H.θ
    return t₁*Distributions.cdf(Distributions.Normal(),θ^(-1)+0.5*θ*log(t₁/t₂))+t₂*Distributions.cdf(Distributions.Normal(),θ^(-1)+0.5*θ*log(t₂/t₁))
end

# specific A funcion of HuslerReissCopula
function A(H::HuslerReissCopula, t::Real)
    θ = H.θ
    term1 = t * Distributions.cdf(Distributions.Normal(), θ^(-1) + 0.5 * θ * log(t / (1 - t)))
    term2 = (1 - t) * Distributions.cdf(Distributions.Normal(), θ^(-1) + 0.5 * θ * log((1 - t) / t))
    return term1 + term2
end

function dA(H::HuslerReissCopula, t::Real)
    θ = H.θ
    # Derivada of A(x) respected to t
    dA_term1 = Distributions.cdf(Distributions.Normal(), θ^(-1) + 0.5 * θ * log(t / (1 - t))) + 
                  t * Distributions.pdf(Distributions.Normal(), θ^(-1) + 0.5 * θ * log(t / (1 - t))) * (0.5 * θ * (1 / t + 1 / (1 - t)))
                  
    dA_term2 = -Distributions.cdf(Distributions.Normal(), θ^(-1) + 0.5 * θ * log((1 - t) / t)) + 
                  (1 - t) * Distributions.pdf(Distributions.Normal(), θ^(-1) + 0.5 * θ * log((1 - t) / t)) * (0.5 * θ * (-1 / t - 1 / (1 - t)))
    
    return dA_term1 + dA_term2
end