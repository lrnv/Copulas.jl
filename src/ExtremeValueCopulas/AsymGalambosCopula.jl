"""
    AsymGalambosCopula{P}

Fields:

  - α::Real - Dependency parameter
  - θ::Vector - Asymmetry parameters (size 2)

Constructor

    AsymGalambosCopula(α, θ)

The Asymmetric bivariate Galambos copula is parameterized by one dependence parameter ``\\alpha \\in [0, \\infty]`` and two asymmetry parameters ``\\theta_{i} \\in [0,1], i=1,2``. It is an Extreme value copula with Pickands function: 

```math
\\A(t) = 1 - ((\\theta_1t)^{-\\alpha}+(\\theta_2(1-t))^{-\\alpha})^{-\\frac{1}{\\alpha}} 
```

It has a few special cases:

- When α = 0, it is the Independent Copula
- When θ₁ = θ₂ = 0, it is the Independent Copula
- When θ₁ = θ₂ = 1, it is the Galambos Copula

References:
* [Joe1990](@cite) Families of min-stable multivariate exponential and multivariate extreme value distributions. Statist. Probab, 1990.
"""
struct AsymGalambosCopula{P} <: ExtremeValueCopula{P}
    α::P  # Dependence parameter
    θ::Vector{P}  # Asymmetry parameters (size 2)
    function AsymGalambosCopula(α::P, θ::Vector{P}) where {P}
        if length(θ) != 2
            throw(ArgumentError("The vector θ must have 2 elements for the bivariate case"))
        elseif !(0 <= α)
            throw(ArgumentError("The parameter α must be greater than or equal to 0"))
        elseif  !(0 <= θ[1] <= 1)  || !(0 <= θ[2] <= 1)  
            throw(ArgumentError("All parameters θ must be in the interval [0, 1]"))
        elseif α == 0 || (θ[1] == 0 && θ[2] == 0)
            return IndependentCopula(2)
        elseif θ[1] == 1 && θ[2] == 1
            return GalambosCopula(α)
        else
            return new{P}(α, θ)
        end
    end
end

function 𝘈(C::AsymGalambosCopula, t::Real)
    α = C.α
    θ = C.θ

    term1 = (θ[1] * t)^(-α)
    term2 = (θ[2] * (1 - t))^(-α)
    
    inner_term = term1 + term2

    result = 1 - inner_term^(-1 / α)
    return result
end