"""
    AsymMixedCopula{P}

Fields:

  - θ::Vector - parameters (size 2)

Constructor

    AsymMixedCopula(θ)

The Asymmetric bivariate Mixed copula is parameterized by two parameters ``\\theta_{i}, i=1,2`` that must meet the following conditions:
* θ₁ ≥ 0
* θ₁+θ₂ ≤ 1
* θ₁+2θ₂ ≤ 1
* θ₁+3θ₂ ≥ 0

It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = \\theta_{2}t^3 + \\theta_{1}t^2-(\\theta_1+\\theta_2)t+1
```

It has a few special cases:

- When θ₁ = θ₂ = 0, it is the Independent Copula
- When θ₂ = 0, it is the Mixed Copula

References:
* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct AsymMixedCopula{P} <: ExtremeValueCopula{P}
    θ::Vector{P}  # Asymmetry parameters

    function AsymMixedCopula(θ::Vector{P}) where {P}
        if length(θ) != 2
            throw(ArgumentError("The vector θ must have 2 elements for the bivariate case"))
        elseif !(0 <= θ[1])
            throw(ArgumentError("The parameter θ₁ must be greater than or equal to 0"))
        elseif  !(θ[1]+θ[2] <= 1) 
            throw(ArgumentError("the sum of θ₁+θ₂ ≤ 1"))
        elseif !(θ[1]+2*θ[2] <= 1)
            throw(ArgumentError("the sum of θ₁+2θ₂ ≤ 1"))
        elseif !(0 <= θ[1]+3*θ[2])
            throw(ArgumentError("the sum of 0 ≤ θ₁+3θ₂"))
        elseif θ[1] == 0 && θ[2] == 0
            return IndependentCopula(2)
        else
            return new{P}(θ)
        end
    end
end

function A(C::AsymMixedCopula, t::Real)
    θ₁, θ₂ = C.θ[1], C.θ[2]
    return θ₂*t^3 + θ₁*t^2 - (θ₁+θ₂)*t + 1
end