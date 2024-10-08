"""
    AsymLogCopula{P}

Fields:

  - α::Real - Dependency parameter
  - θ::Vector - Asymmetry parameters (size 2)

Constructor

    AsymLogCopula(α, θ)

The Asymmetric bivariate Logistic copula is parameterized by one dependence parameter ``\\alpha \\in [1, \\infty]`` and two asymmetry parameters ``\\theta_{i} \\in [0,1], i=1,2``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = (\\theta_1^{\\alpha}(1-t)^{\\alpha} + \\theta_2^{\\alpha}t^{\\alpha})^{\\frac{1}{\\alpha}} + (\\theta_1 - \\theta_2)t + 1 - \\theta_1
```

References:
* [tawn1988bivariate](@cite) : Tawn, Jonathan A. "Bivariate extreme value theory: models and estimation." Biometrika 75.3 (1988): 397-415.
"""
struct AsymLogCopula{P} <: ExtremeValueCopula{P}
    α::P  # Dependence Parameter
    θ₁::P
    θ₂::P
    function AsymLogCopula(α::P, θ::Vector{P}) where {P}
        if length(θ) != 2
            throw(ArgumentError("The vector θ must have 2 elements for the bivariate case"))
        elseif !(1 <= α)
            throw(ArgumentError("The parameter α must be greater than or equal to 1"))
        elseif  !(0 <= θ[1] <= 1)  || !(0 <= θ[2] <= 1)  
            throw(ArgumentError("All parameters θ must be in the interval [0, 1]"))
        else
            return new{P}(α, θ[1],θ[2])
        end
    end
end
A(C::AsymLogCopula, t::Real) = ((C.θ₁^C.α)*(1-t)^C.α + (C.θ₂^C.α)*(t^C.α))^(1/C.α)+(C.θ₁- C.θ₂)*t + 1 - C.θ₁