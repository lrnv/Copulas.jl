"""
    BC2Copula{P}

Fields:

    - θ1::Real - parameter
    - θ1::Real - parameter
    
Constructor

    BC2Copula(θ1, θ2)

The bivariate BC₂ copula is parameterized by two parameters ``\\theta_{i} \\in [0,1], i=1,2``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = \\max\\{\\theta_1 t, \\theta_2(1-t) \\} + \\max\\{(1-\\theta_1)t, (1-\\theta_2)(1-t)\\}
```

References:
* Bivariate extreme-value copulas with discrete Pickands dependence measure. Springer, 2011.
"""
struct BC2Copula{P} <: ExtremeValueCopula{P}
    θ1::P
    θ2::P

    function BC2Copula(θ::Vararg{Real})
        if length(θ) !== 2
            throw(ArgumentError("BC2Copula requires only 2 arguments."))
        end
        T = promote_type(typeof(θ[1]), typeof(θ[2]))
        θ1, θ2 = T(θ[1]), T(θ[2])
        
        if !(0 <= θ1 <= 1) || !(0 <= θ2 <= 1) 
            throw(ArgumentError("All θ parameters must be in [0,1]"))
        end
        return new{T}(θ1, θ2)
    end
end

function ℓ(C::BC2Copula, t::Vector)
    θ1, θ2 = C.θ1, C.θ2
    t₁, t₂ = t
    return max(θ1*t₁, θ2*t₂) + max((1-θ1)*t₁, (1-θ2)*t₂)
end

function 𝜜(C::BC2Copula, t::Real)
    θ1, θ2 = C.θ1, C.θ2
    return max(θ1*t, θ2*(1-t)) + max((1-θ1)*t, (1-θ2)*(1-t))
end

function d𝘈(C::BC2Copula, t::Float64)
    θ1, θ2 = C.θ1, C.θ2
    
    # Conditions for the derivative of the first part
    if θ1*t >= θ2*(1-t)
        f1_derivative = θ1
    else
        f1_derivative = -θ2
    end

    # Conditions for the derivative of the second part
    if (1-θ1)*t >= (1-θ2)*(1-t)
        f2_derivative = 1-θ1
    else
        f2_derivative = -(1-θ2)
    end

    return f1_derivative + f2_derivative
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::BC2Copula, u::AbstractVector{T}) where {T<:Real}
    θ1, θ2 = C.θ1, C.θ2
    v1, v2 = rand(rng, Distributions.Uniform(0,1), 2)
    u[1] = max(v1^(1/θ1),v2^(1/(1-θ1)))
    u[2] = max(v1^(1/θ2),v2^(1/(1-θ2)))
    return u
end