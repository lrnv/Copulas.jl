"""
    MOCopula{P}

Fields:

    - λ1::Real - parameter
    - λ2::Real - parameter
    - λ12::Real - parameter
    
Constructor

    MOCopula(θ)

The bivariate Marshall-Olkin copula is parameterized by ``\\lambda_i \\in [0,\\infty), i = 1, 2, \\{1,2\\}``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = \\frac{\\lambda_1 (1-t)}{\\lambda_1 + \\lambda_{1,2}} + \\frac{\\lambda_2 t}{\\lambda_2 + \\lambda_{1,2}} + \\lambda_{1,2}\\max\\left \\{\\frac{1-t}{\\lambda_1 + \\lambda_{1,2}}, \\frac{t}{\\lambda_2 + \\lambda_{1,2}}  \\right \\} 
```

References:
* Simulating copulas: stochastic models, sampling algorithms, and applications. 2017.
"""
struct MOCopula{P} <: ExtremeValueCopula{P}
    λ1::P
    λ2::P
    λ12::P

    function MOCopula(λ::Vararg{Real})
        if length(λ) !== 3
            throw(ArgumentError("MOCopula requires only 3 arguments."))
        end

        T = promote_type(typeof(λ[1]), typeof(λ[2]), typeof(λ[3]))
        λ1, λ2, λ12 = T(λ[1]), T(λ[2]), T(λ[3])

        if λ1 < 0 || λ2 < 0 || λ12 < 0
            throw(ArgumentError("All λ parameters must be >= 0"))
        end
        
        return new{T}(λ1, λ2, λ12)
    end
end

function 𝘈(C::MOCopula, t::Real)
    λ1, λ2, λ12 = C.λ1, C.λ2, C.λ12
    return max(t + (1-t) * (λ2)/(λ2 + λ12), (t-1) + t * (λ1)/(λ1 + λ12))
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::MOCopula, u::AbstractVector{T}) where {T<:Real}
    λ1, λ2, λ12 = C.λ1, C.λ2, C.λ12
    r, s, t = rand(rng, Distributions.Uniform(0,1),3)
    x = min(-log(r)/λ1, -log(t)/λ12)
    y = min(-log(s)/λ2, -log(t)/λ12)
    u[1] = exp(-(λ1+λ12)*x)
    u[2] = exp(-(λ2+λ12)*y)
    return u
end
