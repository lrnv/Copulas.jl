"""
    MOCopula{P}

Fields:

    - λ₁::Real - parameter
    - λ₂::Real - parameter
    - λ₁₂::Real - parameter
    
Constructor

    MOCopula(θ)

The bivariate Marshall-Olkin copula is parameterized by ``\\lambda_i \\in [0,\\infty), i = 1, 2, \\{1,2\\}``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = \\frac{\\lambda_1 (1-t)}{\\lambda_1 + \\lambda_{1,2}} + \\frac{\\lambda_2 t}{\\lambda_2 + \\lambda_{1,2}} + \\lambda_{1,2}\\max\\left \\{\\frac{1-t}{\\lambda_1 + \\lambda_{1,2}}, \\frac{t}{\\lambda_2 + \\lambda_{1,2}}  \\right \\} 
```

References:
* [mai2012simulating](@cite) Mai, J. F., & Scherer, M. (2012). Simulating copulas: stochastic models, sampling algorithms, and applications (Vol. 4). World Scientific.
"""
struct MOCopula{P} <: ExtremeValueCopula{P}
    a::P
    b::P
    function MOCopula(λ₁,λ₂,λ₁₂)
        if λ₁ < 0 || λ₂ < 0 || λ₁₂ < 0
            throw(ArgumentError("All λ parameters must be >= 0"))
        end
        a, b = λ₁ / (λ₁ + λ₁₂), λ₂ / (λ₂ + λ₁₂)
        a, b, _ = promote(a, b, 1.0)
        return new{typeof(a)}(a,b)
    end
end
Distributions.params(C::MOCopula) = (C.λ₁, C.λ₂, C.λ₁₂)
A(C::MOCopula, t::Real) = max(t + (1-t)*C.b, (1-t)+C.a*t)
_cdf(C::MOCopula, u::AbstractArray{<:Real}) = min(u[1]^C.a * u[2], u[1] * u[2]^C.b)
function Distributions._rand!(rng::Distributions.AbstractRNG, C::MOCopula, u::AbstractVector{T}) where {T<:Real}
    r, s, t = -log.(rand(rng,3)) # Exponentials(1)
    u[1] = exp(-min(r/(1-C.a), t/C.a))
    u[2] = exp(-min(s/(1-C.b), t/C.b))
    return u
end
τ(C::MOCopula) = C.a*C.b/(C.a+C.b-C.a*C.b)