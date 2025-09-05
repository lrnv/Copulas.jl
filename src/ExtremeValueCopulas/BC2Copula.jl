"""
    BC2Copula{P}

Fields:

    - a::Real - parameter
    - a::Real - parameter
    
Constructor

    BC2Copula(a, b)

The bivariate BC2 copula is parameterized by two parameters ``a,b \\in [0,1]``. It is an Extreme value copula with Pickands dependence function: 

```math
A(t) = \\max\\{a t, b (1-t) \\} + \\max\\{(1-a)t, (1-b)(1-t)\\}
```

References:
* [mai2011bivariate](@cite) Mai, J. F., & Scherer, M. (2011). Bivariate extreme-value copulas with discrete Pickands dependence measure. Extremes, 14, 311-324. Springer, 2011.
"""
struct BC2Copula{P} <: ExtremeValueCopula{P}
    a::P
    b::P
    function BC2Copula(a,b)
        T = promote_type(typeof(a), typeof(b))
        if !(0 <= a <= 1) || !(0 <= b <= 1) 
            throw(ArgumentError("Both parameters a and b must be in [0,1]"))
        end
        return new{T}(T(a), T(b))
    end
end

Distributions.params(C::BC2Copula) = (C.a, C.b)
function A(C::BC2Copula, t::Real)
    a, b = C.a, C.b
    return max(a*t, b*(1-t)) + max((1-a)*t, (1-b)*(1-t))
end


function Distributions._rand!(rng::Distributions.AbstractRNG, C::BC2Copula, u::AbstractVector{T}) where {T<:Real}
    a, b = C.a, C.b
    v1, v2 = rand(rng, Distributions.Uniform(0,1), 2)
    u[1] = max(v1^(1/a), v2^(1/(1-a)))
    u[2] = max(v1^(1/b), v2^(1/(1-b)))
    return u
end

τ(C::BC2Copula) = 1 - abs(C.a - C.b)

function ρ(C::BC2Copula)
    a,b = C.a, C.b
    return 2 * (a + b + a*b + max(a,b) - 2a^2 - 2b^2) / (3 - a - b - min(a,b)) / (a + b + max(a,b))
end