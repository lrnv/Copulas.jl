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
A(C::MOCopula, t::Real) = max(t + (1-t)*C.b, (1-t)+C.a*t)
_cdf(C::MOCopula, u::AbstractArray{<:Real}) = min(u[1]^C.a * u[2], u[1] * u[2]^C.b)
function Distributions._rand!(rng::Distributions.AbstractRNG, C::MOCopula, u::AbstractVector{T}) where {T<:Real}
    r, s, t = -log.(rand(rng,3)) # Exponentials(1)
    u[1] = exp(-min(r/(1-C.a), t/C.a))
    u[2] = exp(-min(s/(1-C.b), t/C.b))
    return u
end
τ(C::MOCopula) = C.a*C.b/(C.a+C.b-C.a*C.b)

function Distributions.logcdf(D::BivEVDistortion{<:MOCopula, T}, z::Real) where T
    C = D.C
    a, b = C.a, C.b

    # guard domain of z and conditioning value
    if !(0.0 < z < 1.0)
        return z <= 0 ? T(-Inf) : T(0.0)
    end
    ucond = D.uⱼ
    if !(0.0 < ucond < 1.0)
        return ucond <= 0 ? T(-Inf) : T(log(z)) # conditioning on 1 -> uniform
    end

    if D.j == 2
        # Condition on V = v, free variable is u = z
        u = z; v = ucond
        lu, lv = log(u), log(v)
        # Determine active branch of min(u^a v, u v^b)
        s1 = a*lu + lv
        s2 = lu + b*lv
        if s1 <= s2
            # C = u^a v, dC/dv = u^a = (C/v) * 1
            logC = s1
            factor = 1.0
        else
            # C = u v^b, dC/dv = b u v^{b-1} = (C/v) * b
            logC = s2
            factor = b
        end
        return factor <= 0 ? T(-Inf) : T(logC - log(v) + log(factor))
    else
        # Condition on U = u, free variable is v = z
        v = z; u = ucond
        lu, lv = log(u), log(v)
        s1 = a*lu + lv
        s2 = lu + b*lv
        if s1 <= s2
            # C = u^a v, dC/du = a u^{a-1} v = (C/u) * a
            logC = s1
            factor = a
        else
            # C = u v^b, dC/du = v^b = (C/u) * 1
            logC = s2
            factor = 1.0
        end
        return factor <= 0 ? T(-Inf) : T(logC - log(u) + log(factor))
    end
end

function Distributions.quantile(D::BivEVDistortion{<:MOCopula}, α::Real)
    C = D.C
    a, b = C.a, C.b
    v_or_u = D.uⱼ
    if !(0.0 <= α <= 1.0)
        throw(ArgumentError("α must be in [0,1]"))
    end
    if !(0.0 < v_or_u < 1.0)
        if v_or_u <= 0.0
            return 0.0
        else
            # conditioning on 1 ⇒ uniform
            return α
        end
    end

    if D.j == 2
        # Quantile of U | V = v
        v = v_or_u
        ustar = v^((1-b)/(1-a))
        α1 = b * ustar^a                   # left-limit value at u* (before jump)
        α2 = ustar^a                       # right-continuous CDF value at u*
        if α < α1
            # invert left branch: F = b u v^{b-1}
            return (α / b) * v^(1-b)
        elseif α <= α2
            # atom at u*
            return ustar
        else
            # invert right branch: F = u^a
            return α^(1/a)
        end
    else
        # Quantile of V | U = u
        u = v_or_u
        vstar = u^((1-a)/(1-b))
        α1 = a * vstar^b
        α2 = vstar^b
        if α < α1
            # invert left branch: F = a v u^{a-1}
            return (α / a) * u^(1-a)
        elseif α <= α2
            return vstar
        else
            # invert right branch: F = v^b
            return α^(1/b)
        end
    end
end