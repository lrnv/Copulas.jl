"""
    MOTail{T}

Fields:
  - λ₁::Real      — parameter ≥ 0
  - λ₂::Real      — parameter ≥ 0
  - λ₁₂::Real     — parameter ≥ 0

Constructor

    MOCopula(λ₁, λ₂, λ₁₂)
    ExtremeValueCopula(MOTail(λ₁, λ₂, λ₁₂))

The (bivariate) Marshall-Olkin extreme-value copula is parameterized by ``\\lambda_i \\in [0,\\infty), i = 1, 2, \\{1,2\\}``
Its Pickands dependence function is

```math
A(t) = \\frac{\\lambda_1 (1-t)}{\\lambda_1 + \\lambda_{1,2}} + \\frac{\\lambda_2 t}{\\lambda_2 + \\lambda_{1,2}} + \\lambda_{1,2}\\max\\left \\{\\frac{1-t}{\\lambda_1 + \\lambda_{1,2}}, \\frac{t}{\\lambda_2 + \\lambda_{1,2}} \\right \\}
```

Special cases:

* If λ₁₂ = 0, reduces to an asymmetric independence-like form.
* If λ₁ = λ₂ = 0, degenerates to complete dependence.

References:

* [mai2012simulating](@cite) Mai, J. F., & Scherer, M. (2012). Simulating copulas: stochastic models, sampling algorithms, and applications (Vol. 4). World Scientific.
"""
struct MOTail{T} <: Tail2
    λ₁::T
    λ₂::T
    λ₁₂::T
    function MOTail(λ₁, λ₂, λ₁₂)
        (λ₁ ≥ 0 && λ₂ ≥ 0 && λ₁₂ ≥ 0) || throw(ArgumentError("All λ must be ≥ 0"))
        T = promote_type(typeof(λ₁), typeof(λ₂), typeof(λ₁₂))
        return new{T}(T(λ₁), T(λ₂), T(λ₁₂))
    end
end

const MOCopula{T} = ExtremeValueCopula{2, MOTail{T}}
Distributions.params(tail::MOTail) = (tail.λ₁, tail.λ₂, tail.λ₁₂)
MOCopula(λ₁, λ₂, λ₁₂) = ExtremeValueCopula(MOTail(λ₁, λ₂, λ₁₂))

function A(E::MOTail, t::Real)
    tt = _safett(t)
    λ₁, λ₂, λ₁₂ = E.λ₁, E.λ₂, E.λ₁₂
    term1 = λ₁ * (1-tt) / (λ₁ + λ₁₂)
    term2 = λ₂ * tt / (λ₂ + λ₁₂)
    term3 = λ₁₂ * max((1-tt)/(λ₁ + λ₁₂), tt/(λ₂ + λ₁₂))
    return term1 + term2 + term3
end

τ(C::ExtremeValueCopula{2,MOTail{T}}) where {T} = begin
    a = C.E.λ₁/(C.E.λ₁+C.E.λ₁₂)
    b = C.E.λ₂/(C.E.λ₂+C.E.λ₁₂)
    a*b/(a+b-a*b)
end

function Distributions.logcdf(D::BivEVDistortion{MOTail{T}, S}, z::Real) where {T, S}
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

function Distributions.quantile(D::BivEVDistortion{MOTail{T}, S}, α::Real) where {T, S}
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