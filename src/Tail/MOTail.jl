"""
    MOTail{T}

Fields:
  - λ₁::Real      — parameter ≥ 0
  - λ₂::Real      — parameter ≥ 0
  - λ₁₂::Real     — parameter ≥ 0

Constructor

    MOCopula(λ₁, λ₂, λ₁₂)
    ExtremeValueCopula(2, MOTail(λ₁, λ₂, λ₁₂))

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
Distributions.params(tail::MOTail) = (λ₁ = tail.λ₁, λ₂ = tail.λ₂, λ₃ = tail.λ₁₂)
MOCopula(λ₁, λ₂, λ₁₂) = ExtremeValueCopula(2, MOTail(λ₁, λ₂, λ₁₂))
MOCopula(d::Integer, λ₁, λ₂, λ₁₂) = ExtremeValueCopula(2, MOTail(λ₁, λ₂, λ₁₂))

function A(tail::MOTail{T}, t::Real) where T
    tt = _safett(t)
    zz = zero(promote_type(T, typeof(tt)))
    λ₁, λ₂, λ₁₂ = tail.λ₁, tail.λ₂, tail.λ₁₂
    om = 1 - tt
    d1 = λ₁ + λ₁₂
    d2 = λ₂ + λ₁₂
    # Use inv where possible; if a denominator is zero (degenerate), treat the corresponding ratio as zero
    r1 = d1 > 0 ? om * (λ₁ / d1) : zz
    r2 = d2 > 0 ? tt * (λ₂ / d2) : zz
    m1 = d1 > 0 ? (om / d1) : zz
    m2 = d2 > 0 ? (tt / d2) : zz
    term3 = λ₁₂ * max(m1, m2)
    return r1 + r2 + term3
end

τ(C::ExtremeValueCopula{2,MOTail{T}}) where {T} = begin
    a = C.tail.λ₁/(C.tail.λ₁+C.tail.λ₁₂)
    b = C.tail.λ₂/(C.tail.λ₂+C.tail.λ₁₂)
    a*b/(a+b-a*b)
end

# Fitting helpers for EV copulas using Marshall–Olkin tail (λ ≥ 0)
_example(::Type{<:MOCopula}, d) = ExtremeValueCopula(2, MOTail(1.0, 1.0, 1.0))
_unbound_params(::Type{<:MOCopula}, d, θ) = [log(θ.λ₁), log(θ.λ₂), log(θ.λ₃)]
_rebound_params(::Type{<:MOCopula}, d, α) = (; λ₁ = exp(α[1]), λ₂ = exp(α[2]), λ₃ = exp(α[3]))

function Distributions.logcdf(D::BivEVDistortion{MOTail{T}, S}, z::Real) where {T, S}
    a = D.tail.λ₁ / (D.tail.λ₁ + D.tail.λ₁₂)
    b = D.tail.λ₂ / (D.tail.λ₂ + D.tail.λ₁₂)

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
    a = D.tail.λ₁ / (D.tail.λ₁ + D.tail.λ₁₂)
    b = D.tail.λ₂ / (D.tail.λ₂ + D.tail.λ₁₂)
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
        logv = log(v)
        ustar = exp(((1 - b) / (1 - a)) * logv)
        α2 = exp(a * log(ustar))           # right-continuous CDF value at u*
        α1 = b * α2                        # left-limit value at u* (before jump)
        if α < α1
            # invert left branch: F = b u v^{b-1}
            return (α / b) * exp((1 - b) * logv)
        elseif α <= α2
            # atom at u*
            return ustar
        else
            # invert right branch: F = u^a
            return exp((1 / a) * log(α))
        end
    else
        # Quantile of V | U = u
        u = v_or_u
        logu = log(u)
        vstar = exp(((1 - a) / (1 - b)) * logu)
        α2 = exp(b * log(vstar))
        α1 = a * α2
        if α < α1
            # invert left branch: F = a v u^{a-1}
            return (α / a) * exp((1 - a) * logu)
        elseif α <= α2
            return vstar
        else
            # invert right branch: F = v^b
            return exp((1 / b) * log(α))
        end
    end
end