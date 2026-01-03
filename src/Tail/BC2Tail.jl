"""
    BC2Tail{T}, BC2Copula{T}

Fields:
  - a::Real — parameter (a ∈ [0,1])
  - b::Real — parameter (b ∈ [0,1])

Constructor

    BC2Copula(a, b)
    ExtremeValueCopula(2, BC2Tail(a, b))

The bivariate BC2 extreme-value copula is parameterized by two parameters ``a, b \\in [0,1]``. Its Pickands dependence function is

```math
A(t) = \\max\\{a t, b (1-t) \\} + \\max\\{(1-a)t, (1-b)(1-t)\\}, \\quad t \\in [0,1].
```

References:

* [mai2011bivariate](@cite) Mai, J. F., & Scherer, M. (2011). Bivariate extreme-value copulas with discrete Pickands dependence measure. Extremes, 14, 311-324. Springer, 2011.
"""
BC2Tail, BC2Copula

struct BC2Tail{T} <: Tail2
    a::T
    b::T
    function BC2Tail(a, b)
        T = promote_type(typeof(a), typeof(b))
        (0 ≤ a ≤ 1) || throw(ArgumentError("a must be in [0,1]"))
        (0 ≤ b ≤ 1) || throw(ArgumentError("b must be in [0,1]"))
        return new{T}(T(a), T(b))
    end
end

const BC2Copula{T} = ExtremeValueCopula{2, BC2Tail{T}}
Distributions.params(tail::BC2Tail) = (a = tail.a, b = tail.b)
_unbound_params(::Type{<:BC2Tail}, d, θ) = [log(θ.a) - log1p(-θ.a), log(θ.b) - log1p(-θ.b)]
_rebound_params(::Type{<:BC2Tail}, d, α) = begin
    σ(x) = 1 / (1 + exp(-x))
    (; a = σ(α[1]), b = σ(α[2]))
end

function A(tail::BC2Tail, t::Real)
    tt = _safett(t)
    a, b = tail.a, tail.b
    return max(a*tt, b*(1-tt)) + max((1-a)*tt, (1-b)*(1-tt))
end
τ(C::ExtremeValueCopula{2, BC2Tail{T}}) where {T} = 1 - abs(C.tail.a - C.tail.b)
function ρ(C::ExtremeValueCopula{2, BC2Tail{T}}) where {T}
    a, b = C.tail.a, C.tail.b
    num = 2 * (a + b + a*b + max(a,b) - 2a^2 - 2b^2)
    den = (3 - a - b - min(a,b)) * (a + b + max(a,b))
    return num / den
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2, BC2Tail{T}}, u::AbstractVector{S}) where {T,S<:Real}
    a, b = C.tail.a, C.tail.b
    v1, v2 = rand(rng, Distributions.Uniform(0,1), 2)
    u[1] = max(v1^(1/a), v2^(1/(1-a)))
    u[2] = max(v1^(1/b), v2^(1/(1-b)))
    return u
end
function Distributions.logcdf(D::BivEVDistortion{<:BC2Tail{TF1}, TF2}, z::Real) where {TF1,TF2}
    T = promote_type(TF1, TF2, typeof(z))

    a, b = D.tail.a, D.tail.b

    if !(0.0 < z < 1.0)
        return z <= 0 ? T(-Inf) : T(0.0)
    end
    ucond = D.uⱼ
    if !(0.0 < ucond < 1.0)
        return ucond <= 0 ? T(-Inf) : T(log(z))
    end

    if D.j == 2
        # Condition on V = v, free = u = z
        u = z; v = ucond
        lu, lv = log(u), log(v)
        c1 = a*lu <= b*lv             # decide for min(u^a, v^b)
        c2 = (1-a)*lu <= (1-b)*lv     # decide for min(u^{1-a}, v^{1-b})
        if c1 && c2
            # C = u, dC/dv = 0
            return T(-Inf)
        elseif c1 && !c2
            # C = u^a v^{1-b}, dC/dv = (1-b) u^a v^{-b}
            logC = a*lu + (1-b)*lv
            factor = (1-b)
            return factor <= 0 ? T(-Inf) : T(logC - log(v) + log(factor))
        elseif !c1 && c2
            # C = v^b u^{1-a}, dC/dv = b v^{b-1} u^{1-a}
            logC = b*lv + (1-a)*lu
            factor = b
            return factor <= 0 ? T(-Inf) : T(logC - log(v) + log(factor))
        else
            # both mins pick v parts: C = v, dC/dv = 1
            logC = lv
            factor = 1.0
            return T(logC - log(v) + log(factor))
        end
    else
        # Condition on U = u, free = v = z
        v = z; u = ucond
        lu, lv = log(u), log(v)
        c1 = a*lu <= b*lv
        c2 = (1-a)*lu <= (1-b)*lv
        if c1 && c2
            # C = u, dC/du = 1
            logC = lu
            factor = 1.0
            return T(logC - log(u) + log(factor))
        elseif c1 && !c2
            # C = u^a v^{1-b}, dC/du = a u^{a-1} v^{1-b}
            logC = a*lu + (1-b)*lv
            factor = a
            return factor <= 0 ? T(-Inf) : T(logC - log(u) + log(factor))
        elseif !c1 && c2
            # C = v^b u^{1-a}, dC/du = (1-a) v^b u^{-a}
            logC = b*lv + (1-a)*lu
            factor = (1-a)
            return factor <= 0 ? T(-Inf) : T(logC - log(u) + log(factor))
        else
            # both mins pick v parts: C = v, dC/du = 0
            return T(-Inf)
        end
    end
end
function Distributions.quantile(D::BivEVDistortion{BC2Tail{T}, S}, α::Real) where {T, S}
    a, b = D.tail.a, D.tail.b
    t = D.uⱼ
    if !(0.0 <= α <= 1.0)
        throw(ArgumentError("α must be in [0,1]"))
    end
    if !(0.0 < t < 1.0)
        if t <= 0.0
            return 0.0
        else
            return α
        end
    end

    if D.j == 2
        # Quantile of U | V = v (v = t)
        v = t
        # thresholds where mins switch
        u1 = (a == 0) ? 0.0 : v^(b/a)              # boundary for min(u^a, v^b)
        u2 = (a == 1) ? 1.0 : v^((1-b)/(1-a))      # boundary for min(u^{1-a}, v^{1-b})
        if u2 <= u1
            # order: 0 -- u2 (jump) -- (cont: case B) -- u1 (jump to 1)
            α0 = (1-b) * (u2 / v)                  # jump at u2
            if α <= α0
                return u2
            elseif α < 1 - b
                # continuous: F = (1-b) u^a v^{-b}
                return ((α) * v^b / (1-b))^(1/a)
            else
                # jump at u1 to 1
                return u1
            end
        else
            # order: 0 -- u1 (jump) -- (cont: case C) -- u2 (jump to 1)
            α0 = b * (u1^(1-a) * v^(b-1))          # jump at u1
            if α <= α0
                return u1
            elseif α < b
                # continuous: F = b u^{1-a} v^{b-1}
                return (α * v^(1-b) / b)^(1/(1-a))
            else
                return u2
            end
        end
    else
        # Quantile of V | U = u (u = t)
        u = t
        v1 = (b == 0) ? 0.0 : u^(a/b)
        v2 = (b == 1) ? 1.0 : u^((1-a)/(1-b))
        if v2 <= v1
            # order: 0 -- v2 (jump) -- (cont with factor a) -- v1 (jump to 1)
            α0 = a * (v2 / u)                      # jump at v2
            if α <= α0
                return v2
            elseif α < a
                # continuous: F = a v u^{a-1}
                return (α * u^(1-a) / a)
            else
                return v1
            end
        else
            # order: 0 -- v1 (jump) -- (cont with factor 1-a) -- v2 (jump to 1)
            α0 = (1-a) * (v1^b * u^(-a))           # jump at v1
            if α <= α0
                return v1
            elseif α < 1 - a
                # continuous: F = (1-a) v^b u^{-a}
                return (α * u^a / (1-a))^(1/b)
            else
                return v2
            end
        end
    end
end
