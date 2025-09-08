###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::BivEVDistortion, z::Real)
    T = typeof(z)
    # bounds and degeneracies
    z ≤ 0    && return T(-Inf)   # P(X ≤ 0) = 0
    z ≥ 1    && return T(0)      # P(X ≤ 1) = 1
    D.uⱼ ≤ 0 && return T(-Inf)   # conditioning on 0 ⇒ degenerate at 0
    D.uⱼ ≥ 1 && return T(log(z)) # conditioning on 1 ⇒ uniform[0,1], so logcdf = log(z)

    if D.j == 2
        # Condition on the second variable : V = D.uⱼ, free = u=z
        x, y = -log(z), -log(D.uⱼ)
        s = x + y
        w = x / s
        Aw, dAw = A(D.C, w), dA(D.C, w)
        tolog = Aw - w * dAw
        logval = -s * Aw + y
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = -log(D.uⱼ), -log(z)
        s = x + y
        w = x / s
        Aw, dAw = A(D.C, w), dA(D.C, w)
        tolog = Aw + (1 - w) * dAw
        logval = -s * Aw + x
    end

    # upper clip but no lower clip
    return min(logval + log(max(tolog, T(0))), T(0))
end

function Distributions.logcdf(D::BivEVDistortion{<:CuadrasAugeCopula, T}, z::Real) where T
    # bounds and degeneracies
    z ≤ 0    && return T(-Inf)
    z ≥ 1    && return T(0)
    D.uⱼ ≤ 0 && return T(-Inf)
    D.uⱼ ≥ 1 && return T(log(z))

    z ≥ D.uⱼ && return (1-D.C.θ) * log(z)
    return log1p(-D.C.θ) + log(z) - D.C.θ * log(D.uⱼ)
end

function Distributions.quantile(D::BivEVDistortion{<:CuadrasAugeCopula, T}, α::Real) where T
    α ≤ 0 && return T(0)
    α ≥ 1 && return T(1)
    D.uⱼ ≤ 0 && return T(0)
    D.uⱼ ≥ 1 && return α

    la = log(α)
    lu = log(D.uⱼ)
    lt = log1p(-D.C.θ)
    θ = D.C.θ

    la < lt + (1-θ)*lu && return exp(la - lt + θ * lu)
    la ≤      (1-θ)*lu && return D.uⱼ
    return exp(la / (1 - θ))
end

###############################################################################
#####  Specialization: Marshall–Olkin (MOCopula)
###############################################################################
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

###############################################################################
#####  Specialization: BC2 Copula (piecewise with two kinks)
###############################################################################
function Distributions.logcdf(D::BivEVDistortion{<:BC2Copula, T}, z::Real) where T
    C = D.C
    a, b = C.a, C.b

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

function Distributions.quantile(D::BivEVDistortion{<:BC2Copula}, α::Real)
    C = D.C
    a, b = C.a, C.b
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