###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TT,T} <: Distortion
    tail::TT
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
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        tolog = Aw - w * dAw
        logval = -s * Aw + y
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = -log(D.uⱼ), -log(z)
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        tolog = Aw + (1 - w) * dAw
        logval = -s * Aw + x
    end

    # upper clip but no lower clip
    return min(logval + log(max(tolog, T(0))), T(0))
end
function Distributions.logpdf(D::BivEVDistortion, z::Real)
    T = typeof(z)
    # bounds and degeneracies
    z ≤ 0    && return T(-Inf)
    z ≥ 1    && return T(-Inf)
    D.uⱼ ≤ 0 && return T(-Inf)
    D.uⱼ ≥ 1 && return T(0)  # conditioning on 1 ⇒ uniform density = 1 => logpdf = 0

    if D.j == 2
        # Condition on the second variable : V = D.uⱼ, free = u=z
        x, y = -log(z), -log(D.uⱼ)
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        ddAw = d²A(D.tail, w)
        Tval = Aw - w * dAw
        Tval ≤ 0 && return T(-Inf)

        logval = -s * Aw + y
        # derivatives
        # logval' = (Aw + (y/s)*dAw) / z
        lvp = (Aw + (y / s) * dAw) / z
        # T'(w)*dw/dz = w * ddAw * y / (z * s^2)
        tp_term = w * ddAw * y / (z * s^2)
        B = tp_term + Tval * lvp
        B ≤ 0 && return T(-Inf)

        return logval + log(B)
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = -log(D.uⱼ), -log(z)
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        ddAw = d²A(D.tail, w)
        Tval = Aw + (1 - w) * dAw
        Tval ≤ 0 && return T(-Inf)

        logval = -s * Aw + x
        # derivatives
        # logval' = (Aw - (x/s)*dAw) / z
        lvp = (Aw - (x / s) * dAw) / z
        # T'(w)*dw/dz = x * (1 - w) * ddAw / (z * s^2)
        tp_term = x * (1 - w) * ddAw / (z * s^2)
        B = tp_term + Tval * lvp
        B ≤ 0 && return T(-Inf)

        return logval + log(B)
    end
end