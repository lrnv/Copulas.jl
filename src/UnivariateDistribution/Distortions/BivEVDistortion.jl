###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::BivEVDistortion, z::Real)
    # Gérer explicitement les bornes
    T = typeof(z)
    if z ≤ 0
        return T(-Inf)          # P(X ≤ 0) = 0
    elseif z ≥ 1
        return T(0)           # P(X ≤ 1) = 1
    elseif D.uⱼ ≤ 0
        return T(-Inf)          # conditioning on 0 ⇒ degenerate at 0
    elseif D.uⱼ ≥ 1
        return log(z)        # conditioning on 1 ⇒ uniform[0,1], so logcdf = log(z)
    end

    if D.j == 2
        # Condition on the second variable : V = D.uⱼ, free = u=z
        x, y = -log(z), -log(D.uⱼ)
        s = x + y
        w = x / s
        Aw, dAw = A(D.C, w), dA(D.C, w)
        logval = -s * Aw + y + log(Aw - w * dAw)
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = -log(D.uⱼ), -log(z)
        s = x + y
        w = x / s
        Aw, dAw = A(D.C, w), dA(D.C, w)
        logval = -s * Aw + x + log(Aw + (1 - w) * dAw)
    end

    # upper clip but no lower clip
    return min(logval, 0.0)
end

