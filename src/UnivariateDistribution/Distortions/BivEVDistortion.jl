###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
    function BivEVDistortion(C::ExtremeValueCopula, j, u)
        uf = float(u)
        return new{typeof(C), typeof(uf)}(C, j, uf)
    end
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