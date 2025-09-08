###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::BivEVDistortion, z::Real)
    if D.j == 2
        # Condition on the second variable : V = D.uⱼ, free = u=z
        x, y = -log(z), -log(D.uⱼ)
        s = x + y
        w = x / s
        Aw, dAw = A(D.C, w), dA(D.C, w)
        return -s * Aw + y + log(Aw - w * dAw)
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = -log(D.uⱼ), -log(z)
        s = x + y
        w = x / s
        Aw, dAw = A(D.C, w), dA(D.C, w)
        return -s * Aw + x + log(Aw + (1 - w) * dAw)
    end
end

