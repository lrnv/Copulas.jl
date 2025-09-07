###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::BivEVDistortion, u::Real)
    x, y = -log(u), -log(D.uⱼ)
    s = x + y
    w = x / s
    Aw, dAw = A(D.C, w), dA(D.C, w)

    if D.j == 2
        # condition sur la deuxième composante (V = uⱼ)
        return -s * Aw + y + log(Aw - w * dAw)
    else
        # condition sur la première composante (U = uⱼ)
        return -s * Aw + x + log(Aw + (1 - w) * dAw)
    end
end

