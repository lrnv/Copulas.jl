###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
end
function Distributions.cdf(D::BivEVDistortion, u::Real)
    x, y = -log(u), -log(D.uⱼ)
    w = x / (x + y)
    Aw, dAw = A(D.C, w), dA(D.C, w)
    u = D.j ==2 ? w : 1-w
    return clamp(exp(- (x + y) * Aw + y) * (Aw - u * dAw), 0, 1)
end

###########################################################################
#####  Closed-form quantile for Cuadras–Augé (Marshall–Olkin) EV copula
###########################################################################
function Distributions.quantile(D::BivEVDistortion{<:CuadrasAugeCopula}, α::Real)
    θ = D.C.θ
    αf = clamp(float(α), 0.0, 1.0)
    if θ == 0.0
        return αf
    end
    v = float(D.uⱼ)
    if v <= 0.0
        return 0.0
    elseif v >= 1.0
        return αf^(1/(1-θ))
    end
    a1 = (1-θ) * v^(1-θ)
    a2 = v^(1-θ)
    if αf < a1
        return αf * v^θ / (1-θ)
    elseif αf <= a2
        return v
    else
        return αf^(1/(1-θ))
    end
end