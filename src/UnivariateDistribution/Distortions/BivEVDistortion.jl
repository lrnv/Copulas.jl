###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
end
function Distributions.cdf(D::BivEVDistortion, u::Real)
    uf = clamp(float(u), 0.0, 1.0)
    uf <= 0.0 && return 0.0
    uf >= 1.0 && return 1.0
    v = clamp(float(D.uⱼ), 0.0, 1.0)
    # Handle degenerate conditioning boundaries: as v -> 0, H_{i|j}(u|v) -> u
    if v <= 0.0
        return uf
    end
    x, y = -log(uf), -log(v)
    w = x / (x + y)
    Aw = A(D.C, w)
    dAw = dA(D.C, w)
    if D.j == 2
        # H_{1|2}(u|v) = ∂_2 C(u,v) = exp(-(x+y)A + y) * (A - w A')
        return clamp(exp(- (x + y) * Aw + y) * (Aw - w * dAw), 0.0, 1.0)
    else
        # H_{2|1}(v|u) = ∂_1 C(u,v) = exp(-(x+y)A + x) * (A + (1-w) A')
        return clamp(exp(- (x + y) * Aw + x) * (Aw + (1 - w) * dAw), 0.0, 1.0)
    end
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