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
    return exp(- (x + y) * Aw + y) * (Aw - u * dAw)
end
function DistortionFromCop(C::ExtremeValueCopula, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int64)
    @assert length(C) == 2
    return BivEVDistortion(C, Int8(js[1]), float(uⱼₛ[1]))
end
