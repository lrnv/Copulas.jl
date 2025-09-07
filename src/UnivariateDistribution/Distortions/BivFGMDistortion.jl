###########################################################################
#####  FGMCopula fast-path (bivariate, p=1)
###########################################################################
struct BivFGMDistortion{T} <: Distortion
    θ::T
    j::Int8
    uⱼ::T
end
@inline function Distributions.cdf(D::BivFGMDistortion, u::Real)
    θ = D.θ
    if D.j == 2
        v = clamp(float(D.uⱼ), 0.0, 1.0); uu = clamp(float(u), 0.0, 1.0)
        return uu + θ * uu * (1 - uu) * (1 - 2v)
    else
        u_j = clamp(float(D.uⱼ), 0.0, 1.0); vv = clamp(float(u), 0.0, 1.0)
        return vv + θ * vv * (1 - vv) * (1 - 2u_j)
    end
end
@inline function Distributions.quantile(D::BivFGMDistortion, α::Real)
    T = Float64; a = zero(T); b = one(T)
    f(u) = Distributions.cdf(D, u) - clamp(float(α), 0.0, 1.0)
    return Roots.find_zero(f, (a, b), Roots.Bisection(); xtol = sqrt(eps(T)))
end
@inline function DistortionFromCop(C::FGMCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,T}, ::Int64) where {T}
    return BivFGMDistortion(float(C.θ[1]), Int8(js[1]), float(uⱼₛ[1]))
end
