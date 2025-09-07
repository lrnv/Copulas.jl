###########################################################################
#####  PlackettCopula fast-path (bivariate, p=1)
###########################################################################
struct PlackettDistortion{T} <: Distortion
    θ::T
    j::Int8
    uⱼ::T
end
@inline _plackett_term1(θ, u, v) = (θ - 1) * (u + v) + 1
@inline function _plackett_sqrtD(θ, u, v)
    η = θ - 1; t1 = _plackett_term1(θ, u, v)
    return sqrt(max(t1 * t1 - 4 * θ * η * u * v, 0.0))
end
@inline function _dC_dv_plackett(θ, u, v)
    t1 = _plackett_term1(θ, u, v); sD = _plackett_sqrtD(θ, u, v)
    return 0.5 * (1 - (t1 - 2 * θ * u) / sD)
end
@inline function Distributions.cdf(D::PlackettDistortion, u::Real)
    θ = D.θ
    u_i = clamp(float(u), eps(Float64), 1 - eps(Float64))
    u_j = clamp(float(D.uⱼ), eps(Float64), 1 - eps(Float64))
    if D.j == 2
        num = _dC_dv_plackett(θ, u_i, u_j); den = _dC_dv_plackett(θ, 1.0, u_j)
    else
        num = _dC_dv_plackett(θ, u_j, u_i); den = _dC_dv_plackett(θ, u_j, 1.0)
    end
    r = num / den; return ifelse(r < 0, 0.0, ifelse(r > 1, 1.0, r))
end
@inline function Distributions.quantile(D::PlackettDistortion, α::Real)
    T = Float64; a = zero(T); b = one(T)
    f(u) = Distributions.cdf(D, u) - clamp(float(α), 0.0, 1.0)
    return Roots.find_zero(f, (a, b), Roots.Bisection(); xtol = sqrt(eps(T)))
end
## DistortionFromCop moved next to PlackettCopula
