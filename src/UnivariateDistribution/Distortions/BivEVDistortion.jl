###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TC,T} <: Distortion
    C::TC
    j::Int8
    uⱼ::T
end
function Distributions.cdf(D::BivEVDistortion, u::Real)
    u_i = clamp(float(u), eps(Float64), 1 - eps(Float64))
    u_j = clamp(float(D.uⱼ), eps(Float64), 1 - eps(Float64))
    if D.j == 2
        u′, v′ = -log(u_i), -log(u_j)
        val, du, dv, dudv = _der_ℓ(D.C, u′, v′)
        val1, du1, dv1, dudv1 = _der_ℓ(D.C, 0.0, v′)
        Cuv = exp(-val); C1v = exp(-val1)
        return (Cuv / C1v) * (dv / dv1)
    else
        u′, v′ = -log(u_j), -log(u_i)
        val, du, dv, dudv = _der_ℓ(D.C, u′, v′)
        val1, du1, dv1, dudv1 = _der_ℓ(D.C, u′, 0.0)
        Cuv = exp(-val); Cu1 = exp(-val1)
        return (Cuv / Cu1) * (du / du1)
    end
end
function Distributions.quantile(D::BivEVDistortion, α::Real)
    T = Float64; a = zero(T); b = one(T)
    f(u) = Distributions.cdf(D, u) - float(α)
    return Roots.find_zero(f, (a, b), Roots.Bisection(); xtol = sqrt(eps(T)))
end
function DistortionFromCop(C::ExtremeValueCopula, js::NTuple{1,Int}, uⱼₛ::NTuple{1,T}, ::Int64) where {T}
    j = js[1]; @assert (length(C) == 2)
    return BivEVDistortion(C, Int8(j), float(uⱼₛ[1]))
end
