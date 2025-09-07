###########################################################################
#####  FGMCopula fast-path (bivariate, p=1)
###########################################################################
struct BivFGMDistortion{T} <: Distortion
    θ::T
    j::Int8
    uⱼ::T
end
@inline Distributions.cdf(D::BivFGMDistortion, u::Real) = u + D.θ * u * (1 - u) * (1 - 2D.uⱼ)
@inline function Distributions.quantile(D::BivFGMDistortion, α::Real)
    a = D.θ * (1 - 2D.uⱼ)
    # Handle near-independence stably
    abs(a) < sqrt(eps(T)) && return α

    # Solve a u^2 - (1+a) u + α = 0 and pick the root in [0,1]
    return ((1 + a) - sqrt((1 + a)^2 - 4*a*α)) / 2a
end
@inline DistortionFromCop(C::FGMCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, ::Int) = BivFGMDistortion(float(C.θ[1]), Int8(js[1]), float(uⱼₛ[1]))
