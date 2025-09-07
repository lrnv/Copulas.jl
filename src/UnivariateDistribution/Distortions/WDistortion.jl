###########################################################################
#####  Fréchet bounds fast-paths: W (lower)
###########################################################################
struct WDistortion{T} <: Distortion
    v::T
    j::Int8
end
Distributions.cdf(D::WDistortion, u::Real) = max(u + D.v - 1, 0) / D.v
Distributions.quantile(D::WDistortion, α::Real) = α * D.v + (1 - D.v)
DistortionFromCop(::WCopula, js::Tuple{Int64}, uⱼₛ::Tuple{Float64}, i::Int64) = WDistortion(float(uⱼₛ[1]), Int8(js[1]))
