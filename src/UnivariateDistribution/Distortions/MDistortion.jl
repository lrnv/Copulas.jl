###########################################################################
#####  Fréchet bounds fast-paths: M (upper)
###########################################################################
struct MDistortion{T} <: Distortion
    v::T
    j::Int8
end
Distributions.cdf(D::MDistortion, u::Real) = min(u / D.v, 1)
Distributions.quantile(D::MDistortion, α::Real) = α * D.v
DistortionFromCop(::MCopula{2}, js::NTuple{1,Int64}, uⱼₛ::NTuple{1,T}, i::Int64) where {T} = MDistortion(float(uⱼₛ[1]), Int8(js[1]))
