###########################################################################
#####  Fréchet bounds fast-paths: M (upper)
###########################################################################
struct MDistortion{T} <: Distortion
    v::T
    j::Int8
end
Distributions.cdf(D::MDistortion, u::Real) = min(u / D.v, 1)
Distributions.quantile(D::MDistortion, α::Real) = α * D.v
DistortionFromCop(::MCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, i::Int) = MDistortion(float(uⱼₛ[1]), Int8(js[1]))
