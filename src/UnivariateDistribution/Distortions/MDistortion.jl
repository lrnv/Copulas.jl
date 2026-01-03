###########################################################################
#####  Fréchet bounds fast-paths: M (upper)
###########################################################################
struct MDistortion{T} <: Distortion
    v::T
    j::Int8
end
Distributions.cdf(D::MDistortion, u::Real) = min(u / D.v, 1)
Distributions.pdf(D::MDistortion, u::Real) = u > D.v ? zero(u) : one(u) / D.v
Distributions.quantile(D::MDistortion, α::Real) = α * D.v
