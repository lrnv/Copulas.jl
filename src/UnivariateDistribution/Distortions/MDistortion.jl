###########################################################################
#####  Fréchet bounds fast-paths: M (upper)
###########################################################################
struct MDistortion{T} <: Distortion
    v::T
    j::Int8
end
function Distributions.cdf(D::MDistortion, u::Real)
    z = u / D.v
    return clamp(z, zero(z), one(z))
end
Distributions.quantile(D::MDistortion, α::Real) = α * D.v
function Distributions.logpdf(D::MDistortion, u::Real)
    T = promote_type(typeof(float(u)), typeof(D.v))
    return 0 <= u <= D.v ? T(-log(D.v)) : T(-Inf)
end
