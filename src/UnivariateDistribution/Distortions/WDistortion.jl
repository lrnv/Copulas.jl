###########################################################################
#####  Fréchet bounds fast-paths: W (lower)
###########################################################################
struct WDistortion{T} <: Distortion
    v::T
    j::Int8
end
function Distributions.cdf(D::WDistortion, u::Real)
    z = (u + D.v - 1) / D.v
    return clamp(z, zero(z), one(z))
end
Distributions.quantile(D::WDistortion, α::Real) = α * D.v + (1 - D.v)
function Distributions.logpdf(D::WDistortion, u::Real)
    T = promote_type(typeof(float(u)), typeof(D.v))
    return 0 <= u <= 1 && u + D.v > 1 ? T(-log(D.v)) : T(-Inf)
end
