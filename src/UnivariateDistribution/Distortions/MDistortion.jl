###########################################################################
#####  Fréchet bounds fast-paths: M (upper)
###########################################################################
struct MDistortion{T} <: Distortion
    v::T
    j::Int8
end
function Distributions.cdf(D::MDistortion, u::Real)
    T = promote_type(typeof(float(u)), typeof(D.v))
    return u < D.v ? zero(T) : one(T)
end
Distributions.quantile(D::MDistortion, α::Real) = D.v
function Distributions.logpdf(D::MDistortion, u::Real)
    T = promote_type(typeof(float(u)), typeof(D.v))
    return u == D.v ? zero(T) : T(-Inf)
end
