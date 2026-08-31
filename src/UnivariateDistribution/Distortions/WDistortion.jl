###########################################################################
#####  Fréchet bounds fast-paths: W (lower)
###########################################################################
struct WDistortion{T} <: Distortion
    v::T
    j::Int8
end
distortion_measure_style(::Type{<:WDistortion}) = NonAbsolutelyContinuousMeasure()
function Distributions.cdf(D::WDistortion, u::Real)
    T = promote_type(typeof(float(u)), typeof(D.v))
    return u < one(T) - D.v ? zero(T) : one(T)
end
Distributions.quantile(D::WDistortion, α::Real) = one(D.v) - D.v
function Distributions.logpdf(D::WDistortion, u::Real)
    T = promote_type(typeof(float(u)), typeof(D.v))
    return u == one(T) - D.v ? zero(T) : T(-Inf)
end
