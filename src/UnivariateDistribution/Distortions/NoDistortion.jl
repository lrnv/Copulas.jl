###########################################################################
#####  Identity distortion (used e.g. by IndependentCopula)
###########################################################################
struct NoDistortion <: Distortion end
Distributions.cdf(::NoDistortion, u::Real) = clamp(u, zero(u), one(u))
Distributions.quantile(::NoDistortion, α::Real) = α
(::NoDistortion)(X::Distributions.UnivariateDistribution) = X
(::NoDistortion)(::Distributions.Uniform) = Distributions.Uniform()
function Distributions.logpdf(::NoDistortion, u::Real)
    T = typeof(float(u))
    return 0 <= u <= 1 ? zero(T) : T(-Inf)
end
