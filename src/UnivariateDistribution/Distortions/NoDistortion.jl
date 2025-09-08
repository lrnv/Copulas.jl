###########################################################################
#####  Identity distortion (used e.g. by IndependentCopula)
###########################################################################
struct NoDistortion <: Distortion end
@inline Distributions.cdf(::NoDistortion, u::Real) = u
@inline Distributions.quantile(::NoDistortion, α::Real) = α
@inline (::NoDistortion)(X::Distributions.UnivariateDistribution) = X
@inline (::NoDistortion)(::Distributions.Uniform) = Distributions.Uniform()
