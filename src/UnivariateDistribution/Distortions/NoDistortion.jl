###########################################################################
#####  Identity distortion (used e.g. by IndependentCopula)
###########################################################################
struct NoDistortion <: Distortion end
Distributions.cdf(::NoDistortion, u::Real) = u
Distributions.quantile(::NoDistortion, α::Real) = α
(::NoDistortion)(X::Distributions.UnivariateDistribution) = X
(::NoDistortion)(::Distributions.Uniform) = Distributions.Uniform()
