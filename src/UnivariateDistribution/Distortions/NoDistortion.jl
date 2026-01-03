###########################################################################
#####  Identity distortion (used e.g. by IndependentCopula)
###########################################################################
struct NoDistortion <: Distortion end
Distributions.cdf(::NoDistortion, u::Real) = u
Distributions.pdf(::NoDistortion, u::Real) = one(u)
Distributions.quantile(::NoDistortion, α::Real) = α
(::NoDistortion)(X::Distributions.UnivariateDistribution) = X
(::NoDistortion)(::Distributions.Uniform) = Distributions.Uniform()
