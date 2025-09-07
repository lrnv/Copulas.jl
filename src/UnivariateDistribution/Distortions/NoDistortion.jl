###########################################################################
#####  Identity distortion (used e.g. by IndependentCopula)
###########################################################################
struct NoDistortion <: Distortion end

# Distribution interface on [0,1]
@inline Distributions.cdf(::NoDistortion, u::Real) = u
@inline Distributions.quantile(::NoDistortion, α::Real) = α

# Push-forward behavior: identity on any base marginal
@inline (::NoDistortion)(X::Distributions.UnivariateDistribution) = X
# And specifically keep Uniform as Uniform (override Distortion fallback)
@inline (::NoDistortion)(::Distributions.Uniform) = Distributions.Uniform()
