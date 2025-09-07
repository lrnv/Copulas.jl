###########################################################################
#####  Identity distortion (used e.g. by IndependentCopula)
###########################################################################
"""
    NoDistortion() <: Distortion

Identity uniform-scale distortion: cdf(u) = u, quantile(α) = α on [0,1].

Notes
- Acts as the identity map on UnivariateDistribution: D(X) == X.
- When applied to Uniform(0,1), it returns itself which behaves as Uniform.
"""
struct NoDistortion <: Distortion end

# Distribution interface on [0,1]
@inline Distributions.cdf(::NoDistortion, u::Real) = u
@inline Distributions.quantile(::NoDistortion, α::Real) = α

# Push-forward behavior: identity on any base marginal
@inline (::NoDistortion)(X::Distributions.UnivariateDistribution) = X
# And specifically keep Uniform as Uniform (override Distortion fallback)
@inline (::NoDistortion)(::Distributions.Uniform) = Distributions.Uniform()
