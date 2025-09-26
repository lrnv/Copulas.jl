"""
		NoDistortion()

Parameters
	* (none)

Identity distortion: cdf(u)=u. Used when conditioning yields no change (e.g.
Independent copula, trivial subsets) or as a fallback in histogram/bin cases.
"""
struct NoDistortion <: Distortion end
Distributions.cdf(::NoDistortion, u::Real) = u
Distributions.quantile(::NoDistortion, α::Real) = α
(::NoDistortion)(X::Distributions.UnivariateDistribution) = X
(::NoDistortion)(::Distributions.Uniform) = Distributions.Uniform()
