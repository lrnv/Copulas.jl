"""
        GaussianDistortion(μz, σz)

Parameters
    * `μz` – standardized conditional shift
    * `σz > 0` – conditional scale factor

Conditional distortion for a Gaussian copula / elliptical model: transforms a
Uniform(0,1) variate into the conditional marginal via standard normal quantiles.
Used internally by [`condition`](@ref) specialized fast path for GaussianCopula.
"""
struct GaussianDistortion{T} <: Distortion
    μz::T
    σz::T
end
function Distributions.cdf(d::GaussianDistortion, u::Real)
    N = Distributions.Normal()
    q = Distributions.quantile(N, u)
    return Distributions.cdf(N, (q - d.μz)/d.σz)
end
function Distributions.quantile(d::GaussianDistortion, α::Real)
    N = Distributions.Normal()
    q = Distributions.quantile(N, α)
    return Distributions.cdf(N, d.μz + d.σz * q)
end
function (D::GaussianDistortion)(X::Distributions.Normal)
    μ, σ = Distributions.location(X), Distributions.scale(X)
    return Distributions.Normal(μ + σ*D.μz, σ*D.σz)
end
## Methods moved to EllipticalCopulas/GaussianCopula.jl
