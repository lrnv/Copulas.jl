###########################################################################
#####  GaussianCopula fast-paths
###########################################################################
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

function Distributions.logpdf(d::GaussianDistortion, u::Real)
    (0 < u < 1) || return -Inf
    σ = d.σz
    σ > 0 || return -Inf
    N = Distributions.Normal()
    q = Distributions.quantile(N, u)                 # Φ^{-1}(u)
    w = (q - float(d.μz)) / σ                        # standardized conditional

    # log f(u) = log φ(w) - log σ - log φ(q)
    return Distributions.logpdf(N, w) - log(σ) - Distributions.logpdf(N, q)
end

Distributions.pdf(d::GaussianDistortion, u::Real) = exp(Distributions.logpdf(d, u))
