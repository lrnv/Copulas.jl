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
function Distributions.logcdf(d::GaussianDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(d.μz), typeof(d.σz)))
    u <= 0 && return T(-Inf)
    u >= 1 && return zero(T)
    N = Distributions.Normal()
    q = Distributions.quantile(N, T(u))
    return T(Distributions.logcdf(N, (q - T(d.μz)) / T(d.σz)))
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
    T = float(promote_type(typeof(u), typeof(d.μz), typeof(d.σz)))
    0 < u < 1 || return T(-Inf)
    N = Distributions.Normal()
    q = T(Distributions.quantile(N, T(u)))
    z = (q - T(d.μz)) / T(d.σz)
    return (q^2 - z^2) / 2 - log(abs(T(d.σz)))
end
