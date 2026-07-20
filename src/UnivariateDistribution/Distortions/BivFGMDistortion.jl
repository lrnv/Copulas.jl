###########################################################################
#####  FGMCopula fast-path (bivariate, p=1)
###########################################################################
struct BivFGMDistortion{T} <: Distortion
    θ::T
    j::Int8
    uⱼ::T
end
Distributions.cdf(D::BivFGMDistortion, u::Real) = u + D.θ * u * (1 - u) * (1 - 2D.uⱼ)
function Distributions.logcdf(D::BivFGMDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(D.θ), typeof(D.uⱼ)))
    u <= 0 && return T(-Inf)
    u >= 1 && return zero(T)
    a = T(D.θ) * (one(T) - 2 * T(D.uⱼ))
    return log(T(u)) + log1p(a * (one(T) - T(u)))
end
function Distributions.quantile(D::BivFGMDistortion, α::Real)
    a = D.θ * (1 - 2D.uⱼ)
    # Handle near-independence stably
    T = typeof(α)
    abs(a) < sqrt(eps(T)) && return α

    # Solve a u^2 - (1+a) u + α = 0 and pick the root in [0,1]
    return ((1 + a) - sqrt((1 + a)^2 - 4*a*α)) / 2a
end
function Distributions.logpdf(D::BivFGMDistortion, u::Real)
    T = float(promote_type(typeof(u), typeof(D.θ), typeof(D.uⱼ)))
    0 <= u <= 1 || return T(-Inf)
    v = D.θ * (1 - 2 * D.uⱼ) * (1 - 2 * u)
    p = 1 + v
    p <= 0 && return T(-Inf)
    return log1p(v)
end
