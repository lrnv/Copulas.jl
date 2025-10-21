###########################################################################
#####  PlackettCopula fast-path (bivariate, p=1)
###########################################################################
struct PlackettDistortion{T} <: Distortion
    θ::T
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::PlackettDistortion, u::Real) 
    θ, v = D.θ, D.uⱼ

    η = θ - 1
    t1 = η * (u + v) + 1
    s1 = sqrt(t1 * t1 - 4θ * η * u * v)
    num = log1p(- (t1 - 2θ * u) / s1)

    t2 = η * (1 + v) + 1
    s2 = sqrt(t2 * t2 - 4θ * η * v)
    den = log1p(- (t2 - 2 * θ) / s2)

    return num - den
end
## DistortionFromCop moved next to PlackettCopula

function Distributions.logpdf(D::PlackettDistortion, u::Real)
    θ = float(D.θ)
    v = float(D.uⱼ)
    u = float(u)

    # Support and parameter checks
    (0 < u < 1)      || return -Inf
    (0 < v && v < 1) || return -Inf
    θ > 0            || return -Inf
    θ != 1           || return 0.0  # independence copula: density is 1 on (0,1)^2

    η = θ - 1
    t1 = η * (u + v) + 1
    Δ2 = t1 * t1 - 4 * θ * η * u * v
    Δ2 <= 0 && return -Inf
    Δ = sqrt(Δ2)

    # Plackett copula density c(u,v)
    # c = [ η (t1 - 2 θ u)(t1 - 2 θ v) - (η - 2 θ) Δ^2 ] / (2 Δ^3)
    a = (t1 - 2 * θ * u)
    b = (t1 - 2 * θ * v)
    num = η * a * b - (η - 2 * θ) * Δ2
    num <= 0 && return -Inf
    den = 2 * (Δ^3)
    return log(num) - log(den)
end