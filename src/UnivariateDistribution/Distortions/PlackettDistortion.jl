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
