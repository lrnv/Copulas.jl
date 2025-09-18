"""
        PlackettDistortion(θ, j, u_j)

Parameters
    * `θ > 0` – Plackett dependence parameter
    * `j ∈ {1,2}` – conditioned coordinate
    * `u_j ∈ (0,1)` – conditioning value

Conditional distortion for the Plackett copula (bivariate fast path) using a
stable `logcdf` formulation.
"""
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
