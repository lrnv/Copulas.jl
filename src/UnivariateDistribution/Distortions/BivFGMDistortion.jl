"""
        BivFGMDistortion(θ, j, u_j)

Parameters
    * `θ` – FGM dependence parameter (|θ| ≤ 1 typically)
    * `j ∈ {1,2}` – conditioned index
    * `u_j ∈ (0,1)` – conditioning value

Conditional distortion for the bivariate FGM copula (fast Rosenblatt path).
"""
struct BivFGMDistortion{T} <: Distortion
    θ::T
    j::Int8
    uⱼ::T
end
Distributions.cdf(D::BivFGMDistortion, u::Real) = u + D.θ * u * (1 - u) * (1 - 2D.uⱼ)
function Distributions.quantile(D::BivFGMDistortion, α::Real)
    a = D.θ * (1 - 2D.uⱼ)
    # Handle near-independence stably
    T = typeof(α)
    abs(a) < sqrt(eps(T)) && return α

    # Solve a u^2 - (1+a) u + α = 0 and pick the root in [0,1]
    return ((1 + a) - sqrt((1 + a)^2 - 4*a*α)) / 2a
end
