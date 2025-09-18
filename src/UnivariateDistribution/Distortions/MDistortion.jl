"""
        MDistortion(v, j)

Parameters
    * `v ∈ (0,1]` – conditioning value of the fixed variable
    * `j ∈ {1,2,…}` – conditioned index (bivariate use common)

Upper Fréchet (M) bound conditional distortion: U_i | U_j=v with dependence
`min(u_i, v)` leads to cdf(u)=min(u/v,1).
"""
struct MDistortion{T} <: Distortion
    v::T
    j::Int8
end
Distributions.cdf(D::MDistortion, u::Real) = min(u / D.v, 1)
Distributions.quantile(D::MDistortion, α::Real) = α * D.v
