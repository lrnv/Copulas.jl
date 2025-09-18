"""
        WDistortion(v, j)

Parameters
    * `v ∈ (0,1]` – conditioning value
    * `j` – conditioned index

Lower Fréchet (W) bound conditional distortion: U_i | U_j=v with dependence
`max(u_i + v - 1, 0)` gives cdf(u)=max(u+v-1,0)/v.
"""
struct WDistortion{T} <: Distortion
    v::T
    j::Int8
end
Distributions.cdf(D::WDistortion, u::Real) = max(u + D.v - 1, 0) / D.v
Distributions.quantile(D::WDistortion, α::Real) = α * D.v + (1 - D.v)
