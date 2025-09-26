"""
        ArchimedeanDistortion(G, p, sJ, den)

Parameters
    * `G` – Archimedean generator
    * `p` – number of conditioned coordinates
    * `sJ` – sum of inverse generator values `∑ ϕ⁻¹(u_j)`
    * `den` – normalization constant `ϕ^{(p)}(sJ)`

Closed-form conditional distortion for Archimedean copulas derived from tilted
generator derivatives. Produced internally by conditioning routines.
"""
struct ArchimedeanDistortion{TG, T, p} <: Distortion
    G::TG
    sJ::T
    den::T
    ArchimedeanDistortion(G::TG, p::Int, sJ::T, den::T) where {T<:Real, TG} = new{TG, T, p}(G, sJ, den)
end
function Distributions.cdf(D::ArchimedeanDistortion{TG, T, p}, u::Real) where {TG, T, p}
    return ϕ⁽ᵏ⁾(D.G, Val{p}(), D.sJ + ϕ⁻¹(D.G, float(u))) / D.den
end
function Distributions.quantile(D::ArchimedeanDistortion{TG, T, p}, α::Real) where {TG, T, p}
    y = ϕ⁽ᵏ⁾⁻¹(D.G, Val{p}(), α * D.den; start_at = D.sJ)
    return ϕ(D.G, y - D.sJ)
end
## ConditionalCopula moved next to ArchimedeanCopula definition
