###########################################################################
#####  ArchimedeanCopula fast-paths
###########################################################################
struct ArchimedeanDistortion{TG, T} <: Distortion
    G::TG
    p::Int
    sJ::T
    den::T
    ArchimedeanDistortion(G::TG, p::Int, sJ::T, den::T) where {T<:Real, TG} = new{TG, T}(G, p, sJ, den)
end
function Distributions.cdf(D::ArchimedeanDistortion{TG, T}, u::Real) where {TG, T}
    return ϕ⁽ᵏ⁾(D.G, D.p, D.sJ + ϕ⁻¹(D.G, float(u))) / D.den
end
function Distributions.quantile(D::ArchimedeanDistortion{TG, T}, α::Real) where {TG, T}
    y = ϕ⁽ᵏ⁾⁻¹(D.G, D.p, α * D.den; start_at = D.sJ)
    return ϕ(D.G, y - D.sJ)
end
## ConditionalCopula moved next to ArchimedeanCopula definition
