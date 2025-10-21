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
function Distributions.pdf(D::ArchimedeanDistortion{TG, T}, u::Real) where {TG, T}
    # Support on (0,1); treat boundaries with zero density
    (0 < u < 1) || return zero(promote_type(eltype(u), T))
    return ϕ⁽ᵏ⁾(D.G, D.p + 1, D.sJ + ϕ⁻¹(D.G, u)) * ϕ⁻¹⁽¹⁾(D.G, u) / D.den
end
function Distributions.logpdf(D::ArchimedeanDistortion{TG, T}, u::Real) where {TG, T}
    # Support on (0,1); treat boundaries with zero density
    (0 < u < 1) || return promote_type(eltype(u), T)(-Inf)
    return log(abs(ϕ⁽ᵏ⁾(D.G, D.p + 1, D.sJ + ϕ⁻¹(D.G, u)))) + log(abs(ϕ⁻¹⁽¹⁾(D.G, u))) - log(abs(D.den))
end

