###########################################################################
#####  ArchimedeanCopula fast-paths
###########################################################################
struct ArchimedeanDistortion{TG, T, p} <: Distortion
    G::TG
    sJ::T
    den::T
    function ArchimedeanDistortion(G::Generator, p::Int, sJ::T) where {T<:Real}
        den = ϕ⁽ᵏ⁾(G, Val{p}(), sJ)
        return new{typeof(G), T, p}(G, sJ, den)
    end
end
function Distributions.cdf(D::ArchimedeanDistortion{TG, T, p}, u::Real) where {TG, T, p}
    return ϕ⁽ᵏ⁾(D.G, Val{p}(), D.sJ + ϕ⁻¹(D.G, float(u))) / D.den
end
function Distributions.quantile(D::ArchimedeanDistortion{TG, T, p}, α::Real) where {TG, T, p}
    y = ϕ⁽ᵏ⁾⁻¹(D.G, Val{p}(), α * D.den; start_at = D.sJ)
    return ϕ(D.G, y - D.sJ)
end
## ConditionalCopula moved next to ArchimedeanCopula definition
