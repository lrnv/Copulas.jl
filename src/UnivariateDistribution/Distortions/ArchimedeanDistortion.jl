###########################################################################
#####  ArchimedeanCopula fast-paths
###########################################################################
"""
    TiltedGenerator(G, p, sJ) <: Generator

Archimedean generator tilted by conditioning on `p` components fixed at values
with cumulative generator sum `sJ = ∑ ϕ⁻¹(u_j)`. It defines

    ϕ_tilt(t) = ϕ^{(p)}(sJ + t) / ϕ^{(p)}(sJ)

and higher derivatives accordingly:

    ϕ_tilt^{(k)}(t) = ϕ^{(k+p)}(sJ + t) / ϕ^{(p)}(sJ)

which yields the conditional copula within the Archimedean family for the
remaining d-p variables.
"""
struct TiltedGenerator{TG, T, p} <: Generator
    G::TG
    sJ::T
    den::T
    function TiltedGenerator(G::Generator, ::Val{p}, sJ::T) where {p,T<:Real}
        den = ϕ⁽ᵏ⁾(G, Val{p}(), sJ)
        return new{typeof(G), T, p}(G, sJ, den)
    end
end
max_monotony(G::TiltedGenerator{TG, T, p}) where {TG, T, p} = max(0, max_monotony(G.G) - p)
ϕ(G::TiltedGenerator{TG, T, p}, t::Real) where {TG, T, p} = ϕ⁽ᵏ⁾(G.G, Val{p}(), G.sJ + t) / G.den
ϕ⁻¹(G::TiltedGenerator{TG, T, p}, x::Real) where {TG, T, p} = ϕ⁽ᵏ⁾⁻¹(G.G, Val{p}(), x * G.den; start_at = G.sJ) - G.sJ
ϕ⁽ᵏ⁾(G::TiltedGenerator{TG, T, p}, ::Val{k}, t::Real) where {TG, T, p, k} = ϕ⁽ᵏ⁾(G.G, Val{k + p}(), G.sJ + t) / G.den
ϕ⁽ᵏ⁾⁻¹(G::TiltedGenerator{TG, T, p}, ::Val{k}, y::Real; start_at = G.sJ) where {TG, T, p, k} = ϕ⁽ᵏ⁾⁻¹(G.G, Val{k + p}(), y * G.den; start_at = start_at) - G.sJ
ϕ⁽¹⁾(G::TiltedGenerator{TG, T, p}, t) where {TG, T, p} = ϕ⁽ᵏ⁾(G, Val{1}(), t)

struct ArchimedeanDistortion{TG, T, p} <: Distortion
    G::TG
    sJ::T
    den::T
    function ArchimedeanDistortion(G::Generator, p::Int, sJ::T) where {T<:Real}
        den = ϕ⁽ᵏ⁾(G, Val{p}(), sJ)
        return new{typeof(G), T, p}(G, sJ, den)
    end
end
@inline function Distributions.cdf(D::ArchimedeanDistortion{TG, T, p}, u::Real) where {TG, T, p}
    return ϕ⁽ᵏ⁾(D.G, Val{p}(), D.sJ + ϕ⁻¹(D.G, float(u))) / D.den
end
@inline function Distributions.quantile(D::ArchimedeanDistortion{TG, T, p}, α::Real) where {TG, T, p}
    y = ϕ⁽ᵏ⁾⁻¹(D.G, Val{p}(), α * D.den; start_at = D.sJ)
    return ϕ(D.G, y - D.sJ)
end
function DistortionFromCop(C::ArchimedeanCopula, js::NTuple{p,Int64}, uⱼₛ::NTuple{p,Float64}, i::Int64) where {p}
    @assert length(js) == length(uⱼₛ)
    sJ = zero(eltype(uⱼₛ))
    @inbounds for u in uⱼₛ
        sJ += ϕ⁻¹(C.G, float(u))
    end
    return ArchimedeanDistortion(C.G, p, float(sJ))
end
## ConditionalCopula moved next to ArchimedeanCopula definition
