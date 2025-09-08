"""
    TiltedGenerator(G, p, sJ) <: Generator

Archimedean generator tilted by conditioning on `p` components fixed at values
with cumulative generator sum `sJ = ∑ ϕ⁻¹(u_j)`. It defines

    ϕ_tilt(t) = ϕ^{(p)}(sJ + t) / ϕ^{(p)}(sJ)

and higher derivatives accordingly:

    ϕ_tilt^{(k)}(t) = ϕ^{(k+p)}(sJ + t) / ϕ^{(p)}(sJ)

which yields the conditional copula within the Archimedean family for the
remaining d-p variables.
You will get a TiltedGenerator if you condition() an archimedean copula.
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
