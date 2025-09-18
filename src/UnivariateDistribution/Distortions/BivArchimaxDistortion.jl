"""
        BivArchimaxDistortion(gen, tail, j, u_j)

Parameters
    * `gen` – Archimedean generator (ϕ)
    * `tail` – EV tail / STDF A
    * `j ∈ {1,2}` – conditioned coordinate
    * `u_j` – conditioning value in (0,1)

Conditional distortion for bivariate Archimax copulas, combining generator and
tail derivatives for a stable logcdf.
"""
struct BivArchimaxDistortion{TG,TT,T} <: Distortion
    gen::TG
    tail::TT
    j::Int8
    uⱼ::T
end

function Distributions.logcdf(D::BivArchimaxDistortion, z::Real)
    T = typeof(z)
    # Bounds and degeneracies on unit interval and conditioning value
    z ≤ 0    && return T(-Inf)
    z ≥ 1    && return T(0)
    D.uⱼ ≤ 0 && return T(-Inf)
    D.uⱼ ≥ 1 && return T(log(z))

    x = ϕ⁻¹(D.gen, z)
    y = ϕ⁻¹(D.gen, D.uⱼ)
    S = x + y
    S <= 0 && return T(-Inf)
    t  = D.j==2 ? _safett(y / S) : _safett(x/S)
    A0 = A(D.tail, t)
    A1 = dA(D.tail, t)
    r = D.j==2 ? (A0 + (1 - t) * A1)  : (A0 - t * A1) 
    r = max(r, T(0))
    return min(log(-ϕ⁽¹⁾(D.gen, S * A0)) + log(-ϕ⁻¹⁽¹⁾(D.gen, D.uⱼ) * r), T(0))
end
