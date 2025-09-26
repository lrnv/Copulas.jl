"""
        BivEVDistortion(tail, j, u_j)

Parameters
    * `tail` – extreme value tail / STDF object
    * `j ∈ {1,2}` – conditioned coordinate index
    * `u_j ∈ (0,1)` – conditioning value

Bivariate extreme value conditional distortion (d=2, p=1) used for fast
Rosenblatt / inverse transforms in extreme value copulas.
"""
struct BivEVDistortion{TT,T} <: Distortion
    tail::TT
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::BivEVDistortion, z::Real)
    T = typeof(z)
    # bounds and degeneracies
    z ≤ 0    && return T(-Inf)   # P(X ≤ 0) = 0
    z ≥ 1    && return T(0)      # P(X ≤ 1) = 1
    D.uⱼ ≤ 0 && return T(-Inf)   # conditioning on 0 ⇒ degenerate at 0
    D.uⱼ ≥ 1 && return T(log(z)) # conditioning on 1 ⇒ uniform[0,1], so logcdf = log(z)

    if D.j == 2
        # Condition on the second variable : V = D.uⱼ, free = u=z
        x, y = -log(z), -log(D.uⱼ)
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        tolog = Aw - w * dAw
        logval = -s * Aw + y
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = -log(D.uⱼ), -log(z)
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        tolog = Aw + (1 - w) * dAw
        logval = -s * Aw + x
    end

    # upper clip but no lower clip
    return min(logval + log(max(tolog, T(0))), T(0))
end