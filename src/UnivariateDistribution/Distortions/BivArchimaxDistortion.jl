###########################################################################
#####  Bivariate Archimax Copulas fast-path (d=2, p=1)
###########################################################################

# Conditional U_i | U_j=u_j for ArchimaxCopula with generator ϕ and EV tail A
# C(u,v) = ϕ( (x+y) A(t) ), where x=ϕ⁻¹(u), y=ϕ⁻¹(v), t=y/(x+y)
# For j=2 (condition on v), the 1D conditional CDF is
#   H_{1|2}(u | v) = ∂/∂v C(u,v) (when j==2)
#   H_{2|1}(u | v) = ∂/∂v C(v,u) (where j==1)
# and analogously for j=1.
# We implement a numerically stable logcdf based on Archimax factorization.

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
function Distributions.logpdf(D::BivArchimaxDistortion, z::Real)
    T = typeof(z)
    z ≤ 0    && return T(-Inf)
    z ≥ 1    && return T(-Inf)
    D.uⱼ ≤ 0 && return T(-Inf)
    D.uⱼ ≥ 1 && return T(0)

    x = ϕ⁻¹(D.gen, z)
    y = ϕ⁻¹(D.gen, D.uⱼ)
    S = x + y
    S <= 0 && return T(-Inf)
    t  = D.j==2 ? _safett(y / S) : _safett(x / S)

    A0 = A(D.tail, t)
    A1 = dA(D.tail, t)
    A2 = d2A(D.tail, t)
    r = D.j==2 ? (A0 + (1 - t) * A1)  : (A0 - t * A1)
    r = max(r, T(0))

    G = S * A0
    phi1G = ϕ⁽¹⁾(D.gen, G)
    phi2G = ϕ⁽²⁾(D.gen, G)

    inv_cond = ϕ⁻¹⁽¹⁾(D.gen, D.uⱼ)   # derivative of inverse at conditioning value
    dx_dz   = ϕ⁻¹⁽¹⁾(D.gen, z)      # derivative of inverse at z

    dGdx = D.j==2 ? (A0 - t * A1) : (A0 + (1 - t) * A1)
    drdx = - (t * (1 - t) / S) * A2

    inner = phi2G * dGdx * r + phi1G * drdx
    val = inv_cond * dx_dz * inner

    val <= zero(T) && return T(-Inf)
    return T(log(val))
end