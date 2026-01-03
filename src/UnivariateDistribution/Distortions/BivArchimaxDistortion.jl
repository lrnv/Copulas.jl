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
    # Support and degeneracies
    z ≤ 0    && return T(-Inf)
    z ≥ 1    && return T(-Inf)
    D.uⱼ ≤ 0 && return T(-Inf)
    # If conditioning value is 1, the conditional is Uniform(0,1): logpdf = 0
    D.uⱼ ≥ 1 && return T(0)

    # Setup common quantities
    x = ϕ⁻¹(D.gen, z)
    y = ϕ⁻¹(D.gen, D.uⱼ)
    S = x + y
    S <= 0 && return T(-Inf)
    t  = D.j==2 ? _safett(y / S) : _safett(x / S)
    A0 = A(D.tail, t)
    A1 = dA(D.tail, t)
    A2 = d²A(D.tail, t)
    # r and its derivative wrt x
    r  = D.j==2 ? (A0 + (1 - t) * A1)  : (A0 - t * A1)
    r  = max(r, T(0))
    dt_dx = (D.j==2 ? -(t / S) : ((1 - t) / S))
    dr_dx = -(t * (1 - t) / S) * A2
    # U = S * A(t), dU/dx
    dU_dx = D.j==2 ? (A0 - t * A1) : (A0 + (1 - t) * A1)
    U = S * A0
    # Generator derivatives at U
    ϕ1 = ϕ⁽¹⁾(D.gen, U)
    ϕ2 = ϕ⁽ᵏ⁾(D.gen, 2, U)
    # g' = -ϕ2(U) * dU_dx * r + (-ϕ1(U)) * dr_dx
    term1 = -ϕ2 * dU_dx * r
    term2 = -ϕ1 * dr_dx
    gprime = term1 + term2
    # Full derivative: h'(z) = (-ϕ⁻¹)'(u_j) * g'(z) * (ϕ⁻¹)'(z)
    K     = -ϕ⁻¹⁽¹⁾(D.gen, D.uⱼ)
    dx_dz =  ϕ⁻¹⁽¹⁾(D.gen, z)
    val = K * gprime * dx_dz
    (val <= 0 || !isfinite(val)) && return T(-Inf)
    return log(val)
end
