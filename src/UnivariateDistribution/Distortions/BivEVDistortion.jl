###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TT,T} <: Distortion
    tail::TT
    j::Int8
    uⱼ::T
end
function Distributions.logcdf(D::BivEVDistortion{TT,TF1}, z::Real) where {TT,TF1}
    T = promote_type(typeof(z), TF1)
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

function Distributions.logpdf(D::BivEVDistortion{TT,TF1}, z::Real) where {TT,TF1}
    T = promote_type(typeof(z), TF1)
    # Support and degeneracies
    z ≤ 0    && return T(-Inf)
    z ≥ 1    && return T(-Inf)
    D.uⱼ ≤ 0 && return T(-Inf)
    # Conditioning at 1 ⇒ Uniform(0,1): pdf ≡ 1 ⇒ logpdf = 0
    D.uⱼ ≥ 1 && return T(0)

    # EV copula density at (z, uⱼ): c(u,v) = exp(-ℓ) * (ℓ_x ℓ_y - ℓ_{xy}) / (u v)
    x = -log(z)
    y = -log(D.uⱼ)
    val, du, dv, dudv = _biv_der_ℓ(D.tail, (x, y))
    core = -dudv + du*dv
    (core <= 0 || !isfinite(core)) && return T(-Inf)
    return -val + log(core) + x + y
end