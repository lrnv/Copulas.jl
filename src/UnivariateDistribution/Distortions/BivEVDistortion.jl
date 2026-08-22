###########################################################################
#####  Bivariate Extreme Value Copulas fast-path (d=2, p=1)
###########################################################################
struct BivEVDistortion{TT,T} <: Distortion
    tail::TT
    j::Int8
    uⱼ::T
    negloguⱼ::T
end
function BivEVDistortion(tail, j::Int8, uⱼ::Real)
    uⱼ = float(uⱼ)
    negloguⱼ = uⱼ > zero(uⱼ) ? -log(uⱼ) : typeof(uⱼ)(Inf)
    return BivEVDistortion{typeof(tail),typeof(uⱼ)}(tail, j, uⱼ, negloguⱼ)
end

@inline function _biv_ev_endpoint_shape(D::BivEVDistortion, lower::Bool, ::Type{T}) where T
    use_left = (D.j == 2) == lower
    slope = use_left ? _pickands_left_slope(D.tail, one(T)) :
                       _pickands_right_slope(D.tail, one(T))
    return max(one(T) + (use_left ? slope : -slope), zero(T))
end

@inline function _biv_ev_endpoint_logcdf(D, z, lower::Bool, ::Type{T}) where T
    κ = _biv_ev_endpoint_shape(D, lower, T)
    lower && return iszero(κ) ? zero(T) : κ * log(T(z))
    return iszero(κ) ? T(-Inf) : log(T(z)) + log(κ)
end

@inline function _biv_ev_endpoint_logpdf(D, z, lower::Bool, ::Type{T}) where T
    κ = _biv_ev_endpoint_shape(D, lower, T)
    iszero(κ) && return T(-Inf)
    return lower ? log(κ) + (κ - one(T)) * log(T(z)) : log(κ)
end

@inline function _biv_ev_endpoint_quantile(D, p, lower::Bool, ::Type{T}) where T
    pT = T(p)
    κ = _biv_ev_endpoint_shape(D, lower, T)
    lower && return iszero(κ) ? zero(T) : pT^inv(κ)
    iszero(pT) && return zero(T)
    iszero(κ) && return one(T)
    return pT <= κ ? pT / κ : one(T)
end

@inline _ev_kink_tol(x, y) = 16 * _δ(x) * max(one(x), abs(x), abs(y))
@inline _ev_lt(x, y) = x < y - _ev_kink_tol(x, y)
@inline _ev_le(x, y) = x <= y + _ev_kink_tol(x, y)

function Distributions.logcdf(D::BivEVDistortion{TT,TF1}, z::Real) where {TT,TF1}
    T = promote_type(typeof(z), TF1)
    z ≤ 0 && return T(-Inf)
    z ≥ 1 && return T(0)
    D.uⱼ ≤ 0 && return _biv_ev_endpoint_logcdf(D, z, true, T)
    D.uⱼ ≥ 1 && return _biv_ev_endpoint_logcdf(D, z, false, T)

    if D.j == 2
        # Condition on the second variable : V = D.uⱼ, free = u=z
        x, y = -log(z), D.negloguⱼ
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        tolog = Aw - w * dAw
        logval = -s * Aw + y
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = D.negloguⱼ, -log(z)
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
    z ≤ 0 && return T(-Inf)
    z ≥ 1 && return T(-Inf)
    D.uⱼ ≤ 0 && return _biv_ev_endpoint_logpdf(D, z, true, T)
    D.uⱼ ≥ 1 && return _biv_ev_endpoint_logpdf(D, z, false, T)

    if D.j == 2
        # Condition on the second variable : V = D.uⱼ, free = u=z
        x, y = -log(z), D.negloguⱼ
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        ddAw = d²A(D.tail, w)
        Tval = Aw - w * dAw
        Tval ≤ 0 && return T(-Inf)

        logval = -s * Aw + y
        # derivatives
        # logval' = (Aw + (y/s)*dAw) / z
        lvp = (Aw + (y / s) * dAw) / z
        # T'(w)*dw/dz = w * ddAw * y / (z * s^2)
        tp_term = w * ddAw * y / (z * s^2)
        B = tp_term + Tval * lvp
        B ≤ 0 && return T(-Inf)

        return logval + log(B)
    else
        # Condition on the first variable : U = D.uⱼ, free = v=z
        x, y = D.negloguⱼ, -log(z)
        s = x + y
        w = x / s
        Aw, dAw = A(D.tail, w), dA(D.tail, w)
        ddAw = d²A(D.tail, w)
        Tval = Aw + (1 - w) * dAw
        Tval ≤ 0 && return T(-Inf)

        logval = -s * Aw + x
        # derivatives
        # logval' = (Aw - (x/s)*dAw) / z
        lvp = (Aw - (x / s) * dAw) / z
        # T'(w)*dw/dz = x * (1 - w) * ddAw / (z * s^2)
        tp_term = x * (1 - w) * ddAw / (z * s^2)
        B = tp_term + Tval * lvp
        B ≤ 0 && return T(-Inf)

        return logval + log(B)
    end
end
