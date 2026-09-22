"""
    tEVTail(ν, ρ)
    tEVTail(ν, R)
    tEVCopula{d}(ν, ρ)
    tEVCopula(d, ν, ρ)
    tEVCopula{d}(ν, R)
    tEVCopula(d, ν, R)

Extremal-`t` extreme-value copula with degrees of freedom `ν > 0`.

The internal representation is always a correlation matrix `R`. Scalar `ρ`
constructors are convenience adapters to the exchangeable correlation matrix.
For `d = 2`, `ρ` is exactly the off-diagonal matrix entry.

The complete-dependence limit `ρ = 1` is stored as the all-ones correlation
matrix in the same concrete tail type.

See also: [`TCopula`](@ref), [`ExtremeValueCopula`](@ref), [`ℓ`](@ref),
[`Distributions.fit`](@ref).

References:

* [nikoloulopoulos2009extreme](@cite) Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of multivariate t copulas. Extremes, 12, 129-148.
"""
tEVTail, tEVCopula

_tev_is_complete_correlation(R::AbstractMatrix) = all(isone, R)

Paramorph.@paramorph T struct tEVTail{d,T<:Real} <: BivariatePickandsTail
    ν::T ~ Paramorph.open_lower(zero(T))
    R::Matrix{T} ~ Paramorph.closed_correlation_matrix(d)
end

function _tev_tail_from_matrix(::Val{d}, ν::Real, R::AbstractMatrix) where {d}
    size(R) == (d, d) || throw(DimensionMismatch(
        "correlation matrix dimension $(size(R)) does not match d=$d",
    ))
    νf = float(ν)
    νf > 0 || throw(ArgumentError("ν must be > 0"))
    T = promote_type(typeof(νf), float(eltype(R)))
    RF = Matrix{T}(R)
    all(isfinite, RF) || throw(ArgumentError("R must contain only finite entries"))
    return tEVTail{d,T}(T(νf), RF)
end

function _tev_exchangeable_correlation(d::Int, ρ::Real)
    d >= 2 || throw(ArgumentError("extremal-t dimension must be at least two"))
    T = typeof(float(ρ))
    ρf = T(ρ)
    R = fill(ρf, d, d)
    @inbounds for i in 1:d
        R[i, i] = one(T)
    end
    return R
end

function (::Type{tEVTail{d}})(ν::Real, ρ::Real) where {d}
    d >= 2 || throw(ArgumentError("extremal-t dimension must be at least two"))
    νf, ρf = promote(float(ν), float(ρ))
    νf > 0 || throw(ArgumentError("ν must be > 0"))
    lower = -inv(d - 1)
    ρf > lower || throw(ArgumentError(
        "equicorrelation ρ must satisfy ρ > -1/(d-1) in dimension d=$d",
    ))
    ρf <= 1 || throw(ArgumentError("ρ must be ≤ 1"))
    return _tev_tail_from_matrix(
        Val(d), νf, _tev_exchangeable_correlation(d, ρf),
    )
end
tEVTail(ν::Real, ρ::Real) = tEVTail{2}(ν, ρ)

(::Type{tEVTail{d}})(ν::Real, R::AbstractMatrix) where {d} =
    _tev_tail_from_matrix(Val(d), ν, R)
function tEVTail(ν::Real, R::AbstractMatrix)
    size(R, 1) == size(R, 2) || throw(DimensionMismatch("correlation matrix must be square"))
    return tEVTail{size(R, 1)}(ν, R)
end

@inline function limit_kind(tail::tEVTail, ::Val)
    return _tev_is_complete_correlation(tail.R) ? M_LIMIT : NO_LIMIT
end

const tEVCopula{d,T} = ExtremeValueCopula{d,tEVTail{d,T}}
function (::Type{tEVCopula{d}})(ν::Real, ρ::Real) where {d}
    return _wrap_extreme_value(Val(d), tEVTail{d}(ν, ρ))
end
(::Type{tEVCopula})(d::Int, ν::Real, ρ::Real) =
    _wrap_extreme_value(Val(d), tEVTail{d}(ν, ρ))
function (::Type{tEVCopula{d}})(ν::Real, R::AbstractMatrix) where {d}
    return _wrap_extreme_value(Val(d), tEVTail(ν, R))
end
function (::Type{tEVCopula})(d::Int, ν::Real, R::AbstractMatrix)
    return _wrap_extreme_value(Val(d), tEVTail(ν, R))
end

_is_valid_in_dim(::tEVTail{D}, d::Int) where {D} = D == d

_tev_rho(tail::tEVTail{2}) = tail.R[1, 2]
function _tev_correlation(tail::tEVTail{D}, d::Int) where {D}
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    return tail.R
end

Distributions.params(C::ExtremeValueCopula{2,<:tEVTail}) =
    (C.tail.ν, _tev_rho(C.tail))
Distributions.params(C::ExtremeValueCopula{D,<:tEVTail}) where {D} =
    (C.tail.ν, copy(C.tail.R))

_tail_constructor_parameter_names(::Type{<:tEVTail{2}}, _) = (:ν, :ρ)
_tail_constructor_parameter_names(::Type{<:tEVTail}, _) = (:ν, :R)

_available_fitting_methods(
    ::Type{<:ExtremeValueCopula{D,<:tEVTail} where D}, d,
) = d == 2 ? (:mle,) : ()

function _tev_stdf(ν::Real, R::AbstractMatrix, x)
    d = length(x)

    active = findall(xi -> xi > 0, x)
    isempty(active) && return 0.0
    length(active) == 1 && return Float64(x[only(active)])

    xf = Float64.(x[active])
    Rf = Matrix{Float64}(R[active, active])
    m = length(xf)

    scale = maximum(xf)
    y = xf ./ scale
    νf = Float64(ν)
    total = 0.0

    for j in 1:m
        J = [k for k in 1:m if k != j]
        r = Rf[J, j]
        Σcond = (
            Rf[J, J] - r * transpose(r)
        ) / (νf + 1.0)
        Σcond = Matrix(LinearAlgebra.Symmetric(Σcond))

        upper = [
            (y[j] / y[k])^(1 / νf)
            for k in J
        ]

        p = _mvtcdf(νf + 1.0, r, Σcond, upper)
        total += y[j] * p
    end

    return scale * total
end

function ℓ(tail::tEVTail{D}, x) where {D}
    limit_kind(tail, Val(D)) === M_LIMIT && return maximum(x)
    if D == 2
        x1, x2 = x
        s = x1 + x2
        iszero(s) && return zero(s)
        return s * A(tail, x1 / s)
    end
    return _tev_stdf(tail.ν, tail.R, x)
end

function _ellpartial_signlog(tail::tEVTail, x, I::Tuple{Vararg{Int}})
    d = length(x)
    R = _tev_correlation(tail, d)
    ν = tail.ν
    isempty(I) && return 1, log(_tev_stdf(ν, R, x))

    all(xi -> xi >= 0, x) || return 0, -Inf
    all(i -> x[i] > 0, I) || return 0, -Inf

    z = [iszero(xi) ? Inf : inv(Float64(xi)) for xi in x]
    b = length(I)
    b > 0 || throw(ArgumentError("the differentiation block must be nonempty"))

    νf = Float64(ν)
    Bv = collect(I)
    C = [i for i in eachindex(z) if i ∉ I]
    zB = Float64.(z[Bv])

    all(zi -> zi > 0, zB) || return 0, -Inf

    RB = Matrix{Float64}(R[Bv, Bv])
    FB = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(RB))
    rB = zB .^ (1 / νf)
    solved = FB \ rB
    q = LinearAlgebra.dot(rB, solved)

    logdetRB = 2 * sum(log, LinearAlgebra.diag(FB.L))
    logλB =
        (1 - b) * log(νf) +
        ((1 - b) / 2) * log(pi) -
        0.5 * logdetRB +
        SpecialFunctions.loggamma((νf + b) / 2) -
        SpecialFunctions.loggamma((νf + 1) / 2) +
        (1 / νf - 1) * sum(log, zB) -
        ((νf + b) / 2) * log(q)

    logq = if isempty(C)
        logλB
    else
        RCB = Matrix{Float64}(R[C, Bv])
        RBC = Matrix{Float64}(R[Bv, C])
        RCC = Matrix{Float64}(R[C, C])

        μ = RCB * solved
        base = RCC - RCB * (FB \ RBC)
        Σcond = (q / (νf + b)) .* base
        Σcond = Matrix(LinearAlgebra.Symmetric(Σcond))

        upper = Vector{Float64}(undef, length(C))
        @inbounds for (a, i) in enumerate(C)
            zi = z[i]
            upper[a] = isinf(zi) ? Inf : Float64(zi)^(1 / νf)
        end

        p = _mvtcdf(νf + b, μ, Σcond, upper)
        iszero(p) ? -Inf : logλB + log(p)
    end

    isfinite(logq) || return 0, -Inf

    logjac = 2 * sum(log(Float64(x[i])) for i in I)
    logabs = logq - logjac
    return isodd(length(I)) ? 1 : -1, logabs
end

function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d,<:tEVTail},
    X::AbstractMatrix{T},
) where {d,T<:Real}
    return _rand_with_ev_limits!(rng, C, X) do
        ν = C.tail.ν
        R = C.tail.R
        cache = ntuple(d) do m
            J = [i for i in 1:d if i != m]
            r = Vector{Float64}(R[J, m])
            Σ = Matrix{Float64}(R[J, J]) - r * transpose(r)
            F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ))
            (; J, r, F)
        end
        logq = Vector{Float64}(undef, d)
        logz = Vector{Float64}(undef, d)

        @inbounds for col in axes(X, 2)
            fill!(logz, -Inf)
            s = 0.0

            while true
                s += Random.randexp(rng) / d
                logradius = -log(s)

                if all(isfinite, logz) && logradius <= minimum(logz)
                    break
                end

                m = Random.rand(rng, 1:d)
                entry = cache[m]

                wm = sqrt(Random.rand(rng, Distributions.Chisq(Float64(ν) + 1.0)))
                fill!(logq, -Inf)
                logq[m] = Float64(ν) * log(wm)

                q = length(entry.J)
                if q > 0
                    ξ = Random.randn(rng, q)
                    wJ = entry.r .* wm .+ entry.F.L * ξ
                    for a in 1:q
                        wi = wJ[a]
                        wi > 0 && (logq[entry.J[a]] = Float64(ν) * log(wi))
                    end
                end

                logsum = LogExpFunctions.logsumexp(logq)
                for i in eachindex(logq)
                    logq[i] -= logsum
                end

                for i in 1:d
                    candidate = logradius + logq[i]
                    if candidate > logz[i]
                        logz[i] = candidate
                    end
                end
            end

            for i in 1:d
                X[i, col] = T(exp(-exp(-logz[i])))
            end
        end

        return X
    end
end

function A(tail::tEVTail{2}, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    tt = _safett(t)
    isone(ρ) && return max(tt, one(tt) - tt)
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    om = 1 - tt
    log_t = log(tt)
    log_om = log1p(-tt)
    log_r = log_t - log_om
    log_s = log_om - log_t

    rα = exp(α * log_r)
    sα = exp(α * log_s)

    Z1 = C * (rα - ρ)
    Z2 = C * (sα - ρ)

    D = Distributions.TDist(ν + 1)
    F1 = Distributions.cdf(D, Z1)
    F2 = Distributions.cdf(D, Z2)

    return tt * F1 + om * F2
end
function dA(tail::tEVTail{2}, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν
    tt = _safett(t)
    om = 1 - tt
    log_t = log(tt)
    log_om = log1p(-tt)
    log_r = log_t - log_om
    log_s = log_om - log_t

    rα = exp(α * log_r)
    rαm1 = exp((α - 1) * log_r)
    sα = exp(α * log_s)
    sαm1 = exp((α - 1) * log_s)

    Z1 = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv(om)^2

    Z2 = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv(tt)^2)

    D = Distributions.TDist(ν + 1)
    f1 = Distributions.pdf(D, Z1)
    F1 = Distributions.cdf(D, Z1)
    f2 = Distributions.pdf(D, Z2)
    F2 = Distributions.cdf(D, Z2)

    DB1 = tt * f1 * DZ1 + F1
    DB2 = om * f2 * DZ2 - F2
    return DB1 + DB2
end
function d²A(tail::tEVTail{2}, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    log_t = log(tt)
    log_om = log1p(-tt)
    log_r = log_t - log_om
    log_s = log_om - log_t

    rα = exp(α * log_r)
    rαm1 = exp((α - 1) * log_r)
    rαm2 = exp((α - 2) * log_r)
    sα = exp(α * log_s)
    sαm1 = exp((α - 1) * log_s)
    sαm2 = exp((α - 2) * log_s)

    inv_om = inv(om)
    inv_om2 = inv_om^2
    inv_om3 = inv_om2 * inv_om
    inv_om4 = inv_om2^2
    inv_t = inv(tt)
    inv_t2 = inv_t^2
    inv_t3 = inv_t2 * inv_t
    inv_t4 = inv_t2^2

    Z1 = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv_om2
    DDZ1 = C * α * (2 * rαm1 * inv_om3 + (α - 1) * rαm2 * inv_om4)

    Z2 = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv_t2)
    DDZ2 = C * α * ((α - 1) * sαm2 * inv_t4 + 2 * sαm1 * inv_t3)

    D = Distributions.TDist(ν + 1)
    f1 = Distributions.pdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)
    f2 = Distributions.pdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)

    DDB1 = 2 * f1 * DZ1 + tt * (g1 * f1 * DZ1^2 + f1 * DDZ1)
    DDB2 = om * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2
    return DDB1 + DDB2
end
function _A_dA_d²A(tail::tEVTail{2}, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    log_t = log(tt)
    log_om = log1p(-tt)
    log_r = log_t - log_om
    log_s = log_om - log_t

    rα = exp(α * log_r)
    rαm1 = exp((α - 1) * log_r)
    rαm2 = exp((α - 2) * log_r)
    sα = exp(α * log_s)
    sαm1 = exp((α - 1) * log_s)
    sαm2 = exp((α - 2) * log_s)

    inv_om = inv(om)
    inv_om2 = inv_om^2
    inv_om3 = inv_om2 * inv_om
    inv_om4 = inv_om2^2
    inv_t = inv(tt)
    inv_t2 = inv_t^2
    inv_t3 = inv_t2 * inv_t
    inv_t4 = inv_t2^2

    Z1 = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv_om2
    DDZ1 = C * α * (2 * rαm1 * inv_om3 + (α - 1) * rαm2 * inv_om4)

    Z2 = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv_t2)
    DDZ2 = C * α * ((α - 1) * sαm2 * inv_t4 + 2 * sαm1 * inv_t3)

    D = Distributions.TDist(ν + 1)

    f1 = Distributions.pdf(D, Z1)
    F1 = Distributions.cdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)

    f2 = Distributions.pdf(D, Z2)
    F2 = Distributions.cdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)

    B1 = tt * F1
    DB1 = tt * f1 * DZ1 + F1
    DDB1 = 2 * f1 * DZ1 + tt * (g1 * f1 * DZ1^2 + f1 * DDZ1)

    B2 = om * F2
    DB2 = om * f2 * DZ2 - F2
    DDB2 = om * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2

    Aval = B1 + B2
    DA = DB1 + DB2
    DDA = DDB1 + DDB2
    return Aval, DA, DDA
end