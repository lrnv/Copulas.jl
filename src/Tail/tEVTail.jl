"""
    tEVTail(ν, ρ)
    tEVTail(ν, R)
    tEVCopula{d}(ν, ρ)
    tEVCopula(d, ν, ρ)
    tEVCopula{d}(ν, R)
    tEVCopula(d, ν, R)

Extremal-`t` extreme-value copula with degrees of freedom `ν > 0`.

`tEVCopula(d, ν, ρ)` uses an exchangeable correlation matrix with common
off-diagonal correlation `ρ`. For a non-degenerate `d`-dimensional model,

```math
-\\frac{1}{d-1}<\\rho<1.
```

`tEVCopula{d}(ν, R)` uses a general correlation matrix `R`. `R` must be
`d×d`, finite, symmetric, have unit diagonal, and be
strictly positive definite in the non-degenerate general representation.
A valid `2×2` matrix is equivalent to the scalar parameter given by its
off-diagonal entry `ρ`.

For `d = 2`, the Pickands dependence function is

```math
A(x)=x\\,t_{\\nu+1}(Z_x)+(1-x)t_{\\nu+1}(Z_{1-x}),
```

where

```math
Z_x
=
\\sqrt{\\frac{1+\\nu}{1-\\rho^2}}
\\left[\\left(\\frac{x}{1-x}\\right)^{1/\\nu}-\\rho\\right].
```

Special case:

* `ρ = 1` represents `MCopula(d)`.

Correlation matrices close to singularity represent strong or constrained
dependence and can make multivariate probability and likelihood calculations
ill-conditioned. The degrees of freedom and correlation jointly determine tail
dependence; neither is a standalone strength parameter.

See also: [`TCopula`](@ref), [`ExtremeValueCopula`](@ref), [`ℓ`](@ref),
[`Distributions.fit`](@ref).

References:

* [nikoloulopoulos2009extreme](@cite) Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of multivariate t copulas. Extremes, 12, 129-148.
"""
tEVTail, tEVCopula

_tev_parameter_geometry(::Val{:exchangeable}, d::Int, ::Type{T}) where {T<:Real} =
    Paramorph.bounded_interval(-inv(T(d - 1)), one(T); left_closed=false)
_tev_parameter_geometry(::Val{:general}, d::Int, ::Type{T}) where {T<:Real} =
    Paramorph.correlation_matrix(d)

Paramorph.@paramorph T struct tEVTail{d,R,T<:Real} <: BivariatePickandsTail
    ν::T ~ Paramorph.open_lower(zero(T))
    parameter::Union{T,Matrix{T}} ~ _tev_parameter_geometry(Val(R), d, T)
end

function _tev_general_tail(::Val{d}, ν::Real, R::AbstractMatrix) where {d}
    size(R) == (d, d) || throw(ArgumentError(
        "correlation matrix dimension $(size(R)) does not match d=$d",
    ))
    νf = float(ν)
    νf > 0 || throw(ArgumentError("ν must be > 0"))
    T = promote_type(typeof(νf), float(eltype(R)))
    RF = Matrix{T}(R)
    all(isfinite, RF) || throw(ArgumentError("R must contain only finite entries"))

    # The all-ones matrix is the complete-dependence boundary and is represented
    # exactly by the exchangeable scalar chart. The general correlation chart is
    # the strict positive-definite interior.
    if all(isone, RF)
        return tEVTail{d,:exchangeable,T}(T(νf), one(T))
    end

    try
        return tEVTail{d,:general,T}(T(νf), RF)
    catch err
        (err isa DomainError || err isa LinearAlgebra.PosDefException) || rethrow()
        throw(ArgumentError("R must be a strict correlation matrix"))
    end
end

function (::Type{tEVTail{d}})(ν::Real, ρ::Real) where {d}
    d >= 2 || throw(ArgumentError("extremal-t dimension must be at least two"))
    νf, ρf = promote(float(ν), float(ρ))
    νf > 0 || throw(ArgumentError("ν must be > 0"))
    lower = -inv(d - 1)
    ρf > lower || throw(ArgumentError("equicorrelation ρ must satisfy ρ > -1/(d-1) in dimension d=$d"))
    ρf <= 1 || throw(ArgumentError("ρ must be ≤ 1"))

    # In d=2 every correlation matrix is exchangeable, so interior scalar input
    # is canonicalized to the general matrix representation. Keep ρ=1 in the
    # scalar chart because the general chart is intentionally strict SPD.
    if d == 2 && ρf < 1
        Tρ = typeof(ρf)
        M = fill(Tρ(ρf), 2, 2)
        M[1, 1] = M[2, 2] = one(Tρ)
        return _tev_general_tail(Val(2), νf, M)
    end
    T = typeof(νf + ρf)
    return tEVTail{d,:exchangeable,T}(T(νf), T(ρf))
end
tEVTail(ν::Real, ρ::Real) = tEVTail{2}(ν, ρ)

(::Type{tEVTail{d}})(ν::Real, R::AbstractMatrix) where {d} =
    _tev_general_tail(Val(d), ν, R)
tEVTail(ν::Real, R::AbstractMatrix) =
    tEVTail{size(R, 1)}(ν, R)

@inline _tev_representation(::tEVTail{d,R}) where {d,R} = R
@inline function limit_kind(tail::tEVTail, ::Val)
    _tev_representation(tail) === :exchangeable && isone(tail.parameter) ? M_LIMIT : NO_LIMIT
end

const tEVCopula{d,R,T} = ExtremeValueCopula{d,tEVTail{d,R,T}}
function (::Type{tEVCopula{d}})(ν::Real, ρ::Real) where {d}
    return _wrap_extreme_value(Val(d), tEVTail{d}(ν, ρ))
end
(::Type{tEVCopula})(d::Int, ν::Real, ρ::Real) =
    _wrap_extreme_value(Val(d), tEVTail{d}(ν, ρ))
function (::Type{tEVCopula{d}})(ν::Real, R::AbstractMatrix) where {d}
    return _wrap_extreme_value(Val(d), tEVTail{d}(ν, R))
end
function (::Type{tEVCopula})(d::Int, ν::Real, R::AbstractMatrix)
    return _wrap_extreme_value(Val(d), tEVTail(ν, R))
end

_is_valid_in_dim(::tEVTail{D}, d::Int) where {D} = D == d

_tev_rho(tail::tEVTail{D,:exchangeable}) where {D} = tail.parameter
_tev_rho(tail::tEVTail{D,:general}) where {D} = tail.parameter[1, 2]
function _tev_correlation(tail::tEVTail{D,:exchangeable}, d::Int) where {D}
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    ρ = tail.parameter
    R = fill(float(ρ), d, d)
    @inbounds for i in 1:d
        R[i, i] = one(eltype(R))
    end
    return R
end
_tev_correlation(tail::tEVTail{D,:general}, d::Int) where {D} = begin
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    tail.parameter
end

# Preserve the public natural representation while canonicalizing d=2 storage.
Distributions.params(C::ExtremeValueCopula{2,<:tEVTail}) =
    (C.tail.ν, _tev_rho(C.tail))
Distributions.params(C::ExtremeValueCopula{D,<:tEVTail{D,:exchangeable}}) where {D} =
    (C.tail.ν, C.tail.parameter)
Distributions.params(C::ExtremeValueCopula{D,<:tEVTail{D,:general}}) where {D} =
    (C.tail.ν, copy(C.tail.parameter))

_tail_constructor_parameter_names(::Type{<:tEVTail{2}}, _) = (:ν, :ρ)
_tail_constructor_parameter_names(::Type{<:tEVTail{D,:exchangeable}}, _) where {D} = (:ν, :ρ)
_tail_constructor_parameter_names(::Type{<:tEVTail{D,:general}}, _) where {D} = (:ν, :R)

_available_fitting_methods(
    ::Type{<:ExtremeValueCopula{D,<:tEVTail{D,:general}} where D}, d,
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

function ℓ(tail::tEVTail{D,:exchangeable}, x) where {D}
    isone(tail.parameter) && return maximum(x)
    d = length(x)

    # Preserve the historical closed bivariate route. It is analytic,
    # numerically stable, and compatible with ForwardDiff.
    if d == 2
        x1, x2 = x
        s = x1 + x2
        iszero(s) && return zero(s)
        return s * A(tail, x1 / s)
    end

    R = _tev_correlation(tail, d)
    return _tev_stdf(tail.ν, R, x)
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


function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:tEVTail}, X::AbstractMatrix{T}) where {d,T<:Real}
    return _rand_with_ev_limits!(rng, C, X) do
        ν = C.tail.ν
        R = _tev_correlation(C.tail, d)
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

                # All future radii are smaller. Since normalized spectral weights
                # satisfy Q_i ≤ 1, no future point can improve any coordinate once
                # the next radius lies below the current componentwise minimum.
                if all(isfinite, logz) && logradius <= minimum(logz)
                    break
                end

                m = Random.rand(rng, 1:d)
                entry = cache[m]

                # Size-biasing the Gaussian spectral vector by (W_m^+)^ν gives
                # W_m² ~ χ²_{ν+1}, with the positive square root.
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

ℓ(tail::tEVTail{D,:general}, x) where {D} =
    _tev_stdf(tail.ν, tail.parameter, x)

function A(tail::tEVTail, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    tt = _safett(t)
    isone(ρ) && return max(tt, one(tt) - tt)
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    om = 1 - tt
    # log-ratios for stability
    log_t  = log(tt)
    log_om = log1p(-tt) # = log(1 - t)
    log_r  = log_t - log_om           # log(t/(1-t))
    log_s  = log_om - log_t           # log((1-t)/t)

    rα = exp(α * log_r)
    sα = exp(α * log_s)

    Z1 = C * (rα - ρ)
    Z2 = C * (sα - ρ)

    D = Distributions.TDist(ν + 1)
    F1 = Distributions.cdf(D, Z1)
    F2 = Distributions.cdf(D, Z2)

    return tt * F1 + om * F2
end
function dA(tail::tEVTail, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν
    tt = _safett(t)
    om = 1 - tt
    log_t  = log(tt)
    log_om = log1p(-tt)
    log_r  = log_t - log_om
    log_s  = log_om - log_t

    rα    = exp(α * log_r)
    rαm1  = exp((α - 1) * log_r)
    sα    = exp(α * log_s)
    sαm1  = exp((α - 1) * log_s)

    Z1  = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv(om)^2

    Z2  = C * (sα - ρ)
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
function d²A(tail::tEVTail, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    log_t  = log(tt)
    log_om = log1p(-tt)
    log_r  = log_t - log_om
    log_s  = log_om - log_t

    rα    = exp(α * log_r)
    rαm1  = exp((α - 1) * log_r)
    rαm2  = exp((α - 2) * log_r)
    sα    = exp(α * log_s)
    sαm1  = exp((α - 1) * log_s)
    sαm2  = exp((α - 2) * log_s)

    inv_om  = inv(om)
    inv_om2 = inv_om^2
    inv_om3 = inv_om2 * inv_om
    inv_om4 = inv_om2^2
    inv_t   = inv(tt)
    inv_t2  = inv_t^2
    inv_t3  = inv_t2 * inv_t
    inv_t4  = inv_t2^2

    Z1  = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv_om2
    # d²Z1/dt² using product rule on r^(α-1) * (1-t)^(-2)
    DDZ1 = C * α * ( 2 * rαm1 * inv_om3 + (α - 1) * rαm2 * inv_om4 )

    Z2  = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv_t2)
    # d²Z2/dt² with s = (1-t)/t, s'=-1/t², s''=2/t³
    DDZ2 = C * α * ( (α - 1) * sαm2 * inv_t4 + 2 * sαm1 * inv_t3 )

    D = Distributions.TDist(ν + 1)
    f1 = Distributions.pdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)
    f2 = Distributions.pdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)

    DDB1 = 2 * f1 * DZ1 + tt * (g1 * f1 * DZ1^2 + f1 * DDZ1)
    DDB2 = om * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2
    return DDB1 + DDB2
end
function _A_dA_d²A(tail::tEVTail, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
    om = 1 - tt
    log_t  = log(tt)
    log_om = log1p(-tt)
    log_r  = log_t - log_om
    log_s  = log_om - log_t

    rα    = exp(α * log_r)
    rαm1  = exp((α - 1) * log_r)
    rαm2  = exp((α - 2) * log_r)
    sα    = exp(α * log_s)
    sαm1  = exp((α - 1) * log_s)
    sαm2  = exp((α - 2) * log_s)

    inv_om  = inv(om)
    inv_om2 = inv_om^2
    inv_om3 = inv_om2 * inv_om
    inv_om4 = inv_om2^2
    inv_t   = inv(tt)
    inv_t2  = inv_t^2
    inv_t3  = inv_t2 * inv_t
    inv_t4  = inv_t2^2

    Z1  = C * (rα - ρ)
    DZ1 = C * α * rαm1 * inv_om2
    DDZ1 = C * α * ( 2 * rαm1 * inv_om3 + (α - 1) * rαm2 * inv_om4 )

    Z2  = C * (sα - ρ)
    DZ2 = C * α * sαm1 * (-inv_t2)
    DDZ2 = C * α * ( (α - 1) * sαm2 * inv_t4 + 2 * sαm1 * inv_t3 )

    D = Distributions.TDist(ν + 1)
    
    f1 = Distributions.pdf(D, Z1)
    F1 = Distributions.cdf(D, Z1)
    g1 = Distributions.gradlogpdf(D, Z1)
    
    f2 = Distributions.pdf(D, Z2)
    F2 = Distributions.cdf(D, Z2)
    g2 = Distributions.gradlogpdf(D, Z2)
    
    B1  = tt * F1
    DB1 = tt * f1 * DZ1 + F1
    DDB1 = 2 * f1 * DZ1 + tt * (g1 * f1 * DZ1^2 + f1 * DDZ1)
    
    B2  = om * F2
    DB2 = om * f2 * DZ2 - F2
    DDB2 = om * (g2 * f2 * DZ2^2 + f2 * DDZ2) - 2 * f2 * DZ2

    A  = B1 + B2
    DA = DB1 + DB2
    DDA = DDB1 + DDB2
    return A, DA, DDA
end
