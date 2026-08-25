"""
    tEVTail{T,P}, tEVCopula{d,T}

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
Scalar and matrix parameters are stored by the same `tEVTail` type. A valid
`2×2` matrix retains its matrix representation while the specialized
bivariate analytic kernel recovers `ρ` from its off-diagonal entry.

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

* `ρ = 1` returns `MCopula(d)`.

References:

* [nikoloulopoulos2009extreme](@cite) Nikoloulopoulos, A. K., Joe, H., & Li, H. (2009). Extreme value properties of multivariate t copulas. Extremes, 12, 129-148.
"""
tEVTail, tEVCopula

struct tEVTail{T,P} <: BivariatePickandsTail
    ν::T
    parameter::P
    function tEVTail(ν::Real, ρ::Real)
        (ν > 0)     || throw(ArgumentError("ν must be > 0"))
        (-1 < ρ ≤ 1)|| throw(ArgumentError("ρ must be in (-1,1]"))
        ρ == 1 && return MTail()
        νT, ρT = promote(float(ν), float(ρ))
        return new{typeof(νT),typeof(ρT)}(νT, ρT)
    end
    function tEVTail(ν::Real, R::AbstractMatrix)
        ν > 0 || throw(ArgumentError("ν must be > 0"))
        d1, d2 = size(R)
        d1 == d2 || throw(DimensionMismatch("R must be square"))
        d1 >= 2 || throw(ArgumentError("R must have dimension at least 2"))
        all(isone, R) && return MTail()

        RF = Matrix{Float64}(R)
        all(isfinite, RF) || throw(ArgumentError("R must contain only finite entries"))
        scale = max(1.0, maximum(abs, RF))
        tol = sqrt(eps(Float64)) * scale
        maximum(abs, RF - transpose(RF)) <= tol || throw(ArgumentError("R must be symmetric"))
        @inbounds for i in 1:d1
            abs(RF[i, i] - 1.0) <= tol || throw(ArgumentError("R must have unit diagonal"))
            RF[i, i] = 1.0
        end
        RF = Matrix(LinearAlgebra.Symmetric((RF + transpose(RF)) / 2))
        try
            LinearAlgebra.cholesky(LinearAlgebra.Symmetric(RF); check=true)
        catch
            throw(ArgumentError("R must be strictly positive definite"))
        end

        νf = float(ν)
        return new{typeof(νf),typeof(RF)}(νf, RF)
    end
end
const tEVCopula{d,T} = ExtremeValueCopula{d, tEVTail{T}}
Distributions.params(tail::tEVTail{<:Any,<:Real}) = (ν = tail.ν, ρ = tail.parameter)
Distributions.params(tail::tEVTail{<:Any,<:AbstractMatrix}) = (ν = tail.ν, R = tail.parameter)
_is_valid_in_dim(tail::tEVTail{<:Any,<:Real}, d::Int) =
    d >= 2 && tail.parameter > -inv(d - 1)
_is_valid_in_dim(tail::tEVTail{<:Any,<:AbstractMatrix}, d::Int) =
    d == size(tail.parameter, 1)
_unbound_params(::Type{<:tEVTail{<:Any,<:Real}}, d, θ) =
    [log(θ.ν), atanh(clamp(θ.ρ, -0.999999, 0.999999))]
_rebound_params(::Type{<:tEVTail{<:Any,<:Real}}, d, α) =
    (; ν = exp(α[1]), ρ = tanh(α[2]))
_example(::Type{<:ExtremeValueCopula{D,<:tEVTail} where D}, d) =
    tEVCopula{d}(2.0, 0.5)
_available_fitting_methods(
    ::Type{<:ExtremeValueCopula{D,<:tEVTail{<:Any,<:AbstractMatrix}} where D},
    d,
) = ()

_tev_rho(tail::tEVTail{<:Any,<:Real}) = tail.parameter
_tev_rho(tail::tEVTail{<:Any,<:AbstractMatrix}) = tail.parameter[1, 2]
function _tev_correlation(tail::tEVTail{<:Any,<:Real}, d::Int)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    ρ = tail.parameter
    lower = -inv(d - 1)
    ρ > lower || throw(ArgumentError("equicorrelation ρ must satisfy ρ > -1/(d-1) in dimension d=$d",))
    ρ < 1 || throw(ArgumentError("the non-degenerate equicorrelation representation requires ρ < 1",))

    R = fill(Float64(ρ), d, d)
    @inbounds for i in 1:d
        R[i, i] = 1.0
    end
    return R
end
_tev_correlation(tail::tEVTail{<:Any,<:AbstractMatrix}, ::Int) = tail.parameter

function _tev_mvtcdf(df::Real, μ, Σ::AbstractMatrix, upper; rtol::Real=2e-6,)
    q = length(upper)
    q == 0 && return 1.0

    dff = Float64(df)
    μf = Float64.(μ)
    upperf = Float64.(upper)
    Σf = Matrix{Float64}(LinearAlgebra.Symmetric(Matrix{Float64}(Σ)))

    if q == 1
        σ = sqrt(Σf[1, 1])
        return Distributions.cdf(Distributions.TDist(dff), (upperf[1] - μf[1]) / σ,)
    end

    δ = upperf .- μf
    χ = Distributions.Chisq(dff)
    lo = eps(Float64)
    hi = 1.0 - eps(Float64)

    integrand(p) = begin
        pp = clamp(Float64(p), lo, hi)
        w = Distributions.quantile(χ, pp)
        b = sqrt(w / dff) .* δ
        MvNormalCDF.mvnormcdf(Σf, fill(-Inf, q), b; rng=Random.Xoshiro(0),)[1]
    end

    val = QuadGK.quadgk(integrand, 0.0, 1.0; rtol=rtol)[1]
    return clamp(val, 0.0, 1.0)
end

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

        p = _tev_mvtcdf(νf + 1.0, r, Σcond, upper)
        total += y[j] * p
    end

    return scale * total
end

function ℓ(tail::tEVTail{<:Any,<:Real}, x)
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

function _tev_block_logintensity(ν::Real, R::AbstractMatrix, z, B::Tuple{Vararg{Int}},)
    b = length(B)
    b > 0 || throw(ArgumentError("the differentiation block must be nonempty"))

    νf = Float64(ν)
    Bv = collect(B)
    C = [i for i in eachindex(z) if i ∉ B]
    zB = Float64.(z[Bv])

    all(zi -> zi > 0, zB) || return -Inf

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

    isempty(C) && return logλB

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

    p = _tev_mvtcdf(νf + b, μ, Σcond, upper)
    return iszero(p) ? -Inf : logλB + log(p)
end

function _tev_ellpartial_signlog(ν::Real, R::AbstractMatrix, x, I::Tuple{Vararg{Int}},)
    isempty(I) && return 1, log(_tev_stdf(ν, R, x))

    all(xi -> xi >= 0, x) || return 0, -Inf
    all(i -> x[i] > 0, I) || return 0, -Inf

    z = [
        iszero(xi) ? Inf : inv(Float64(xi))
        for xi in x
    ]

    logq = _tev_block_logintensity(ν, R, z, I)
    isfinite(logq) || return 0, -Inf

    logjac = 2 * sum(log(Float64(x[i])) for i in I)
    logabs = logq - logjac
    return isodd(length(I)) ? 1 : -1, logabs
end

function _ellpartial_signlog(tail::tEVTail, x, I::Tuple{Vararg{Int}})
    d = length(x)
    R = _tev_correlation(tail, d)
    return _tev_ellpartial_signlog(tail.ν, R, x, Tuple(I))
end


function _tev_spectral_cache(R::AbstractMatrix)
    d = size(R, 1)
    size(R, 2) == d || throw(DimensionMismatch("R must be square"))

    return ntuple(d) do m
        J = [i for i in 1:d if i != m]
        r = Vector{Float64}(R[J, m])
        Σ = Matrix{Float64}(R[J, J]) - r * transpose(r)
        F = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(Σ))
        (; J, r, F)
    end
end

function _tev_log_normalized_spectral!(rng::Distributions.AbstractRNG, logq::AbstractVector{Float64}, ν::Real, R::AbstractMatrix, cache,)
    d = length(logq)
    m = Random.rand(rng, 1:d)
    entry = cache[m]

    # Size-biasing the Gaussian spectral vector by (W_m^+)^ν gives
    # W_m^2 ~ χ²_{ν+1}, with the positive square root.
    wm = sqrt(Random.rand(rng, Distributions.Chisq(Float64(ν) + 1.0)))

    fill!(logq, -Inf)
    logq[m] = Float64(ν) * log(wm)

    q = length(entry.J)
    if q > 0
        ξ = Random.randn(rng, q)
        wJ = entry.r .* wm .+ entry.F.L * ξ

        @inbounds for a in 1:q
            wi = wJ[a]
            if wi > 0
                logq[entry.J[a]] = Float64(ν) * log(wi)
            end
        end
    end

    logsum = LogExpFunctions.logsumexp(logq)
    @inbounds for i in eachindex(logq)
        logq[i] -= logsum
    end
    return logq
end

function _tev_rand_multivariate!(rng::Distributions.AbstractRNG, ν::Real, R::AbstractMatrix, X::AbstractMatrix{T},) where {T<:Real}
    d, n = size(X)

    cache = _tev_spectral_cache(R)
    logq = Vector{Float64}(undef, d)
    logz = Vector{Float64}(undef, d)

    @inbounds for col in 1:n
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

            _tev_log_normalized_spectral!(rng, logq, ν, R, cache)

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

ℓ(tail::tEVTail{<:Any,<:AbstractMatrix}, x) =
    _tev_stdf(tail.ν, tail.parameter, x)

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:tEVTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    return _tev_rand_multivariate!(rng, C.tail.ν, _tev_correlation(C.tail, d), X,)
end

function A(tail::tEVTail, t::Real)
    ρ, ν = _tev_rho(tail), tail.ν
    C = sqrt((1 + ν) / (1 - ρ^2))
    α = 1 / ν

    tt = _safett(t)
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
