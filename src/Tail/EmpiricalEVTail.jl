"""
    EmpiricalEVTail, EmpiricalEVCopula

Fields:
  - `tgrid::Vector{Float64}` — evaluation grid in (0,1)
  - `Ahat::Vector{Float64}`  — estimated Pickands function values on `tgrid`
  - `slope::Vector{Float64}` — per-segment slopes for linear interpolation

Constructor

  EmpiricalEVTail(u; method=:ols, grid=401, eps=1e-3, pseudo_values=true)
  ExtremeValueCopula(2, EmpiricalEVTail(u; ...))

The empirical extreme-value (EV) copula (bivariate) is defined from pseudo-observations
`u = (U₁, U₂)` and a nonparametric estimator of the Pickands dependence function. Supported
estimators are:

* `:pickands` — classical Pickands estimator
* `:cfg`      — Capéraà–Fougères–Genest (CFG) estimator
* `:ols`      — OLS-intercept estimator

For stability, the estimated function is always projected onto the class of valid Pickands
functions (convex, bounded between `max(t,1-t)` and `1`, with endpoints fixed at `1`).

Its Pickands function is

    Â(t),  t ∈ (0,1),

evaluated via piecewise linear interpolation on the grid `tgrid`.

References
* [caperaa1997nonparametric] Capéraà, Fougères, Genest (1997) Biometrika
* [gudendorf2011nonparametric] Gudendorf, Segers (2011) Journal of Multivariate Analysis
"""
EmpiricalEVTail, EmpiricalEVCopula
struct EmpiricalEVTail <: Tail2
    tgrid::Vector{Float64}
    Ahat::Vector{Float64}
    slope::Vector{Float64}
end

"""
    EmpiricalEVTail(u; kwargs...)

Construct the empirical Pickands tail from data (2×N).
"""
function EmpiricalEVTail(u::AbstractMatrix; method::Symbol=:ols, grid::Int=401, eps::Real=1e-3, pseudo_values::Bool=true)

    @assert grid ≥ 2
    @assert size(u, 1) == 2 "EmpiricalEVTail expects a (2, n) matrix"
    tgrid = collect(range(eps, 1 - eps; length=grid))
    if pseudo_values
        @assert all(0 .<= u .<= 1) "When pseudo_values=true, u must be in [0,1]"
    end
    U = pseudo_values ? u : pseudos(u)
    lu = @views -log.(U[1, :])
    lv = @views -log.(U[2, :])
    Â = similar(tgrid)

    γ = Base.MathConstants.eulergamma

    if method === :cfg
        @inbounds for (k, t) in pairs(tgrid)
            tt = _safett(t)
            ξ  = min.(lu ./ (1 - tt), lv ./ tt)
            Â[k] = exp(-γ - Statistics.mean(log.(ξ)))
        end
    elseif method === :pickands
        @inbounds for (k, t) in pairs(tgrid)
            tt = _safett(t)
            ξ  = min.(lu ./ (1 - tt), lv ./ tt)
            Â[k] = 1.0 / Statistics.mean(ξ)
        end
    elseif method === :ols 
        n  = size(U, 2)
        x1 = @views -log.(lu) .- γ
        x2 = @views -log.(lv) .- γ

        Z = Matrix{Float64}(undef, n, 3)
        @inbounds Z[:,1] .= 1.0; Z[:,2] .= x1; Z[:,3] .= x2
        ZtZ = LinearAlgebra.Symmetric(Z'Z)
        F   = LinearAlgebra.cholesky(ZtZ)  # practical positive-definite factorization
        P   = F \ (Z')                    # (Z'Z)^(-1) Z'

        y  = similar(lu)
        @inbounds for (k, t) in pairs(tgrid)
            tt = _safett(t)
            ξt = min.(lu ./ (1 - tt), lv ./ tt)
            @. y = -log(ξt) - γ
            β  = P * y
            Â[k] = exp(β[1])             # intercept
        end
    else
        throw(ArgumentError("method should be :ols, :cfg or :pickands (got $method)"))
    end

    # endpoint_correction
    Â[begin] = 1.0; Â[end] = 1.0

    @inbounds for i in 1:grid
        Â[i] = clamp(Â[i], max(tgrid[i], 1 - tgrid[i]), 1.0)
    end

    Δt = diff(tgrid)
    L  = length(Δt)
    s  = [(Â[i+1]-Â[i])/Δt[i] for i in 1:L]
    W  = copy(Δt)
    C  = ones(Int, L)

    i = 1
    while i < length(s)
        if s[i] <= s[i+1] + 1e-14
            i += 1
        else
            newW = W[i] + W[i+1]
            newS = (s[i]*W[i] + s[i+1]*W[i+1]) / newW
            s[i] = newS; W[i] = newW; C[i] += C[i+1]
            deleteat!(s, i+1); deleteat!(W, i+1); deleteat!(C, i+1)
            if i > 1; i -= 1; end
        end
    end

    slope = similar(Δt)
    pos = 1
    @inbounds for j in 1:length(s)
        cnt = C[j]
        for _ in 1:cnt
            slope[pos] = s[j]
            pos += 1
        end
    end
    @assert pos-1 == length(Δt)

    Â[2:end] = Â[1] .+ cumsum(slope .* Δt)
    @inbounds for i in eachindex(tgrid)
        Â[i] = clamp(Â[i], max(tgrid[i], 1 - tgrid[i]), 1.0)
    end
    
    return EmpiricalEVTail(tgrid, Â, slope)
end
const EmpiricalEVCopula = ExtremeValueCopula{2, EmpiricalEVTail}
EmpiricalEVCopula(u; kwargs...) = ExtremeValueCopula(2, EmpiricalEVTail(u; kwargs...))

Base.eltype(::EmpiricalEVTail) = Float64
Distributions.params(t::EmpiricalEVTail) = (tgrid = t.tgrid, Ahat = t.Ahat, slope = t.slope) #for API fit we need modify this

function A(tail::EmpiricalEVTail, t::Real)
    T = typeof(t)
    tt = _safett(t)
    (tt <= 0.0 || tt >= 1.0) && return T(1) # A(0)=A(1)=1

    tg, Ah = tail.tgrid, tail.Ahat
    i = searchsortedlast(tg, tt)
    i <= 0 && return T(Ah[1])
    i >= length(tg) && return T(Ah[end])
    w  = (tt - tg[i]) / (tg[i+1] - tg[i])
    return T((1 - w) * Ah[i] + w * Ah[i+1])
end

function dA(tail::EmpiricalEVTail, t::Real)
    T = typeof(t)
    tt = _safett(t)
    (tt <= 0 || tt >= 1) && return T(0)

    i = searchsortedlast(tail.tgrid, tt)
    (i <= 0 || i >= length(tail.tgrid)) && return T(0)
    return T(tail.slope[i])
end

# Fitting plug-in (empírico) para EmpiricalEVCopula
StatsBase.dof(::EmpiricalEVCopula) = 0
_available_fitting_methods(::Type{<:EmpiricalEVCopula}, d) = (:ols, :cfg, :pickands)
"""
    _fit(::Type{<:EmpiricalEVCopula}, U, method::Union{Val{:ols}, Val{:cfg}, Val{:pickands}};
         grid::Int=401, eps::Real=1e-3, pseudo_values::Bool=true, kwargs...) -> (C, meta)

Empirical bivariate extreme value copula fitting via the Pickands function
(`:ols`, `:cfg`, `:pickands`).

# Arguments
- `U::AbstractMatrix`: 2×n matrix. If `pseudo_values=false`, pseudo-observations are applied.
- `method`: estimator of the Pickands function (`:ols`/`:cfg`/`:pickands`).
- `grid`: number of grid points in `t∈(ε,1−ε)`.
- `eps`: extreme trimming for numerical stability.
- `kwargs...`: forwarded to `EmpiricalEVTail/EmpiricalEVCopula`.

# Returns
- `(C, meta)` where `C::EmpiricalEVCopula` and
`meta = (; emp_kind = :ev_tail, pseudo_values, method = :ols|:cfg|:pickands, grid, eps)`.

**Note**: Method with no free parameters (`dof=0`).
"""
function _fit(::Type{<:EmpiricalEVCopula}, U, method::Union{Val{:ols}, Val{:cfg}, Val{:pickands}}; grid::Int=401, eps::Real=1e-3, pseudo_values::Bool=true, kwargs...)
    m = typeof(method).parameters[1]  # :ols | :cfg | :pickands
    C = EmpiricalEVCopula(U; method=m, grid=grid, eps=eps, pseudo_values=pseudo_values, kwargs...)
    return C, (; emp_kind=:ev_tail, pseudo_values, method=m, grid, eps)
end


# ==============================================================================
# Multivariate empirical extreme-value copula
# ==============================================================================

"""
    EmpiricalEVMultivariateTail(u; method=:ols, degree=nothing,
                                pseudo_values=true)

Shape-constrained nonparametric extreme-value tail in arbitrary dimension.

The pilot Pickands estimator is one of `:ols`, `:cfg`, or `:pickands`.  For
`d ≥ 3`, ordinary convexification is not sufficient to characterize a valid
Pickands dependence function.  The pilot is therefore projected by least
squares onto the class generated by a finite spectral measure supported on a
simplex grid.

If the spectral grid has atoms `v₁,…,vₘ` and masses `h₁,…,hₘ`, the projected
Pickands function is

    A(w) = sum(h[k] * maximum(w .* v[:,k]) for k in 1:m),

subject to

    h[k] ≥ 0,              sum(h[k] * v[i,k] for k in 1:m) = 1

for every margin `i`.  The resulting tail is stored as a
`DiscreteSpectralTail`, so validity of the STDF and exact simulation are
guaranteed by construction.

The OLS estimator is the adaptive CFG estimator of Gudendorf and Segers
(2011).  The spectral projection follows Gudendorf and Segers (2012).

`degree` controls the simplex-grid resolution.  If omitted, a moderate
dimension-adaptive degree is selected to keep the convex projection tractable.
"""
struct EmpiricalEVMultivariateTail <: Tail
    d::Int
    method::Symbol
    degree::Int
    spectral::DiscreteSpectralTail{Float64}
    projection_rmse::Float64
end

Base.eltype(::EmpiricalEVMultivariateTail) = Float64
Distributions.params(t::EmpiricalEVMultivariateTail) = (B = t.spectral.B,)
_is_valid_in_dim(t::EmpiricalEVMultivariateTail, d::Int) = t.d == d
ℓ(t::EmpiricalEVMultivariateTail, x) = ℓ(t.spectral, x)
A(t::EmpiricalEVMultivariateTail, w::NTuple{d,<:Real}) where {d} = ℓ(t, w)

function _empirical_ev_default_degree(d::Int; max_atoms::Int=120)
    d >= 2 || throw(ArgumentError("dimension must be at least two"))
    for degree in 12:-1:1
        binomial(degree + d - 1, d - 1) <= max_atoms && return degree
    end
    return 1
end

function _empirical_ev_compositions(total::Int, d::Int)
    out = Vector{Vector{Int}}()
    cur = zeros(Int, d)

    function visit!(j::Int, left::Int)
        if j == d
            cur[j] = left
            push!(out, copy(cur))
            return
        end
        for a in 0:left
            cur[j] = a
            visit!(j + 1, left - a)
        end
    end

    visit!(1, total)
    return out
end

function _empirical_ev_simplex_grid(d::Int, degree::Int)
    degree >= 1 || throw(ArgumentError("spectral degree must be positive"))
    comps = _empirical_ev_compositions(degree, d)
    V = Matrix{Float64}(undef, d, length(comps))

    @inbounds for (k, α) in enumerate(comps), j in 1:d
        V[j, k] = α[j] / degree
    end

    # Include the barycenter explicitly.  It represents complete dependence
    # exactly even when `degree` is not divisible by d.
    bary = fill(inv(float(d)), d)
    has_bary = any(k -> maximum(abs.(@view(V[:, k]) .- bary)) <= 8eps(Float64), axes(V, 2))
    return has_bary ? V : hcat(V, bary)
end

function _empirical_ev_uniforms(u::AbstractMatrix, pseudo_values::Bool)
    d, n = size(u)
    d >= 2 || throw(ArgumentError("Empirical EV estimation requires d ≥ 2"))
    n >= 2 || throw(ArgumentError("Empirical EV estimation requires at least two observations"))

    if pseudo_values
        all(isfinite, u) || throw(ArgumentError("pseudo-observations must be finite"))
        all(0 .<= u .<= 1) || throw(ArgumentError("when pseudo_values=true, observations must lie in [0,1]",))
        U = Matrix{Float64}(u)
    else
        U = Matrix{Float64}(pseudos(u))
    end

    # Ranks produced by `pseudos` are strictly interior.  For user-supplied
    # pseudo-observations, protect logarithms at exact numerical boundaries.
    lo = eps(Float64)
    hi = 1.0 - eps(Float64)
    @inbounds for i in eachindex(U)
        U[i] = clamp(U[i], lo, hi)
    end
    return U
end

function _empirical_ev_xi!(ξ, L, w)
    d, n = size(L)
    @inbounds for obs in 1:n
        best = Inf
        for j in 1:d
            wj = w[j]
            iszero(wj) && continue
            candidate = L[j, obs] / wj
            candidate < best && (best = candidate)
        end
        ξ[obs] = best
    end
    return ξ
end

function _empirical_ev_raw_pickands!(ξ, L, w)
    _empirical_ev_xi!(ξ, L, w)
    return inv(Statistics.mean(ξ))
end

function _empirical_ev_raw_logcfg!(ξ, L, w)
    _empirical_ev_xi!(ξ, L, w)
    γ = Float64(Base.MathConstants.eulergamma)
    return -γ - Statistics.mean(log, ξ)
end

function _empirical_ev_pilot(U::Matrix{Float64}, W::Matrix{Float64}, method::Symbol,)
    d, n = size(U)
    m = size(W, 2)
    L = -log.(U)
    ξ = Vector{Float64}(undef, n)
    pilot = Vector{Float64}(undef, m)

    if method === :pickands
        endpoint = Vector{Float64}(undef, d)
        e = zeros(Float64, d)
        for j in 1:d
            fill!(e, 0.0)
            e[j] = 1.0
            endpoint[j] = _empirical_ev_raw_pickands!(ξ, L, e)
        end

        @inbounds for k in 1:m
            w = @view W[:, k]
            raw = _empirical_ev_raw_pickands!(ξ, L, w)
            invcorr = inv(raw)
            for j in 1:d
                invcorr -= w[j] * (inv(endpoint[j]) - 1.0)
            end
            Ahat = invcorr > 0 ? inv(invcorr) : raw
            pilot[k] = clamp(Ahat, maximum(w), 1.0)
        end

    elseif method === :cfg
        endpoint_log = Vector{Float64}(undef, d)
        e = zeros(Float64, d)
        for j in 1:d
            fill!(e, 0.0)
            e[j] = 1.0
            endpoint_log[j] = _empirical_ev_raw_logcfg!(ξ, L, e)
        end

        @inbounds for k in 1:m
            w = @view W[:, k]
            logA = _empirical_ev_raw_logcfg!(ξ, L, w)
            for j in 1:d
                logA -= w[j] * endpoint_log[j]
            end
            Ahat = exp(clamp(logA, -50.0, 50.0))
            pilot[k] = clamp(Ahat, maximum(w), 1.0)
        end

    elseif method === :ols
        γ = Float64(Base.MathConstants.eulergamma)
        Z = Matrix{Float64}(undef, n, d + 1)
        @views Z[:, 1] .= 1.0
        @inbounds for j in 1:d, obs in 1:n
            Z[obs, j + 1] = -log(L[j, obs]) - γ
        end

        # SVD pseudoinverse is robust also for nearly/completely dependent
        # margins, where the endpoint regressors may be rank deficient.
        P = LinearAlgebra.pinv(Z)
        y = Vector{Float64}(undef, n)

        @inbounds for k in 1:m
            w = @view W[:, k]
            _empirical_ev_xi!(ξ, L, w)
            @. y = -log(ξ) - γ
            logA = LinearAlgebra.dot(@view(P[1, :]), y)
            Ahat = exp(clamp(logA, -50.0, 50.0))
            pilot[k] = clamp(Ahat, maximum(w), 1.0)
        end
    else
        throw(ArgumentError("method should be :ols, :cfg or :pickands (got $method)",))
    end

    return pilot
end

function _empirical_ev_projection_matrix(W::Matrix{Float64}, V::Matrix{Float64})
    d, r = size(W)
    d == size(V, 1) || throw(DimensionMismatch("simplex grids have different dimensions"))
    m = size(V, 2)
    M = Matrix{Float64}(undef, r, m)

    @inbounds for q in 1:r, k in 1:m
        best = 0.0
        for j in 1:d
            val = W[j, q] * V[j, k]
            val > best && (best = val)
        end
        M[q, k] = best
    end
    return M
end

function _empirical_ev_project_spectral(pilot::Vector{Float64}, V::Matrix{Float64}; maxiter::Int=1500,)
    d, m = size(V)
    length(pilot) == m || throw(DimensionMismatch("pilot values and spectral grid must have the same number of points",))

    W = V
    M = _empirical_ev_projection_matrix(W, V)

    # By symmetry of the full simplex lattice (and the optional barycenter),
    # equal masses d/m satisfy the moment constraints and are strictly
    # positive: an excellent interior starting point for IPNewton.
    h0 = fill(d / m, m)
    target = ones(Float64, d)

    function con!(c, h)
        LinearAlgebra.mul!(c, V, h)
        return c
    end
    function jac!(J, h)
        J .= V
        return J
    end
    function con_hess!(H, h, λ)
        fill!(H, 0.0)
        return H
    end

    lower_h = zeros(Float64, m)
    upper_h = fill(Inf, m)
    constraints = Optim.TwiceDifferentiableConstraints(con!, jac!, con_hess!,lower_h, upper_h, target, target,)

    objective(h) = 0.5 * sum(abs2, M * h - pilot)

    result = Optim.optimize(
        objective,
        constraints,
        h0,
        Optim.IPNewton(),
        Optim.Options(
            iterations=maxiter,
            allow_f_increases=true,
            successive_f_tol=2,
        );
        autodiff=ADTypes.AutoForwardDiff(),
    )

    h = Float64.(Optim.minimizer(result))
    all(isfinite, h) || throw(ArgumentError("spectral projection returned non-finite masses",))
    h .= max.(h, 0.0)

    B = V .* reshape(h, 1, :)
    rowsums = vec(sum(B, dims=2))
    all(rowsums .> 0) || throw(ArgumentError("spectral projection produced a degenerate margin",))

    # IPNewton satisfies the equalities to numerical precision.  Renormalizing
    # each row removes only solver roundoff and guarantees exact valid margins
    # for the public DiscreteSpectralTail constructor.
    B ./= reshape(rowsums, :, 1)
    spectral = DiscreteSpectralTail(B)

    fitted = Vector{Float64}(undef, size(W, 2))
    @inbounds for q in eachindex(fitted)
        fitted[q] = ℓ(spectral, @view W[:, q])
    end
    rmse = sqrt(Statistics.mean(abs2, fitted .- pilot))

    return spectral, rmse
end

function EmpiricalEVMultivariateTail(u::AbstractMatrix; method::Symbol=:ols, degree::Union{Nothing,Int}=nothing, pseudo_values::Bool=true, projection_maxiter::Int=1500,)
    d = size(u, 1)
    d >= 2 || throw(ArgumentError("EmpiricalEVMultivariateTail requires at least two dimensions",))
    method in (:ols, :cfg, :pickands) || throw(ArgumentError("method should be :ols, :cfg or :pickands (got $method)",))

    deg = isnothing(degree) ? _empirical_ev_default_degree(d) : degree
    deg >= 1 || throw(ArgumentError("degree must be positive"))

    U = _empirical_ev_uniforms(u, pseudo_values)
    V = _empirical_ev_simplex_grid(d, deg)
    pilot = _empirical_ev_pilot(U, V, method)
    spectral, rmse = _empirical_ev_project_spectral(pilot, V; maxiter=projection_maxiter,)

    return EmpiricalEVMultivariateTail(d, method, deg, spectral, rmse)
end

"""
    EmpiricalEVMultivariateCopula(u; kwargs...)

Construct a shape-valid multivariate empirical extreme-value copula from a
`d × n` sample.  In dimensions `d ≥ 3`, this is the preferred nonparametric EV
constructor.  The historical `EmpiricalEVCopula` remains the backward-compatible
bivariate implementation.
"""
EmpiricalEVMultivariateCopula(u::AbstractMatrix; kwargs...) =ExtremeValueCopula(size(u, 1), EmpiricalEVMultivariateTail(u; kwargs...),)

StatsBase.dof(::ExtremeValueCopula{d,<:EmpiricalEVMultivariateTail}) where {d} = 0
_available_fitting_methods(::Type{<:ExtremeValueCopula{d,<:EmpiricalEVMultivariateTail}}, dim,) where {d} = (:ols, :cfg, :pickands)

_rand_ev_multivariate!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:EmpiricalEVMultivariateTail}, X::AbstractMatrix{T},) where {d,T<:Real} = _discrete_spectral_rand!(rng, C.tail.spectral, X)

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2,<:EmpiricalEVMultivariateTail}, X::AbstractMatrix{T},) where {T<:Real}
    size(X, 1) == 2 || throw(DimensionMismatch("output must have two rows for a bivariate empirical spectral EV copula",))
    return _discrete_spectral_rand!(rng, C.tail.spectral, X)
end

function Distributions._logpdf(::ExtremeValueCopula{d,<:EmpiricalEVMultivariateTail}, u,) where {d}
    throw(ArgumentError(
        "the shape-constrained multivariate empirical EV copula uses a " *
        "discrete spectral measure and can contain singular components; " *
        "a global Lebesgue log-density is not defined in general",
    ))
end
