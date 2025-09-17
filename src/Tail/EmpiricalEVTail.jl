"""
    EmpiricalEVTail

Fields:
  - `tgrid::Vector{Float64}` — evaluation grid in (0,1)
  - `Ahat::Vector{Float64}`  — estimated Pickands function values on `tgrid`
  - `slope::Vector{Float64}` — per-segment slopes for linear interpolation

Constructor

  EmpiricalEVTail(u; estimator=:ols, grid=401, eps=1e-3, pseudos_values=true)
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
struct EmpiricalEVTail <: Tail2
    tgrid::Vector{Float64}
    Ahat::Vector{Float64}
    slope::Vector{Float64}
end

"""
    EmpiricalEVTail(u; kwargs...)

Construct the empirical Pickands tail from data (2×N or N×2).
"""
function EmpiricalEVTail(u::AbstractMatrix; estimator::Symbol=:ols, grid::Int=401, eps::Real=1e-3, pseudos_values::Bool=true)
    # Ensure 2×n orientation
    Up = _as_pxn(2, u)
    if pseudos_values
        @assert all(0 .<= Up .<= 1)
    else
        Up = pseudos(Up)
    end

    tgrid = collect(range(eps, 1 - eps; length=grid))

    Â = estimator === :ols      ? empirical_pickands_ols(tgrid, Up; pseudos_values=true) :
         estimator === :cfg      ? empirical_pickands_cfg(tgrid, Up; pseudos_values=true) :
         estimator === :pickands ? empirical_pickands(tgrid, Up; pseudos_values=true) :
         throw(ArgumentError("estimator ∈ {:ols,:cfg,:pickands}"))
    Â, slope = _convexify_pickands!(Â, tgrid)
    return EmpiricalEVTail(tgrid, Â, slope)
end

# Convenience: construct the EV copula with the empirical tail
EmpiricalEVCopula(u; kwargs...) = ExtremeValueCopula(2, EmpiricalEVTail(u; kwargs...))

_EULER_GAMMA = Base.MathConstants.eulergamma
Base.eltype(::EmpiricalEVTail) = Float64
Distributions.params(t::EmpiricalEVTail) = (tgrid = t.tgrid, Ahat = t.Ahat, slope = t.slope)
function Base.summary(io::IO, t::EmpiricalEVTail)
    print(io, "EmpiricalEVTail($(length(t.tgrid)) knots)")
end

@inline function _find_segment(tgrid::Vector{Float64}, t::Real)
    # Assume 0 < t < 1; endpoints handled separately
    i = searchsortedlast(tgrid, t)
    if i <= 0
        return 0
    elseif i >= length(tgrid)
        return length(tgrid)    # right endpoint marker
    else
        return i
    end
end

@inline _aslike(t, x::Real) = convert(typeof(t + t - t), x)

# A(t)
function A(tail::EmpiricalEVTail, t::Real)
    tt = _safett(t)
    tg, Ah = tail.tgrid, tail.Ahat

    if tt <= 0.0 || tt >= 1.0
        return _aslike(t, 1.0)  # A(0)=A(1)=1
    end

    i = _find_segment(tg, tt)
    if i == 0
        return _aslike(t, Ah[1])
    elseif i >= length(tg)
        return _aslike(t, Ah[end])
    else
        tL = tg[i]; tR = tg[i+1]
        w  = (tt - tL) / (tR - tL)
        return _aslike(t, (1 - w) * Ah[i] + w * Ah[i+1])
    end
end

# dA(t)
function dA(tail::EmpiricalEVTail, t::Real)
    tt = _safett(t)
    tg, sl = tail.tgrid, tail.slope
    if tt <= 0.0 || tt >= 1.0
        return _aslike(t, 0.0)
    end
    i = _find_segment(tg, tt)
    if i <= 0 || i >= length(tg)
        return _aslike(t, 0.0)
    else
        return _aslike(t, sl[i])
    end
end

# Classical Pickands estimator
function empirical_pickands(tgrid::AbstractVector, U::AbstractMatrix;
                            pseudos_values::Bool=true, endpoint_correction::Bool=true)
    Up = _as_pxn(2, U)
    if pseudos_values
        @assert all(0 .<= Up .<= 1)
    else
        Up = pseudos(Up)
    end

    lu = @views -log.(Up[1, :])
    lv = @views -log.(Up[2, :])

    Â = similar(tgrid, Float64)
    @inbounds for (k, t) in pairs(tgrid)
        tt = _safett(t)
        ξ  = min.(lu ./ (1 - tt), lv ./ tt)
        Â[k] = 1.0 / StatsBase.mean(ξ)
    end
    if endpoint_correction
        Â[begin] = 1.0; Â[end] = 1.0
    end
    return Â
end

# CFG estimator
function empirical_pickands_cfg(tgrid::AbstractVector, U::AbstractMatrix;
                                pseudos_values::Bool=true, endpoint_correction::Bool=true)
    Up = _as_pxn(2, U)
    if pseudos_values
        @assert all(0 .<= Up .<= 1)
    else
        Up = pseudos(Up)
    end
    lu = @views -log.(Up[1, :])
    lv = @views -log.(Up[2, :])
    Â = similar(tgrid, Float64)
    @inbounds for (k, t) in pairs(tgrid)
        tt = _safett(t)
        ξ  = min.(lu ./ (1 - tt), lv ./ tt)
        Â[k] = exp(-_EULER_GAMMA - StatsBase.mean(log.(ξ)))
    end
    if endpoint_correction
        Â[begin] = 1.0; Â[end] = 1.0
    end
    return Â
end

# OLS (intercept) estimator
function empirical_pickands_ols(tgrid::AbstractVector, U::AbstractMatrix;
                                pseudos_values::Bool=true, endpoint_correction::Bool=true)
    Up = _as_pxn(2, U)
    if pseudos_values
        @assert all(0 .<= Up .<= 1)
    else
        Up = pseudos(Up)
    end
    n  = size(Up, 2)
    lu = @views -log.(Up[1, :])       # -log U
    lv = @views -log.(Up[2, :])       # -log V
    x1 = @views -log.(lu) .- _EULER_GAMMA
    x2 = @views -log.(lv) .- _EULER_GAMMA

    Z = Matrix{Float64}(undef, n, 3)
    @inbounds Z[:,1] .= 1.0; Z[:,2] .= x1; Z[:,3] .= x2
    ZtZ = LinearAlgebra.Symmetric(Z'Z)
    F   = LinearAlgebra.cholesky(ZtZ)  # practical positive-definite factorization
    P   = F \ (Z')                    # (Z'Z)^(-1) Z'

    Â = similar(tgrid, Float64)
    y  = similar(lu)
    @inbounds for (k, t) in pairs(tgrid)
        tt = _safett(t)
        ξt = min.(lu ./ (1 - tt), lv ./ tt)
        @. y = -log(ξt) - _EULER_GAMMA
        β  = P * y
        Â[k] = exp(β[1])             # intercept
    end
    if endpoint_correction
        Â[begin] = 1.0; Â[end] = 1.0
    end
    return Â
end

function _convexify_pickands!(Â::Vector{Float64}, t::Vector{Float64})
    n = length(Â); @assert n == length(t) && n ≥ 2
    @inbounds for i in 1:n
        Â[i] = clamp(Â[i], max(t[i], 1 - t[i]), 1.0)
    end

    Δt = diff(t)
    s  = [(Â[i+1]-Â[i])/Δt[i] for i in 1:n-1]
    W  = copy(Δt)
    C  = ones(Int, n-1)

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

    s_exp = similar(Δt)
    pos = 1
    @inbounds for j in 1:length(s)
        cnt = C[j]
        for _ in 1:cnt
            s_exp[pos] = s[j]
            pos += 1
        end
    end
    @assert pos-1 == length(Δt)

    Â[2:end] = Â[1] .+ cumsum(s_exp .* Δt)
    @inbounds for i in 1:n
        Â[i] = clamp(Â[i], max(t[i], 1 - t[i]), 1.0)
    end
    return Â, s_exp
end


