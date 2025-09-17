"""
    EmpiricalEVTail

Fields:
  - `tgrid::Vector{Float64}` — evaluation grid in (0,1)
  - `Ahat::Vector{Float64}`  — estimated Pickands function values on `tgrid`
  - `slope::Vector{Float64}` — per-segment slopes for linear interpolation

Constructor

  EmpiricalEVTail(u; estimator=:ols, grid=401, eps=1e-3, pseudo_values=true)
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

Construct the empirical Pickands tail from data (2×N).
"""
function EmpiricalEVTail(u::AbstractMatrix; estimator::Symbol=:ols, grid::Int=401, eps::Real=1e-3, pseudo_values::Bool=true)

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

    if estimator === :cfg
        @inbounds for (k, t) in pairs(tgrid)
            tt = _safett(t)
            ξ  = min.(lu ./ (1 - tt), lv ./ tt)
            Â[k] = exp(-γ - StatsBase.mean(log.(ξ)))
        end
    elseif estimator === :pickands
        @inbounds for (k, t) in pairs(tgrid)
            tt = _safett(t)
            ξ  = min.(lu ./ (1 - tt), lv ./ tt)
            Â[k] = 1.0 / StatsBase.mean(ξ)
        end
    elseif estimator === :ols 
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
        throw(ArgumentError("estimator should be :ols, :cfg or :pickands (got $estimator)"))
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
EmpiricalEVCopula(u; kwargs...) = ExtremeValueCopula(2, EmpiricalEVTail(u; kwargs...))

Base.eltype(::EmpiricalEVTail) = Float64
Distributions.params(t::EmpiricalEVTail) = (tgrid = t.tgrid, Ahat = t.Ahat, slope = t.slope)

function A(tail::EmpiricalEVTail, t)
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

function dA(tail::EmpiricalEVTail, t)
    T = typeof(t)
    tt = _safett(t)
    (tt <= 0 || tt >= 1) && return T(0)

    i = searchsortedlast(tail.tgrid, tt)
    (i <= 0 || i >= length(tail.tgrid)) && return T(0)
    return T(tail.slope[i])
end
