# =========================== src/Fitting/Stats.jl ===========================
# Conventions:
# - Rows = variables (p), columns = samples (n)
# - This file exposes the StatsBase/Distributions interface for CopulaModel

# ---------- 1) Utils ----------
@inline _as_pxn(C::Copulas.Copula, U::AbstractMatrix) = (size(U,1) == length(C)) ? U : permutedims(U)
@inline _as_pxn(p::Integer,         U::AbstractMatrix) = (size(U,1) == p)        ? U : permutedims(U)

# ---------- 2) Log-likelihood (Distributions) ----------
function Distributions.loglikelihood(C::Copulas.Copula, U::AbstractMatrix{<:Real})
    Up = _as_pxn(C, U)
    total = 0.0
    @inbounds @views @simd for j in axes(Up, 2)
        total += Distributions.logpdf(C, Up[:, j])
    end
    return total
end

Distributions.loglikelihood(C::Copulas.Copula, u::AbstractVector{<:Real}) = Distributions.logpdf(C, u)

Distributions.loglikelihood(M::CopulaModel) = M.ll
Distributions.loglikelihood(M::CopulaModel, U::AbstractMatrix) = Distributions.loglikelihood(M.result, U)

# ---------- 3) Model stats ----------
StatsBase.nobs(M::CopulaModel)    = M.n
StatsBase.isfitted(::CopulaModel) = true
StatsBase.deviance(M::CopulaModel) = -2 * Distributions.loglikelihood(M)

# ---------- 4) Degrees of freedom ----------
StatsBase.dof(M::CopulaModel) = StatsBase.dof(M.result)

StatsBase.dof(C::Copulas.Copula) = try
    length(Distributions.params(C))
catch
    error("Define `Distributions.params(::$(typeof(C)))` o especializa `StatsBase.dof`.")
end

# Parametric special cases
StatsBase.dof(C::Copulas.GaussianCopula) = (p = length(C); p*(p-1) ÷ 2)
StatsBase.dof(C::Copulas.TCopula)        = (p = length(C); p*(p-1) ÷ 2 + 1)

# Nonparametric copulas
StatsBase.dof(::Copulas.EmpiricalCopula)    = 0
StatsBase.dof(::Copulas.BernsteinCopula)    = 0
StatsBase.dof(::Copulas.BetaCopula)         = 0
StatsBase.dof(::Copulas.CheckerboardCopula) = 0
#StatsBase.dof(::Copulas.EmpiricalEVCopula)  = 0

# ---------- 5) Parameters ----------
StatsBase.coef(M::CopulaModel) = Distributions.params(M.result)

function StatsBase.coefnames(M::CopulaModel)
    C = M.result

    # Acceso seguro a Copulas.paramnames si existe y tiene método
    local f = nothing
    if isdefined(Copulas, :paramnames)
        f = getfield(Copulas, :paramnames)
        if !(f isa Function && hasmethod(f, Tuple{typeof(C)}))
            f = nothing
        end
    end

    if f !== nothing
        return collect(string.(f(C)))
    else
        k = length(StatsBase.coef(M))
        return k == 1 ? ["θ"] : ["θ$(j)" for j in 1:k]
    end
end

# ---------- 6) Inference ----------
StatsBase.vcov(M::CopulaModel) = M.vcov

function StatsBase.stderror(M::CopulaModel)
    V = StatsBase.vcov(M)
    V === nothing && throw(ArgumentError("stderror: vcov(M) == nothing."))
    return sqrt.(LinearAlgebra.diag(V))
end

function StatsBase.confint(M::CopulaModel; level::Real=0.95)
    V = StatsBase.vcov(M)
    V === nothing && throw(ArgumentError("confint: vcov(M) == nothing."))
    z = Distributions.quantile(Distributions.Normal(), 1 - (1 - level)/2)
    θ = StatsBase.coef(M)
    se = sqrt.(LinearAlgebra.diag(V))
    return θ .- z .* se, θ .+ z .* se
end

# ---------- 7) Information criteria ----------
StatsBase.aic(M::CopulaModel) = 2*StatsBase.dof(M) - 2*Distributions.loglikelihood(M)
StatsBase.bic(M::CopulaModel) = StatsBase.dof(M)*log(StatsBase.nobs(M)) - 2*Distributions.loglikelihood(M)

function aicc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    corr = (n > k + 1) ? (2k*(k+1)) / (n - k - 1) : Inf
    return StatsBase.aic(M) + corr
end

function hqc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    return -2*Distributions.loglikelihood(M) + 2k*log(log(max(n, 3)))
end

# ---------- 8) Null likelihood ----------
function StatsBase.nullloglikelihood(M::CopulaModel)
    if hasproperty(M.method_details, :null_ll)
        return getfield(M.method_details, :null_ll)
    else
        throw(ArgumentError("nullloglikelihood no disponible en method_details."))
    end
end

StatsBase.nulldeviance(M::CopulaModel) = -2 * StatsBase.nullloglikelihood(M)