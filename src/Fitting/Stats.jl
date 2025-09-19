# =========================== src/Fitting/Stats.jl ===========================
# Conventions:
# - Rows = variables (p), columns = samples (n)
# - This file exposes the StatsBase/Distributions interface for CopulaModel

# ---------- 1) Utils ---------- maybe we can move to utils.jl
@inline _as_pxn(C::Copulas.Copula, U::AbstractMatrix) = (size(U,1) == length(C)) ? U : permutedims(U)

@inline _as_pxn(p::Integer, U::AbstractMatrix) = (size(U,1) == p) ? U : permutedims(U)

# ---------- 2) Log-verosimilitud (Distributions) ----------
function Distributions.loglikelihood(C::Copulas.Copula, U::AbstractMatrix{<:Real})
    Up = _as_pxn(C, U)
    total = 0.0
    @inbounds @views @simd for j in axes(Up, 2)
        total += Distributions.logpdf(C, Up[:, j])
    end
    return total
end

Distributions.loglikelihood(C::Copulas.Copula, u::AbstractVector{<:Real}) =
    Distributions.logpdf(C, u)

Distributions.loglikelihood(M::CopulaModel) = M.ll
Distributions.loglikelihood(M::CopulaModel, U::AbstractMatrix) = Distributions.loglikelihood(copula(M), U)

# ---------- 3) model stats ----------
StatsBase.nobs(M::CopulaModel) = M.n
StatsBase.isfitted(::CopulaModel) = true
StatsBase.deviance(M::CopulaModel) = -2 * Distributions.loglikelihood(M)

# ---------- 4) degree freedom ----------
StatsBase.dof(M::CopulaModel) = StatsBase.dof(copula(M))

StatsBase.dof(C::Copulas.Copula) = try
    length(Distributions.params(C))
catch
    error("Define `Distributions.params(::$(typeof(C)))` o especializa `StatsBase.dof`.")
end


# non parametric copulas
StatsBase.dof(::Copulas.EmpiricalCopula)    = 0
StatsBase.dof(::Copulas.BernsteinCopula)    = 0
StatsBase.dof(::Copulas.BetaCopula)         = 0
StatsBase.dof(::Copulas.CheckerboardCopula) = 0
#StatsBase.dof(::Copulas.EmpiricalEVTail) = 0 


# ---------- 5) parameters ----------
StatsBase.coef(M::CopulaModel) = Distributions.params(copula(M))

function StatsBase.coefnames(M::CopulaModel)
    C = copula(M)

# Avoid UndefVarError if there are no Copulas.paramnames:
    local f = nothing
    if isdefined(Copulas, :paramnames)
        f = getfield(Copulas, :paramnames)
# Ensure that it is callable and that there is a method for this type:
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


# ---------- 6) Infererence ----------
StatsBase.vcov(M::CopulaModel) = M.vcov

# SE / IC need vcov
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

# ---------- 7) Criterios de información ----------
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

StatsBase.nullloglikelihood(M::CopulaModel) = get(M.method_details, :null_ll) do
        throw(ArgumentError("nullloglikelihood no disponible en method_details."))
end

StatsBase.nulldeviance(M::CopulaModel) = -2 * StatsBase.nullloglikelihood(M)

