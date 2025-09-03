# ========================== src/Fitting/FitModel.jl ==========================

struct CopulaModel{C, TM<:Union{Nothing,AbstractMatrix}, TD<:NamedTuple} <: StatsBase.StatisticalModel
    copula        :: C
    n             :: Int
    ll            :: Float64
    method        :: Symbol
    vcov          :: TM            
    converged     :: Bool          
    iterations    :: Int          
    elapsed_sec   :: Float64       
    method_details:: TD         
end

CopulaModel(c::C, n::Integer, ll::Real, method::Symbol;
            vcov=nothing, converged=true, iterations=0, elapsed_sec=NaN,
            method_details=NamedTuple()) where {C} =
    CopulaModel{C, typeof(vcov), typeof(method_details)}(c, n, float(ll), method,
                                                         vcov, converged, iterations,
                                                         float(elapsed_sec), method_details)

# access
copula(M::CopulaModel) = M.copula

# --------------------------- Distributions / StatsBase -----------------------

Distributions.loglikelihood(M::CopulaModel) = M.ll
Distributions.loglikelihood(M::CopulaModel, U::AbstractMatrix) =
    Distributions.loglikelihood(copula(M), U)

StatsBase.nobs(M::CopulaModel) = M.n
StatsBase.dof(M::CopulaModel)  = _copuladof(M.copula)

_copuladof(C::Copulas.Copula) = try
    length(Distributions.params(C))
catch
    error("Define dof o params para $(typeof(C)).")
end

_copuladof(C::Copulas.GaussianCopula) = (p = length(C); p*(p-1) ÷ 2)
_copuladof(C::Copulas.TCopula)        = (p = length(C); p*(p-1) ÷ 2 + 1)

StatsBase.coef(M::CopulaModel) = Distributions.params(M.copula)

function StatsBase.coefnames(M::CopulaModel)
    if hasmethod(Copulas.paramnames, Tuple{typeof(M.copula)})
        # returns Vector{String} by default; StatsBase accepts AbstractString
        return collect(string.(Copulas.paramnames(M.copula)))
    else
        k = length(StatsBase.coef(M))
        return ["θ$(j)" for j in 1:k]
    end
end

vcov(M::CopulaModel) = M.vcov

function StatsBase.stderror(M::CopulaModel)
    V = M.vcov
    V === nothing && throw(ArgumentError("stderror: vcov(M) is nothing for this model."))
    return sqrt.(diag(V))
end

function StatsBase.confint(M::CopulaModel; level::Real=0.95)
    V = M.vcov
    V === nothing && throw(ArgumentError("confint: vcov(M) is nothing for this model."))
    α = 1 - level
    z = 1.959963984540054  # ≈ N(0,1) 97.5% para 0.95
    if level != 0.95
        try
            z = Distributions.quantile(Distributions.Normal(), 1 - α/2)
        catch
            @warn "Using z≈1.96 because Distributions.Normal is not available."
        end
    end
    θ = StatsBase.coef(M)
    se = sqrt.(diag(V))
    lo = θ .- z .* se
    hi = θ .+ z .* se
    return lo, hi
end

# ----------------------------- Selection BIC - AIC - others... -------------------------------------

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

# ------------------------------- Display -------------------------------------

function Base.show(io::IO, M::CopulaModel)
    aic = StatsBase.aic(M); bic = StatsBase.bic(M)
    print(io,
        "CopulaModel($(typeof(copula(M))) ; method=$(M.method), n=$(M.n), ",
        "ll=$(round(M.ll, digits=4)), dof=$(StatsBase.dof(M)), ",
        "AIC=$(round(aic, digits=3)), BIC=$(round(bic, digits=3))")
    if isfinite(M.elapsed_sec)
        print(io, ", elapsed=$(round(M.elapsed_sec, digits=3))s")
    end
    print(io, ")")
end
