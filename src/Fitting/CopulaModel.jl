# One question: what will we do for SklarDist ? 

struct CopulaModel{C, TM<:Union{Nothing,AbstractMatrix}, TD<:NamedTuple} <: StatsBase.StatisticalModel
    result        :: C
    n             :: Int
    ll            :: Float64
    method        :: Symbol
    vcov          :: TM
    converged     :: Bool
    iterations    :: Int
    elapsed_sec   :: Float64
    method_details:: TD
    function CopulaModel(c::C, n::Integer, ll::Real, method::Symbol;
                         vcov=nothing, converged=true, iterations=0, elapsed_sec=NaN,
                         method_details=NamedTuple()) where {C}
        return new{C, typeof(vcov), typeof(method_details)}(
            c, n, float(ll), method, vcov, converged, iterations, float(elapsed_sec), method_details
        )
    end
end
@inline  Distributions.fit(::Type{T}, U::AbstractMatrix; kwargs...) where {T<:Copulas.Copula} = Distributions.fit(CopulaModel, T, U; kwargs...).result
function Distributions.fit(::Type{CopulaModel}, ::Type{T}, U::AbstractMatrix; method::Symbol=:mle, kwargs...) where {T<:Copula}
    C, meta   = _fit(T, U, Val{method}(); kwargs...)
    return CopulaModel(C, size(U, 2), Distributions.loglikelihood(C, U), method;
        vcov       = get(meta, :vcov, nothing),
        converged  = get(meta, :converged, true),
        iterations = get(meta, :iterations, 0),
        elapsed_sec= get(meta, :elapsed_sec, NaN),
        method_details = (; meta..., _extra_pairwise_stats(U)...))
end
function _extra_pairwise_stats(U)
    null_ll = Distributions.loglikelihood(Copulas.IndependentCopula(d), U)
    tau_mean, tau_sd, tau_min, tau_max = _uppertriangle_stats(Statsbase.corkendall(U'))
    rho_mean, rho_sd, rho_min, rho_max = _uppertriangle_stats(Statsbase.corspearman(U'))
    beta_hat=blomqvist_beta(U)
    return (; null_ll, tau_mean, tau_sd, tau_min, tau_max, rho_mean, rho_sd, rho_min, rho_max, beta_hat)
end

# Default version that throws if there is no match. 
function _fit(::Type{T}, U::Abstractmatrix, unknown_method_type; kwargs...) where T<:Copula{d}
    throw(ArgumentError("Fitting procedure for type $T$ with method $(unknown_method_type) is not implemented. Maybe a typo ?"))
end

########################################################################################################
##########  A few methods (should move to the respective copula files i think.)
########################################################################################################

# Then the methods needed for the 4 empirical models (these should be moved in the file of the corresponding models i think.)
function _fit(::Type{<:ExtremeValueCopula}, U, method=:default; pseudo_values = true, estimator::Symbol = :ols, grid::Int = 401, eps::Real = 1e-3, kwargs...)
    C = EmpiricalEVCopula(U; estimator=estimator, grid=grid, eps=eps, pseudos=pseudo_values, kwargs...) # pass kwargs on, you never know. 
    return C, (; estimator=:default, pseudo_values, grid, eps)
end
function _fit(::Type{<:BetaCopula}, U, method=:default; kwargs...)
    C = BetaCopula(U; kwargs...)
    return C, (; estimator=:default, pseudo_values, )
end
function _fit(::Type{<:BernsteinCopula}, U, method=:default; pseudo_values = true, m = nothing, kwargs...)
    C    = BernsteinCopula(EmpiricalCopula(U; pseudo_values=pseudo_values); m=m, kwargs...)
    return C, (; estimator=:default, pseudo_values, m=C.m)
end
function _fit(::Type{<:EmpiricalCopula}, U, method=:default; pseudo_values = true, kwargs...)
    C = EmpiricalCopula(U; pseudo_values=pseudo_values, kwargs...)
    return C, (; estimator=:default, pseudo_values)
end

########################################################################################################
##########  Show function (probably broken now... sorry !!!)
########################################################################################################

# Your show function I did not touch, its probably full of bugs after what i did, sorry about that...
function Base.show(io::IO, M::CopulaModel)
    println(io, "$(typeof(M.result)) fitted via $(M.method)")

    n  = StatsBase.nobs(M)
    ll = Distributions.loglikelihood(M)
    @printf(io, "Number of observations: %d\n", n)

    ll0 = get(M.method_details, :null_ll, NaN)
    if isfinite(ll0); @printf(io, "Null Loglikelihood: %.4f\n", ll0); end
    @printf(io, "Loglikelihood: %.4f\n", ll)

    k = StatsBase.dof(M)
    if isfinite(ll0) && k > 0
        LR = 2*(ll - ll0)
        χ² = Distributions.Chisq(k)
        p  = Distributions.ccdf(χ², LR)
        @printf(io, "LR Test: %.2f ∼ χ²(%d) ⟹  Pr > χ² = %.4g\n", LR, k, p)
    end

    aic = StatsBase.aic(M); bic = StatsBase.bic(M)
    @printf(io, "AIC: %.3f   BIC: %.3f\n", aic, bic)

    if isfinite(M.elapsed_sec) || M.iterations != 0 || M.converged != true
        conv = M.converged ? "true" : "false"
        it   = M.iterations
        tsec = isfinite(M.elapsed_sec) ? @sprintf("%.3fs", M.elapsed_sec) : "NA"
        println(io, "Converged: $(conv)   Iterations: $(it)   Elapsed: $(tsec)")
    end

    level=0.95
    if StatsBase.dof(M) == 0
        md   = M.method_details
        kind = get(md, :emp_kind, :raw)
        d    = get(md, :d, missing)
        n    = get(md, :n, missing)
        pv   = get(md, :pseudo_values, missing)

        # header
        println(io, "Empirical summary ($kind)")
        hdr = "d=$(d), n=$(n)" * (pv === missing ? "" : ", pseudo_values=$(pv)")
        if kind === :bernstein
            m = get(md, :m, nothing); if m !== nothing; hdr *= ", m=$(m)"; end
        elseif kind === :ev_tail
            est = get(md, :estimator, missing)
            grid= get(md, :grid, missing)
            eps = get(md, :eps,  missing)
            hdr *= ", estimator=$(est), grid=$(grid), eps=$(eps)"
        end
        println(io, hdr)

        has_tau  = all(haskey.(Ref(md), (:tau_mean, :tau_sd, :tau_min, :tau_max)))
        has_rho  = all(haskey.(Ref(md), (:rho_mean, :rho_sd, :rho_min, :rho_max)))
        has_beta = haskey(md, :beta_hat)

        if d === missing || d == 2
            println(io, "────────────────────────────")
            @printf(io, "%-10s %12s\n", "Stat", "Value")
            println(io, "────────────────────────────")
            if has_tau;  @printf(io, "%-10s %12.3f\n", "tau",  md[:tau_mean]); end
            if has_rho;  @printf(io, "%-10s %12.3f\n", "rho",  md[:rho_mean]); end
            if has_beta; @printf(io, "%-10s %12.3f\n", "beta", md[:beta_hat]); end
            println(io, "────────────────────────────")
        else
            # Multivariado: mean/sd/min/max por pares
            println(io, "───────────────────────────────────────────────────────")
            @printf(io, "%-10s %10s %10s %10s %10s\n", "Stat","Mean","SD","Min","Max")
            println(io, "───────────────────────────────────────────────────────")
            if has_tau
                @printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                        "tau", md[:tau_mean], md[:tau_sd], md[:tau_min], md[:tau_max])
            end
            if has_rho
                @printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                        "rho", md[:rho_mean], md[:rho_sd], md[:rho_min], md[:rho_max])
            end
            if has_beta
                @printf(io, "%-10s %10.3f %10s %10s %10s\n", "beta", md[:beta_hat], "—", "—", "—")
            end
            println(io, "───────────────────────────────────────────────────────")
        end
        return nothing
    end

    θ  = StatsBase.coef(M)
    nm = StatsBase.coefnames(M)
    V  = StatsBase.vcov(M)

    if V === nothing || isempty(θ)
        println(io, "────────────────────────────────────────")
        @printf(io, "%-14s %12s\n", "Parameter", "Estimate")
        println(io, "────────────────────────────────────────")
        @inbounds for (j, name) in pairs(nm)
            @printf(io, "%-14s %12.6g\n", String(name), θ[j])
        end
        println(io, "────────────────────────────────────────")
        return
    end

    se = sqrt.(LinearAlgebra.diag(V))
    z  = θ ./ se
    p  = 2 .* Distributions.ccdf(Distributions.Normal(), abs.(z))
    lo, hi = StatsBase.confint(M; level=level)
    lvl = Int(round(100*level))

    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    @printf(io, "%-14s %12s %12s %9s %10s %12s %12s\n",
            "Parameter","Estimate","Std.Err","z-value","Pr(>|z|)","$lvl% Lo","$lvl% Hi")
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    @inbounds for j in eachindex(θ)
        @printf(io, "%-14s %12.6g %12.6g %9.3f %10.3g %12.6g %12.6g\n",
                String(nm[j]), θ[j], se[j], z[j], p[j], lo[j], hi[j])
    end
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    return nothing
end

########################################################################################################
##########  Statistics (there is one issue in coefnames, we should re-think the params thing.)
########################################################################################################

Distributions.loglikelihood(C::Copulas.Copula, U::AbstractMatrix{<:Real}) = sum(Base.Fix1(Distributions.logpdf, C), eachcol(U)) # Not sure this is needed, i think Distributions.jl does it automatically ? 
Distributions.loglikelihood(C::Copulas.Copula, u::AbstractVector{<:Real}) = Distributions.logpdf(C, u) # Not sure this is needed, i think Distributions.jl does it automatically ? 

Distributions.loglikelihood(M::CopulaModel) = M.ll
Distributions.loglikelihood(M::CopulaModel, U::AbstractMatrix) = Distributions.loglikelihood(M.result, U)

StatsBase.nobs(M::CopulaModel)    = M.n
StatsBase.isfitted(::CopulaModel) = true
StatsBase.deviance(M::CopulaModel) = -2 * Distributions.loglikelihood(M) # is this needed ? i think there is a default binding ? 

# These degrees of freedom should go to the copulas file i think. 
StatsBase.dof(M::CopulaModel)               = StatsBase.dof(M.result)
StatsBase.dof(C::Copulas.Copula)            = error("Define `Distributions.params(::$(typeof(C)))` o especializa `StatsBase.dof`.")
StatsBase.dof(C::Copulas.GaussianCopula)    = (p = length(C); p*(p-1) ÷ 2)
StatsBase.dof(C::Copulas.TCopula)           = (p = length(C); p*(p-1) ÷ 2 + 1)
StatsBase.dof(::Copulas.EmpiricalCopula)    = 0
StatsBase.dof(::Copulas.BernsteinCopula)    = 0
StatsBase.dof(::Copulas.BetaCopula)         = 0
StatsBase.dof(::Copulas.CheckerboardCopula) = 0
#StatsBase.dof(::Copulas.EmpiricalEVCopula)  = 0

StatsBase.coef(M::CopulaModel) = Distributions.params(M.result)
function StatsBase.coefnames(M::CopulaModel)


    ## Here you indeed have an issue to get params names... 
    ## Maybe we could change the Distributions.params() dispatch on each copula to return a named tuple instead of a tuple ? that way you'll have the names too. 
    ## and even later, we could ensure that typeof(C)(params(C)) returns a copula equivalent to C (if not exactly C), that would be nice. 



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
function StatsBase.nullloglikelihood(M::CopulaModel)
    if hasproperty(M.method_details, :null_ll)
        return getfield(M.method_details, :null_ll)
    else
        throw(ArgumentError("nullloglikelihood no disponible en method_details."))
    end
end
StatsBase.nulldeviance(M::CopulaModel) = -2 * StatsBase.nullloglikelihood(M)