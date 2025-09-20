########################  src/Fitting/CopulaModel.jl  #########################
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

@inline Distributions.fit(::Type{T}, U::AbstractMatrix; kwargs...) where {T<:Copulas.Copula} =
    Distributions.fit(CopulaModel, T, U; kwargs...).result

# ============= Generic fit that returns CopulaModel =============
function Distributions.fit(::Type{CopulaModel}, ::Type{T}, U::AbstractMatrix;
                           method::Symbol = :mle, summaries::Bool=false, kwargs...) where {T<:Copulas.Copula}
    d, n  = size(U)
    C, meta = _fit(T, U, Val{method}(); kwargs...)
    ll     = Distributions.loglikelihood(C, U)
    ll0    = try
        Distributions.loglikelihood(Copulas.IndependentCopula(d), U)
    catch
        NaN
    end

    md = merge((; d, n, method),
               meta,
               (; null_ll = ll0),
               _extra_pairwise_stats(U, summaries))  # optional...

    return CopulaModel(C, n, ll, method;
        vcov         = get(md, :vcov, nothing),
        converged    = get(md, :converged, true),
        iterations   = get(md, :iterations, 0),
        elapsed_sec  = get(md, :elapsed_sec, NaN),
        method_details = md)
end

# Fallback: if there is no _fit implemented for (T, method)
function _fit(::Type{T}, ::AbstractMatrix, ::Any; kwargs...) where {T<:Copulas.Copula}
    throw(ArgumentError("There is no _fit implemented for $(T) with the requested method."))
end

# ============= Generic empirical route (T = Copulas.Copula, method = :emp) =============
# Supports: :beta, :bernstein, :raw (empirical), :ev_tail (empirical EV tail)
function _fit(::Type{<:Copulas.Copula}, U::AbstractMatrix, ::Val{:emp};
              emp_kind::Symbol = :beta,
              pseudo_values::Bool = true,
              m::Union{Nothing, Int, NTuple{N,Int}} = nothing,
              estimator::Symbol = :ols,   # only EV empirical
              grid::Int = 401, eps::Real = 1e-3,
              kwargs...) where {N}
    d, n = size(U)

    if emp_kind === :beta
        C = BetaCopula(U; kwargs...)
        return C, (; estimator=:emp, emp_kind=:beta, pseudo_values)

    elseif emp_kind === :bernstein
        base = EmpiricalCopula(U; pseudo_values=pseudo_values)
        C    = BernsteinCopula(base; m=m)
        return C, (; estimator=:emp, emp_kind=:bernstein, pseudo_values, m=C.m)

    elseif emp_kind === :raw || emp_kind === :emp
        C = EmpiricalCopula(U; pseudo_values=pseudo_values, kwargs...)
        return C, (; estimator=:emp, emp_kind=:raw, pseudo_values)

    elseif emp_kind === :ev_tail
        C = EmpiricalEVCopula(U; estimator=estimator, grid=grid, eps=eps,
                              pseudo_values=pseudo_values)
        return C, (; estimator=:emp, emp_kind=:ev_tail, pseudo_values, grid, eps)

    else
        throw(ArgumentError("emp_kind ∈ (:beta, :bernstein, :raw, :ev_tail)."))
    end
end

# ============= Optional summary utilities (tau, rho, beta) =============

# Optional summaries (enabled with summaries=true)
function _extra_pairwise_stats(U::AbstractMatrix, summaries::Bool)
    summaries || return (;)
    τm, τs, τmin, τmax = _uppertriangle_stats(StatsBase.corkendall(U'))
    ρm, ρs, ρmin, ρmax = _uppertriangle_stats(StatsBase.corspearman(U'))
    βhat = blomqvist_beta(U)
    return (; tau_mean=τm, tau_sd=τs, tau_min=τmin, tau_max=τmax,
             rho_mean=ρm, rho_sd=ρs, rho_min=ρmin, rho_max=ρmax,
             beta_hat=βhat)
end

# ============= StatsBase / Distributions =============
Distributions.loglikelihood(C::Copulas.Copula, U::AbstractMatrix{<:Real}) =
    sum(Base.Fix1(Distributions.logpdf, C), eachcol(U))

Distributions.loglikelihood(C::Copulas.Copula, u::AbstractVector{<:Real}) =
    Distributions.logpdf(C, u)

Distributions.loglikelihood(M::CopulaModel) = M.ll
Distributions.loglikelihood(M::CopulaModel, U::AbstractMatrix) =
    Distributions.loglikelihood(M.result, U)

StatsBase.nobs(M::CopulaModel)     = M.n
StatsBase.isfitted(::CopulaModel)  = true
StatsBase.deviance(M::CopulaModel) = -2 * Distributions.loglikelihood(M)

StatsBase.dof(M::CopulaModel) = StatsBase.dof(M.result)
StatsBase.dof(C::Copulas.Copula) = try
    length(Distributions.params(C))
catch
    error("Define `Distributions.params(::$(typeof(C)))` or specialize `StatsBase.dof`.")
end
StatsBase.dof(::Copulas.EmpiricalCopula)    = 0
StatsBase.dof(::Copulas.BernsteinCopula)    = 0
StatsBase.dof(::Copulas.BetaCopula)         = 0
StatsBase.dof(::Copulas.CheckerboardCopula) = 0
StatsBase.dof(C::Copulas.GaussianCopula)    = (p = length(C); p*(p-1) ÷ 2)
StatsBase.dof(C::Copulas.TCopula)           = (p = length(C); p*(p-1) ÷ 2 + 1)

# coefficients/names (if SklarDist, only report the copula)
result(M::CopulaModel)      = M.result
is_sklar(M::CopulaModel)    = result(M) isa SklarDist
copula_of(M::CopulaModel)   = is_sklar(M) ? result(M).C : result(M)
marginals_of(M::CopulaModel)= is_sklar(M) ? result(M).m : nothing

StatsBase.coef(M::CopulaModel) = Distributions.params(copula_of(M))
function StatsBase.coefnames(M::CopulaModel)
    C = copula_of(M)
    if isdefined(Copulas, :paramnames) && hasmethod(Copulas.paramnames, Tuple{typeof(C)})
        return collect(string.(Copulas.paramnames(C)))
    else
        k = length(StatsBase.coef(M))
        return k == 1 ? ["θ"] : ["θ$(j)" for j in 1:k]
    end
end
#(optional vcov) and vcov its very important... for inference 
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
        throw(ArgumentError("nullloglikelihood not available in method_details."))
    end
end
StatsBase.nulldeviance(M::CopulaModel) = -2 * StatsBase.nullloglikelihood(M)

# =============(show) =============
function _print_coeftable(io::IO, M::CopulaModel; level::Real=0.95)
    θ  = StatsBase.coef(M)
    nm = StatsBase.coefnames(M)
    V  = StatsBase.vcov(M)

    if V === nothing || isempty(θ)
        println(io, "────────────────────────────────────────")
        Printf.@printf(io, "%-14s %12s\n", "Parameter", "Estimate")
        println(io, "────────────────────────────────────────")
        @inbounds for (j, name) in pairs(nm)
            Printf.@printf(io, "%-14s %12.6g\n", String(name), θ[j])
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
    Printf.@printf(io, "%-14s %12s %12s %9s %10s %12s %12s\n",
            "Parameter","Estimate","Std.Err","z-value","Pr(>|z|)","$lvl% Lo","$lvl% Hi")
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
    @inbounds for j in eachindex(θ)
        Printf.@printf(io, "%-14s %12.6g %12.6g %9.3f %10.3g %12.6g %12.6g\n",
                String(nm[j]), θ[j], se[j], z[j], p[j], lo[j], hi[j])
    end
    println(io, "────────────────────────────────────────────────────────────────────────────────────────")
end

# empirical summary (when dof==0 or method=:emp)
function _print_empirical_summary(io::IO, M::CopulaModel)
    md   = M.method_details
    kind = get(md, :emp_kind, :raw)
    d    = get(md, :d, missing)
    n    = get(md, :n, missing)
    pv   = get(md, :pseudo_values, missing)

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
        Printf.@printf(io, "%-10s %12s\n", "Stat", "Value")
        println(io, "────────────────────────────")
        if has_tau;  Printf.@printf(io, "%-10s %12.3f\n", "tau",  md[:tau_mean]); end
        if has_rho;  Printf.@printf(io, "%-10s %12.3f\n", "rho",  md[:rho_mean]); end
        if has_beta; Printf.@printf(io, "%-10s %12.3f\n", "beta", md[:beta_hat]); end
        println(io, "────────────────────────────")
    else
        println(io, "───────────────────────────────────────────────────────")
        Printf.@printf(io, "%-10s %10s %10s %10s %10s\n", "Stat","Mean","SD","Min","Max")
        println(io, "───────────────────────────────────────────────────────")
        if has_tau
            Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "tau", md[:tau_mean], md[:tau_sd], md[:tau_min], md[:tau_max])
        end
        if has_rho
            Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "rho", md[:rho_mean], md[:rho_sd], md[:rho_min], md[:rho_max])
        end
        if has_beta
            Printf.@printf(io, "%-10s %10.3f %10s %10s %10s\n", "beta", md[:beta_hat], "—", "—", "—")
        end
        println(io, "───────────────────────────────────────────────────────")
    end
end

# ====== For Sklar this is optinal... you can prove it
function _dist_paramnames(d)
    T = typeof(d)
    if     T <: Distributions.Gamma;       return ("α","θ")
    elseif T <: Distributions.Beta;        return ("α","β")
    elseif T <: Distributions.LogNormal;   return ("μ","σ")
    elseif T <: Distributions.Normal;      return ("μ","σ")
    elseif T <: Distributions.Exponential; return ("θ",)
    elseif T <: Distributions.Weibull;     return ("k","λ")
    elseif T <: Distributions.Pareto;      return ("α","θ")
    else
        k = length(Distributions.params(d))
        return ntuple(i->"θ$(i)", k)
    end
end

# print section [ Copula ] for SklarDist or copula alone
function _print_copula_section(io::IO, M::CopulaModel; level=0.95)
    C = copula_of(M)
    θ = StatsBase.coef(M)
    nm = StatsBase.coefnames(M)
    V  = StatsBase.vcov(M)
    lvl = Int(round(100*level))

    println(io, "──────────────────────────────────────────────────────────")
    println(io, "[ Copula ]")
    println(io, "──────────────────────────────────────────────────────────")
    fam = _copula_family_label(C)
    Printf.@printf(io, "%-16s %-9s %10s %10s %12s\n",
                   "Family","Param","Estimate","Std.Err","$lvl% CI")

    if V === nothing || isempty(θ)
        @inbounds for j in eachindex(θ)
            Printf.@printf(io, "%-16s %-9s %10.3g %10s %12s\n",
                           fam, String(nm[j]), θ[j], "—", "—")
        end
    else
        se = sqrt.(LinearAlgebra.diag(V))
        lo, hi = StatsBase.confint(M; level=level)
        @inbounds for j in eachindex(θ)
            Printf.@printf(io, "%-16s %-9s %10.3g %10.3g [%0.3g, %0.3g]\n",
                           fam, String(nm[j]), θ[j], se[j], lo[j], hi[j])
        end
    end

    # Fila informativa con τ(θ) si existe
    if isdefined(Copulas, :τ) && hasmethod(Copulas.τ, Tuple{typeof(C)})
        τth = Copulas.τ(C)
        Printf.@printf(io, "%-16s %-9s %10.3g %10s %12s\n",
                       "Kendall", "τ(θ)", τth, "—", "—")
    end
end

########## Sección [ Marginals ] ##########

function _print_marginals_section(io::IO, M::CopulaModel; level=0.95)
    S = result(M)::SklarDist
    println(io, "──────────────────────────────────────────────────────────")
    println(io, "[ Marginals ]")
    println(io, "──────────────────────────────────────────────────────────")
    Printf.@printf(io, "%-6s %-12s %-7s %10s %10s %12s\n",
                   "Margin","Dist","Param","Estimate","Std.Err","$(Int(round(100*level)))% CI")
    for (i, mi) in enumerate(S.m)
        pname = _margin_name(mi)
        θi    = Distributions.params(mi)
        names = _dist_paramnames(mi)
        @inbounds for j in eachindex(θi)
            lab = (j == 1) ? "#$(i)" : ""
            Printf.@printf(io, "%-6s %-12s %-7s %10.3g %10s %12s\n",
                           lab, pname, names[j], θi[j], "—", "—")
        end
    end
end

########## show (usa cabeceras “bonitas”) ##########

function Base.show(io::IO, M::CopulaModel)
    R = result(M)

    # Cabecera sin parámetros de tipo feos
    if R isa SklarDist
        println(io, "SklarDist{Copula=", _copula_family_label(R.C),
                ", Margins=", _margins_tuple_label(R.m), "} fitted via ", M.method)
    else
        println(io, _copula_family_label(R), " fitted via ", M.method)
    end

    n  = StatsBase.nobs(M)
    ll = Distributions.loglikelihood(M)
    Printf.@printf(io, "Number of observations: %d\n", n)

    ll0 = get(M.method_details, :null_ll, NaN)
    if isfinite(ll0)
        Printf.@printf(io, "Null Loglikelihood:  %10.4f\n", ll0)
    end
    Printf.@printf(io, "Loglikelihood:       %10.4f\n", ll)

    # Para el test LR usa g.l. de la CÓPULA si es SklarDist
    kcop = (R isa SklarDist) ? StatsBase.dof(copula_of(M)) : StatsBase.dof(M)
    if isfinite(ll0) && kcop > 0
        LR = 2*(ll - ll0)
        p  = Distributions.ccdf(Distributions.Chisq(kcop), LR)
        Printf.@printf(io, "LR Test (vs indep. copula): %.2f ~ χ²(%d)  =>  p = %.4g\n", LR, kcop, p)
    end

    aic = StatsBase.aic(M); bic = StatsBase.bic(M)
    Printf.@printf(io, "AIC: %.3f   BIC: %.3f\n", aic, bic)

    if isfinite(M.elapsed_sec) || M.iterations != 0 || M.converged != true
        conv = M.converged ? "true" : "false"
        tsec = isfinite(M.elapsed_sec) ? Printf.@sprintf("%.3fs", M.elapsed_sec) : "NA"
        println(io, "Converged: $(conv)   Iterations: $(M.iterations)   Elapsed: $(tsec)")
    end

    # Ramas: SklarDist → secciones; empírico → resumen; si no, tabla coef.
    if R isa SklarDist
        _print_copula_section(io, M)
        _print_marginals_section(io, M)
    elseif StatsBase.dof(M) == 0 || M.method == :emp
        _print_empirical_summary(io, M)
    else
        _print_coeftable(io, M)
    end
end

######################### IFM: fit for SklarDist ######################### #
# Note: Assumes you already defined the SklarDist{CT,TplMargins} type.
# This fits margins (MLE), builds parametric U or ECDF,
# and fits the copula with your backend (_fit(CT, U, Val{copula_method}())).

function Distributions.fit(::Type{CopulaModel},
                           ::Type{SklarDist{CT,TplMargins}},
                           X::AbstractMatrix;
                           copula_method::Symbol = :mle,
                           u_from::Symbol        = :parametric,   # :parametric o :ecdf
                           summaries::Bool       = false,
                           margins_kwargs::NamedTuple = NamedTuple(),
                           copula_kwargs::NamedTuple  = NamedTuple()) where
                           {CT<:Copulas.Copula, TplMargins<:Tuple}

    d, n = size(X)
    marg_types = TplMargins.parameters
    (length(marg_types) == d) ||
        throw(ArgumentError("SklarDist: #marginals $(length(marg_types)) ≠ d=$d"))

    m = ntuple(i -> Distributions.fit(marg_types[i], @view X[i, :]; margins_kwargs...), d)

# ------------------------------------------------------------------
# U-shaped construction for copula fitting
# - :parametric → PIT with clamp at (0,1)
# - :ecdf → pseudo-observations by ranges
# ------------------------------------------------------------------
    U = if u_from === :parametric
        ϵ = 0.5 / (n + 1)
        Utmp = Matrix{Float64}(undef, d, n)
        @inbounds for i in 1:d, j in 1:n
            uij = Distributions.cdf(m[i], X[i, j])
            Utmp[i, j] = clamp(uij, ϵ, 1 - ϵ)
        end
        Utmp
    elseif u_from === :ecdf
        pseudos(X)
    else
        throw(ArgumentError("u_from ∈ (:parametric, :ecdf)"))
    end
    # Copula fit... with method specific
    C, cmeta = _fit(CT, U, Val{copula_method}(); copula_kwargs...)

    S  = SklarDist(C, m)
    ll = Distributions.loglikelihood(S, X)

    ll0 = 0.0
    @inbounds for j in axes(X, 2)
        for i in 1:d
            ll0 += Distributions.logpdf(m[i], X[i, j])
        end
    end

    # ------------------------------------------------------------------
    # 6) Optional summaries (about U)
    # ------------------------------------------------------------------
    extra = _extra_pairwise_stats(U, summaries)

    meta = merge(
        cmeta,
        (; null_ll = ll0, u_from,
           margins = map(typeof, m),
           has_summaries = summaries, d=d, n=n),
        extra,
    )

    return CopulaModel(S, n, ll, copula_method;
        vcov           = get(cmeta, :vcov, nothing),   # vcov of the copula (if you compute it)
        converged      = get(cmeta, :converged, true),
        iterations     = get(cmeta, :iterations, 0),
        elapsed_sec    = get(cmeta, :elapsed_sec, NaN),
        method_details = meta)
end


# Por si tu Distributions no suma por columnas automáticamente para SklarDist:
Distributions.loglikelihood(S::SklarDist, X::AbstractMatrix{<:Real}) = sum(Base.Fix1(Distributions.logpdf, S), eachcol(X))