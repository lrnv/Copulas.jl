##############################################################################################################################
####### sCopulaModel interface, main fitting functions and StatsBase bindings.
##############################################################################################################################
struct CopulaModel{CT, TM<:Union{Nothing,AbstractMatrix}, TD<:NamedTuple} <: StatsBase.StatisticalModel
    result        :: CT
    n             :: Int
    ll            :: Float64
    method        :: Symbol
    vcov          :: TM
    converged     :: Bool
    iterations    :: Int
    elapsed_sec   :: Float64
    method_details:: TD
    function CopulaModel(c::CT, n::Integer, ll::Real, method::Symbol;
                         vcov=nothing, converged=true, iterations=0, elapsed_sec=NaN,
                         method_details=NamedTuple()) where {C}
        return new{CT, typeof(vcov), typeof(method_details)}(
            c, n, float(ll), method, vcov, converged, iterations, float(elapsed_sec), method_details
        )
    end
end

@inline  Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U; kwargs...) = Distributions.fit(CopulaModel, T, U; summaries=false..., kwargs...).result
function Distributions.fit(::Type{CopulaModel}, ::Type{<:Copula}, U; method = :default, summaries=true, kwargs...)
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
               _extra_pairwise_stats(U, !summaries))  # optional...

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
function Distributions.fit(::Type{CopulaModel},::Type{SklarDist{CT,TplMargins}}, X; copula_method = :default, sklar_method = :parametric,
                           summaries = true, margins_kwargs = NamedTuple(), copula_kwargs = NamedTuple()) where
                           {CT<:Copulas.Copula, TplMargins<:Tuple}

    d, n = size(X)
    marg_types = TplMargins.parameters
    (length(marg_types) == d) || throw(ArgumentError("SklarDist: #marginals $(length(marg_types)) ≠ d=$d"))
    m = ntuple(i -> Distributions.fit(marg_types[i], @view X[i, :]; margins_kwargs...), d)
    # Only one margins_kwargs while people mught want to pass diferent kwargs for diferent marginals... but OK for the moment.

    U = similar(X)
    if sklar_method === :parametric
        for i in 1:d
            U[i,:] .= Distributions.cdf(m[i], X[i,:])
        end
    elseif sklar_method === :ecdf
        U .= pseudos(X)
    else
        throw(ArgumentError("sklar_method ∈ (:parametric, :ecdf)"))
    end
    # Copula fit... with method specific
    C, cmeta = _fit(CT, U, Val{copula_method}(); copula_kwargs...)

    S  = SklarDist(C, m)
    ll = Distributions.loglikelihood(S, X)

    null_ll = 0.0
    @inbounds for j in axes(X, 2)
        for i in 1:d
            null_ll += Distributions.logpdf(m[i], X[i, j])
        end
    end

    return CopulaModel(S, n, ll, copula_method;
        vcov           = get(cmeta, :vcov, nothing),   # vcov of the copula (if you compute it)
        converged      = get(cmeta, :converged, true),
        iterations     = get(cmeta, :iterations, 0),
        elapsed_sec    = get(cmeta, :elapsed_sec, NaN),
        method_details = (; cmeta..., null_ll, sklar_method, margins = map(typeof, m), 
              has_summaries = summaries, d=d, n=n, _extra_pairwise_stats(U, !summaries)...))
end

StatsBase.dof(::Copulas.EmpiricalCopula)    = 0
StatsBase.dof(::Copulas.BernsteinCopula)    = 0
StatsBase.dof(::Copulas.BetaCopula)         = 0
StatsBase.dof(::Copulas.CheckerboardCopula) = 0
StatsBase.dof(C::Copulas.GaussianCopula)    = (p = length(C); p*(p-1) ÷ 2)
StatsBase.dof(C::Copulas.TCopula)           = (p = length(C); p*(p-1) ÷ 2 + 1)

function _extra_pairwise_stats(U::AbstractMatrix, bypass::Bool)
    bypass && return (;)
    τm, τs, τmin, τmax = _uppertriangle_stats(StatsBase.corkendall(U'))
    ρm, ρs, ρmin, ρmax = _uppertriangle_stats(StatsBase.corspearman(U'))
    βhat = blomqvist_beta(U)
    return (; tau_mean=τm, tau_sd=τs, tau_min=τmin, tau_max=τmax,
             rho_mean=ρm, rho_sd=ρs, rho_min=ρmin, rho_max=ρmax,
             beta_hat=βhat)
end
Distributions.loglikelihood(C::Copulas.Copula, U::AbstractMatrix{<:Real}) = sum(Base.Fix1(Distributions.logpdf, C), eachcol(U))
Distributions.loglikelihood(C::Copulas.Copula, u::AbstractVector{<:Real}) = Distributions.logpdf(C, u)
Distributions.loglikelihood(M::CopulaModel) = M.ll
Distributions.loglikelihood(M::CopulaModel, U::AbstractMatrix) = Distributions.loglikelihood(M.result, U)

StatsBase.nobs(M::CopulaModel)     = M.n
StatsBase.isfitted(::CopulaModel)  = true
StatsBase.deviance(M::CopulaModel) = -2 * Distributions.loglikelihood(M)
StatsBase.dof(M::CopulaModel) = StatsBase.dof(M.result)
function StatsBase.dof(C::Copulas.Copula)
    try
        length(Distributions.params(C))
    catch
        error("Define `Distributions.params(::$(typeof(C)))` or specialize `StatsBase.dof`.")
    end
end

copula_of(M::CopulaModel)   = M.result isa SklarDist ? M.result.C : M.result

StatsBase.coef(M::CopulaModel) = Distributions.params(copula_of(M)) # why ? params of the marginals should also be taken into account. 

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



##############################################################################################################################
####### Speficic methods for certain copulas, shoul dbe moved to repective files. 
##############################################################################################################################
_fit(T::Type{<:ExtremeValueCopula}, U, ::Val{:default}; kwargs...) = _fit(T, U, ::Val{:ols}; kwargs...)
function _fit(::Type{<:ExtremeValueCopula}, U, method::Union{Val{:ols}, Val{:cfg}, Val{:pickands}}; pseudo_values = true, estimator::Symbol = :ols, grid::Int = 401, eps::Real = 1e-3, kwargs...)
    m = typeof(method).parameters[1]
    C = EmpiricalEVCopula(U; estimator=estimator, grid=grid, eps=eps, pseudos=pseudo_values, estimator=m, kwargs...) # pass kwargs on, you never know. 
    return C, (; estimator=m, pseudo_values, grid, eps)
end
function _fit(::Type{<:BetaCopula}, U, ::Val{:default}; kwargs...)
    C = BetaCopula(U; kwargs...)
    return C, (; estimator=:segers2017, pseudo_values, )
end
function _fit(::Type{<:BernsteinCopula}, U, ::Val{:default}; pseudo_values = true, m = nothing, kwargs...)
    C = BernsteinCopula(EmpiricalCopula(U; pseudo_values=pseudo_values); m=m, kwargs...)
    return C, (; estimator=:segers2017, pseudo_values, m=C.m)
end
function _fit(::Type{<:EmpiricalCopula}, U, ::Val{:default}; pseudo_values = true, kwargs...)
    C = EmpiricalCopula(U; pseudo_values=pseudo_values, kwargs...)
    return C, (; estimator=:deheuvels, pseudo_values)
end

# helper
params(::Type{T}) where {T<:Copula} = throw("No params() function defined for type T = $T...")
params(CT::Type{<:ArchimedeanCopula}) = params(generatorof(CT))
function _fit(CT::Type{<:ArchimedeanCopula{d, <:UnivariateGenerator} where d}, U, ::Val{:itau})
    d = size(U,1)
    GT   = generatorof(CT)
    θs   = map(v -> τ⁻¹(GT, clamp(v, -1, 1)), _uppertriangle_stats(StatsBase.corkendall(U')))
    θ = clamp(StatsBase.mean(θs), _θ_bounds(GT, d)...)
    return CT(d, θ), (; estimator=:itau, eps)
end
function _fit(CT::Type{<:ArchimedeanCopula{d, <:UnivariateGenerator} where d}, U, ::Val{:irho})
    d = size(U,1)
    GT   = generatorof(CT)
    θs   = map(v -> ρ⁻¹(GT, clamp(v, -1, 1)), _uppertriangle_stats(StatsBase.corspearman(U')))
    θ = clamp(StatsBase.mean(θs), _θ_bounds(GT, d)...)
    return CT(d, θ), (; estimator=:irho, eps)
end
function _fit(CT::Type{<:ArchimedeanCopula{d, <:UnivariateGenerator} where d}, U, ::Val{:ibeta})
    d = size(U,1)
    β̂ = clamp(blomqvist_beta(U), -1, 1)
    GT = generatorof(CT)
    a, b = sort(_θ_bounds(GT, d))
    f(θ) = β(CT(d, θ)) - β̂
    fa, fb = f(a), f(b)
    if sign(fa) == sign(fb) # if no bracket → β̂ out of range → nearest end
        θstar = (abs(fa) ≤ abs(fb)) ? a : b
        return CT(d, θstar), (; estimator=:ibeta, epsβ)
    end
    θ = Roots.find_zero(f, (a, b), Roots.Brent(); xatol=1e-10, rtol=0.0)
    return CT(d, θ), (; estimator=:ibeta, epsβ)
end
function _fit(CT::Type{<:ArchimedeanCopula{d, <:UnivariateGenerator} where d}, U, ::Val{:mle};
              start::Union{Symbol,Real}=:itau, xtol::Real=1e-8)
    d = size(U,1)
    GT = generatorof(CT)
    lo, hi = _θ_bounds(GT, d)
    θ0 = start isa Real ? start : 
         start ∈ (:itau, :irho) ? _fit(CT, U, Val{start}()) : 
         throw("The start parameter you provided is not either a real number, :itau or :irho")
    θ0 = clamp(θ0, lo, hi)
    f(θ) = -Distributions.loglikelihood(CT(d, θ), U)
    t = @elapsed (res = Optim.optimize(f, lo, hi, θ0, Optim.Fminbox(GradientDescent()); abs_tol=xtol))
    θ̂     = Optim.minimizer(res)
    return CT(d, θ̂), (; estimator=:mle, θ̂=θ̂, optimizer=:GradientDescent,
                        xtol=xtol, converged=Optim.converged(res), 
                        iterations=Optim.iterations(res), elapsed_sec=t)
end

############################################################################
# These theta bounds are not really true, you should match them with the 
# max monotony 
# in fact they are bascially "inverses" of the max monotony functions. 
# They should be moved to the geenrator files would be easier. 
_θ_bounds(::Type{<:ClaytonGenerator},      d::Integer) = (-1/(d-1),  Inf)
_θ_bounds(::Type{<:AMHGenerator},          d::Integer) = (-1,  1)
_θ_bounds(::Type{<:GumbelGenerator},        ::Integer) = (1.0,       Inf)
_θ_bounds(::Type{<:JoeGenerator},           ::Integer) = (1.0,       Inf)
_θ_bounds(::Type{<:FrankGenerator},        d::Integer) = d ≥ 3 ? (nextfloat(0.0),  Inf) : (-Inf, Inf) ### This is wrong. 
_θ_bounds(::Type{<:GumbelBarnettGenerator}, ::Integer) = (0.0, 1.0)








##############################################################################################################################
####### show function. PLease do not construct other functions for it. 
##############################################################################################################################

function Base.show(io::IO, M::CopulaModel)
    R = M.result
    # Header: family/margins without helper functions
    if R isa SklarDist
        # Build copula family label
        famC = String(nameof(typeof(R.C)))
        famC = endswith(famC, "Copula") ? famC[1:end-6] : famC
        famC = string(famC, " d=", length(R.C))
        # Margins label
        mnames = map(mi -> String(nameof(typeof(mi))), R.m)
        margins_lbl = "(" * join(mnames, ", ") * ")"
        println(io, "SklarDist{Copula=", famC, ", Margins=", margins_lbl, "} fitted via ", M.method)
    else
        fam = String(nameof(typeof(R)))
        fam = endswith(fam, "Copula") ? fam[1:end-6] : fam
        fam = string(fam, " d=", length(R))
        println(io, fam, " fitted via ", M.method)
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

    # Branches: SklarDist → sections; empirical → summary; else → coefficient table
    if R isa SklarDist
        # [ Copula ] section
        C = copula_of(M)
        θ = StatsBase.coef(M)
        nm = StatsBase.coefnames(M)
        V  = StatsBase.vcov(M)
        lvl = 95
        println(io, "──────────────────────────────────────────────────────────")
        println(io, "[ Copula ]")
        println(io, "──────────────────────────────────────────────────────────")
        fam = String(nameof(typeof(C))); fam = endswith(fam, "Copula") ? fam[1:end-6] : fam; fam = string(fam, " d=", length(C))
        Printf.@printf(io, "%-16s %-9s %10s %10s %12s\n", "Family","Param","Estimate","Std.Err","$lvl% CI")
        if V === nothing || isempty(θ)
            @inbounds for j in eachindex(θ)
                Printf.@printf(io, "%-16s %-9s %10.3g %10s %12s\n", fam, String(nm[j]), θ[j], "—", "—")
            end
        else
            se = sqrt.(LinearAlgebra.diag(V))
            lo, hi = StatsBase.confint(M; level=0.95)
            @inbounds for j in eachindex(θ)
                Printf.@printf(io, "%-16s %-9s %10.3g %10.3g [%0.3g, %0.3g]\n", fam, String(nm[j]), θ[j], se[j], lo[j], hi[j])
            end
        end
        if isdefined(Copulas, :τ) && hasmethod(Copulas.τ, Tuple{typeof(C)})
            τth = Copulas.τ(C)
            Printf.@printf(io, "%-16s %-9s %10.3g %10s %12s\n", "Kendall", "τ(θ)", τth, "—", "—")
        end

        # [ Marginals ] section
        S = R::SklarDist
        println(io, "──────────────────────────────────────────────────────────")
        println(io, "[ Marginals ]")
        println(io, "──────────────────────────────────────────────────────────")
        Printf.@printf(io, "%-6s %-12s %-7s %10s %10s %12s\n", "Margin","Dist","Param","Estimate","Std.Err","$lvl% CI")
        for (i, mi) in enumerate(S.m)
            pname = String(nameof(typeof(mi)))
            θi    = Distributions.params(mi)
            # Inline param name mapping
            T = typeof(mi)
            names = if     T <: Distributions.Gamma;       ("α","θ")
                    elseif T <: Distributions.Beta;        ("α","β")
                    elseif T <: Distributions.LogNormal;   ("μ","σ")
                    elseif T <: Distributions.Normal;      ("μ","σ")
                    elseif T <: Distributions.Exponential; ("θ",)
                    elseif T <: Distributions.Weibull;     ("k","λ")
                    elseif T <: Distributions.Pareto;      ("α","θ")
                    else
                        k = length(θi); ntuple(j->"θ$(j)", k)
                    end
            @inbounds for j in eachindex(θi)
                lab = (j == 1) ? "#$(i)" : ""
                Printf.@printf(io, "%-6s %-12s %-7s %10.3g %10s %12s\n", lab, pname, names[j], θi[j], "—", "—")
            end
        end
    elseif StatsBase.dof(M) == 0 || M.method == :emp
        # Empirical summary
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
                Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n", "tau", md[:tau_mean], md[:tau_sd], md[:tau_min], md[:tau_max])
            end
            if has_rho
                Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n", "rho", md[:rho_mean], md[:rho_sd], md[:rho_min], md[:rho_max])
            end
            if has_beta
                Printf.@printf(io, "%-10s %10.3f %10s %10s %10s\n", "beta", md[:beta_hat], "—", "—", "—")
            end
            println(io, "───────────────────────────────────────────────────────")
        end
    else
        # Coefficient table
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
        else
            se = sqrt.(LinearAlgebra.diag(V))
            z  = θ ./ se
            p  = 2 .* Distributions.ccdf(Distributions.Normal(), abs.(z))
            lo, hi = StatsBase.confint(M; level=0.95)
            println(io, "────────────────────────────────────────────────────────────────────────────────────────")
            Printf.@printf(io, "%-14s %12s %12s %9s %10s %12s %12s\n", "Parameter","Estimate","Std.Err","z-value","Pr(>|z|)","95% Lo","95% Hi")
            println(io, "────────────────────────────────────────────────────────────────────────────────────────")
            @inbounds for j in eachindex(θ)
                Printf.@printf(io, "%-14s %12.6g %12.6g %9.3f %10.3g %12.6g %12.6g\n", String(nm[j]), θ[j], se[j], z[j], p[j], lo[j], hi[j])
            end
            println(io, "────────────────────────────────────────────────────────────────────────────────────────")
        end
    end
end
