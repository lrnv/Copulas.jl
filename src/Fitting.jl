"""
    CopulaModel{CT, TM, TD} <: StatsBase.StatisticalModel

A fitted copula model.

This type stores the result of fitting a copula (or a Sklar distribution) to
pseudo-observations or raw data, together with auxiliary information useful
for statistical inference and model comparison.

# Fields
- `result::CT`          — the fitted copula (or `SklarDist`).
- `n::Int`              — number of observations used in the fit.
- `ll::Float64`         — log-likelihood at the optimum.
- `method::Symbol`      — fitting method used (e.g. `:mle`, `:itau`, `:deheuvels`).
- `vcov::Union{Nothing, AbstractMatrix}` — estimated covariance of the parameters, if available.
- `converged::Bool`     — whether the optimizer reported convergence.
- `iterations::Int`     — number of iterations used in optimization.
- `elapsed_sec::Float64` — time spent in fitting.
- `method_details::NamedTuple` — additional method-specific metadata (grid size, pseudo-values, etc.).

`CopulaModel` implements the standard `StatsBase.StatisticalModel` interface:
[`nobs`](@ref), [`coef`](@ref), [`coefnames`](@ref), [`vcov`](@ref),
[`aic`](@ref), [`bic`](@ref), [`deviance`](@ref), etc.

See also [`fit`](@ref) and [`_copula_of`](@ref).
"""
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
                         method_details=NamedTuple()) where {CT}
        return new{CT, typeof(vcov), typeof(method_details)}(
            c, n, float(ll), method, vcov, converged, iterations, float(elapsed_sec), method_details
        )
    end
end

# Fallbacks that throw if the interface s not implemented correctly. 
Distributions.params(C::Copula) = throw("You need to specify the Distributions.params() function as returning a named tuple with parameters.")
_example(CT::Type{<:Copula}, d) = throw("You need to specify the `_example(CT::Type{T}, d)` function for your copula type, returning an example of the copula type in dimension d.")
_unbound_params(CT::Type{Copula}, d, θ) = throw("You need to specify the _unbound_param method, that takes the namedtuple returned by `Distributions.params(CT(d, θ))` and trasform it into a raw vector living in R^p.")
_rebound_params(CT::Type{Copula}, d, α) = throw("You need to specify the _rebound_param method, that takes the output of _unbound_params and reconstruct the namedtuple that `Distributions.params(C)` would have returned.")
function _fit(CT::Type{<:Copula}, U, ::Val{:mle})
    # @info "Running the MLE routine from the generic implementation"
    d   = size(U,1)
    function cop(α)
        par = _rebound_params(CT, d, α)
        return CT(d, par...) ####### Using a "," here forces the constructor to accept raw values, while a ";" passes named values. Not sure which is best. 
    end
    α₀  = _unbound_params(CT, d, Distributions.params(_example(CT, d)))

    loss(C) = -Distributions.loglikelihood(C, U)
    res = try
        Optim.optimize(loss ∘ cop, α₀, Optim.LBFGS(); autodiff=:forward)
    catch err
        # @warn "LBFGS with AD failed ($err), retrying with NelderMead"
        Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    end
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return CT(d, θhat...),
           (; θ̂=θhat, optimizer = Optim.summary(res), converged = Optim.converged(res), iterations = Optim.iterations(res))
end

"""
    _fit(::Type{<:Copula}, U, ::Val{method}; kwargs...)

Internal entry point for fitting routines.

Each copula family implements `_fit` methods specialized on `Val{method}`.
They must return a pair `(copula, meta)` where:
- `copula` is the fitted copula instance,
- `meta::NamedTuple` holds method–specific metadata to be stored in `method_details`.

This is not intended for direct use by end–users.  
Use [`fit(CopulaModel, ...)`](@ref) instead.
"""
function _fit(CT::Type{<:Copula}, U, method::Union{Val{:itau}, Val{:irho}, Val{:ibeta}})
    # @info "Running the itau/irho/ibeta routine from the generic implementation"
    d   = size(U,1)

    cop(α) = CT(d, _rebound_params(CT, d, α)...)
    α₀  = _unbound_params(CT, d, Distributions.params(_example(CT, d)))
    @assert length(α₀) <= d*(d-1)/2 "Cannot use $method since there are too much parameters."

    fun = method isa Val{:itau} ? StatsBase.corkendall : method isa Val{:irho} ? StatsBase.corspearman : corblomqvist
    est = fun(U')
    loss(C) = sum(abs2, est .- fun(C))

    res = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return CT(d, θhat...),
           (; θ̂=θhat, optimizer = Optim.summary(res), converged = Optim.converged(res), iterations = Optim.iterations(res))
end
"""
    fit(CT::Type{<:Copula}, U; kwargs...) -> CT

Quick fit: devuelve solo la cópula ajustada (atajo de `fit(CopulaModel, CT, U; summaries=false, kwargs...).result`).
"""
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U, method; kwargs...) = Distributions.fit(T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; copula_method=method, kwargs...)
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U; kwargs...) = Distributions.fit(CopulaModel, T, U; summaries=false, kwargs...).result
"""
    _available_fitting_methods(::Type{<:Copula})

Return the tuple of fitting methods available for a given copula family.

This is used internally by [`fit`](@ref) to check validity of the `method` argument
and to select a default method when `method=:default`.

# Example
```julia
_available_fitting_methods(GumbelCopula)
# → (:mle, :itau, :irho, :ibeta)
```
"""
_available_fitting_methods(::Type{<:Copula}) = (:mle, :itau, :irho, :ibeta)
_available_fitting_methods(C::Copula) = _available_fitting_methods(typeof(C))

function _find_method(CT, method)
    avail = _available_fitting_methods(CT)
    isempty(avail) && error("No fitting methods available for $CT.")
    if method === :default 
        method = avail[1]
        # @info "Choosing default method '$(method)' among $avail..."
    elseif method ∉ avail 
        error("Method '$method' not available for $CT. Available: $(join(avail, ", ")).")
    end
    return method
end
"""
    fit(CopulaModel, CT::Type{<:Copula}, U; method=:default, summaries=true, kwargs...)

Fit a copula of type `CT` to pseudo-observations `U`.

# Arguments
- `U::AbstractMatrix` — a `d×n` matrix of data (each column is an observation).
  If the input is raw data, use `SklarDist` fitting instead to estimate both
  margins and copula simultaneously.
- `method::Symbol`    — fitting method; defaults to the first available one
  (see [`_available_fitting_methods`](@ref)).
- `summaries::Bool`   — whether to compute pairwise summary statistics
  (Kendall's τ, Spearman's ρ, Blomqvist's β).
- `kwargs...`         — additional method-specific keyword arguments
  (e.g. `pseudo_values=true`, `grid=401` for extreme-value tails, etc.).

# Returns
A [`CopulaModel`](@ref) containing the fitted copula and metadata.

# Examples
```julia
U = rand(GumbelCopula(2, 3.0), 500)

M = fit(CopulaModel, GumbelCopula, U; method=:mle)
println(M)

# Quick fit: returns only the copula
C = fit(GumbelCopula, U; method=:itau)
```
"""
function Distributions.fit(::Type{CopulaModel}, CT::Type{<:Copula}, U; method = :default, summaries=true, kwargs...)
    d, n = size(U)
    # Choose the fitting method: 
    method = _find_method(CT, method)

    t = @elapsed (rez = _fit(CT, U, Val{method}(); kwargs...))
    C, meta = rez
    ll = Distributions.loglikelihood(C, U)
    md = (; d, n, method, meta..., null_ll=0.0, elapsed_sec=t, _extra_pairwise_stats(U, !summaries)...)
    return CopulaModel(C, n, ll, method;
        vcov         = get(md, :vcov, nothing),
        converged    = get(md, :converged, true),
        iterations   = get(md, :iterations, 0),
        elapsed_sec  = get(md, :elapsed_sec, NaN),
        method_details = md)
end

_available_fitting_methods(::Type{SklarDist}) = (:ifm, :ecdf)
"""
    fit(CopulaModel, SklarDist{CT, TplMargins}, X; copula_method=:default, sklar_method=:default,
                                           summaries=true, margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple())

Joint margin and copula adjustment (Sklar approach).
`sklar_method ∈ (:ifm, :ecdf)` controls whether parametric CDFs (`:ifm`) or pseudo-observations (`:ecdf`) are used.
"""
function Distributions.fit(::Type{CopulaModel},::Type{SklarDist{CT,TplMargins}}, X; copula_method = :default, sklar_method = :default,
                           summaries = true, margins_kwargs = NamedTuple(), copula_kwargs = NamedTuple()) where
                           {CT<:Copulas.Copula, TplMargins<:Tuple}

    sklar_method = _find_method(SklarDist, sklar_method)
    copula_method = _find_method(CT, copula_method)

    d, n = size(X)
    marg_types = TplMargins.parameters
    (length(marg_types) == d) || throw(ArgumentError("SklarDist: #marginals $(length(marg_types)) ≠ d=$d"))
    m = ntuple(i -> Distributions.fit(marg_types[i], @view X[i, :]; margins_kwargs...), d)
    # Only one margins_kwargs while people mught want to pass diferent kwargs for diferent marginals... but OK for the moment.

    
    U = similar(X)
    if sklar_method === :ifm
        for i in 1:d
            U[i,:] .= Distributions.cdf.(m[i], X[i,:])
        end
    elseif sklar_method === :ecdf
        U .= pseudos(X)
    end

    # Copula fit... with method specific
    C, cmeta = _fit(CT, U, Val{copula_method}(); copula_kwargs...)

    S  = SklarDist(C, m)
    ll = Distributions.loglikelihood(S, X)

    null_ll = 0.0
    @inbounds for j in axes(X, 2)
        for i in 1:d
            null_ll += Distributions.logpdf.(m[i], X[i, j])
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

function _extra_pairwise_stats(U::AbstractMatrix, bypass::Bool)
    bypass && return (;)
    τm, τs, τmin, τmax = _uppertriangle_stats(StatsBase.corkendall(U'))
    ρm, ρs, ρmin, ρmax = _uppertriangle_stats(StatsBase.corspearman(U'))
    βm, βs, βmin, βmax = _uppertriangle_stats(corblomqvist(U'))
    return (; tau_mean=τm, tau_sd=τs, tau_min=τmin, tau_max=τmax,
             rho_mean=ρm, rho_sd=ρs, rho_min=ρmin, rho_max=ρmax,
             beta_mean=βm, beta_sd=βs, beta_min=βmin, beta_max=βmax)
end
Distributions.loglikelihood(C::Copulas.Copula, U::AbstractMatrix{<:Real}) = sum(Base.Fix1(Distributions.logpdf, C), eachcol(U))
Distributions.loglikelihood(C::Copulas.Copula, u::AbstractVector{<:Real}) = Distributions.logpdf(C, u)
Distributions.loglikelihood(M::CopulaModel) = M.ll
Distributions.loglikelihood(M::CopulaModel, U::AbstractMatrix) = Distributions.loglikelihood(M.result, U)

"""
    nobs(M::CopulaModel) -> Int

Number of observations used in the model fit.
"""
StatsBase.nobs(M::CopulaModel)     = M.n
StatsBase.isfitted(::CopulaModel)  = true

"""
    deviance(M::CopulaModel) -> Float64

Deviation of the fitted model (-2 * loglikelihood).
"""
StatsBase.deviance(M::CopulaModel) = -2 * Distributions.loglikelihood(M)
StatsBase.dof(M::CopulaModel) = StatsBase.dof(M.result)

"""
    _copula_of(M::CopulaModel)

Returns the copula object contained in the model, even if the result is a `SklarDist`.
"""
_copula_of(M::CopulaModel)   = M.result isa SklarDist ? M.result.C : M.result

"""
    coef(M::CopulaModel) -> Vector{Float64}

Vector with the estimated parameters of the copula.
"""
StatsBase.coef(M::CopulaModel) = collect(values(Distributions.params(_copula_of(M)))) # why ? params of the marginals should also be taken into account. 

"""
    coefnames(M::CopulaModel) -> Vector{String}

Names of the estimated copula parameters.
"""
StatsBase.coefnames(M::CopulaModel) = string.(keys(Distributions.params(_copula_of(M))))
StatsBase.dof(C::Copulas.Copula) = length(values(Distributions.params(C)))

#(optional vcov) and vcov its very important... for inference 
"""
    vcov(M::CopulaModel) -> Union{Nothing, Matrix{Float64}}

Variance and covariance matrix of the estimators.
Can be `nothing` if not available.
"""
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

"""
    aic(M::CopulaModel) -> Float64

Akaike information criterion for the fitted model.
"""
StatsBase.aic(M::CopulaModel) = 2*StatsBase.dof(M) - 2*Distributions.loglikelihood(M)

"""
    bic(M::CopulaModel) -> Float64

Bayesian information criterion for the fitted model.
"""
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



# helper
params(::Type{T}) where {T<:Copula} = throw("No params() function defined for type T = $T...")




##############################################################################################################################
####### show function. PLease do not construct other functions for it. 
##############################################################################################################################
function _linearize_params(params::NamedTuple)
    vals = Float64[]
    names = String[]
    for (k, v) in pairs(params)
        if isa(v, Number)
            push!(vals, float(v))
            push!(names, String(k))
        elseif isa(v, AbstractMatrix)
            for i in axes(v, 1), j in axes(v, 2)
                push!(vals, float(v[i, j]))
                push!(names, "$(k)_$(i)_$(j)")
            end
        elseif isa(v, AbstractVector)
            for i in eachindex(v)
                push!(vals, float(v[i]))
                push!(names, "$(k)_$(i)")
            end
        else
            try
                push!(vals, float(v))
                push!(names, String(k))
            catch
            end
        end
    end
    return vals, names
end
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
    Printf.@printf(io, "Number of observations: %9d\n", n)

    ll0 = get(M.method_details, :null_ll, NaN)
    if isfinite(ll0)
        Printf.@printf(io, "Null Loglikelihood:  %12.4f\n", ll0)
    end
    Printf.@printf(io, "Loglikelihood:       %12.4f\n", ll)

    # Para el test LR usa g.l. de la CÓPULA si es SklarDist
    kcop = (R isa SklarDist) ? StatsBase.dof(_copula_of(M)) : StatsBase.dof(M)
    if isfinite(ll0) && kcop > 0
        LR = 2*(ll - ll0)
        p  = Distributions.ccdf(Distributions.Chisq(kcop), LR)
        Printf.@printf(io, "LR Test (vs indep. copula): %.2f ~ χ²(%d)  =>  p = %.4g\n", LR, kcop, p)
    end

    aic = StatsBase.aic(M); bic = StatsBase.bic(M)
    Printf.@printf(io, "AIC: %.3f       BIC: %.3f\n", aic, bic)
    if isfinite(M.elapsed_sec) || M.iterations != 0 || M.converged != true
        conv = M.converged ? "true" : "false"
        tsec = isfinite(M.elapsed_sec) ? Printf.@sprintf("%.3fs", M.elapsed_sec) : "NA"
        println(io, "Converged: $(conv)   Iterations: $(M.iterations)   Elapsed: $(tsec)")
    end

    # Branches: SklarDist → sections; empirical → summary; else → coefficient table
    if R isa SklarDist
        # [ Copula ] section
        C = _copula_of(M)
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
        kind = get(md, :emp_kind, :unspecified)
        d    = get(md, :d, missing)
        n    = get(md, :n, missing)
        pv   = get(md, :pseudo_values, missing)

        hdr = "d=$(d), n=$(n)" * (pv === missing ? "" : ", pseudo_values=$(pv)")
        extra = ""
        if kind === :bernstein
            m = get(md, :m, nothing)
            extra = m === nothing ? "" : ", m=$(m)"
        elseif kind === :exact
            m = get(md, :m, nothing)
            extra = m === nothing ? "" : ", m=$(m)"
        elseif kind === :ev_tail
            method = get(md, :method, :unspecified)
            grid   = get(md, :grid, missing)
            eps    = get(md, :eps,  missing)
            extra  = ", method=$(method), grid=$(grid), eps=$(eps)"
        end

        println(io, "Empirical summary ($kind)")
        println(io, hdr * extra)

        # Estadísticos clásicos
        has_tau  = all(haskey.(Ref(md), (:tau_mean, :tau_sd, :tau_min, :tau_max)))
        has_rho  = all(haskey.(Ref(md), (:rho_mean, :rho_sd, :rho_min, :rho_max)))
        has_beta = all(haskey.(Ref(md), (:beta_mean, :beta_sd, :beta_min, :beta_max)))

        if d === missing || d == 2
            println(io, "────────────────────────────")
            Printf.@printf(io, "%-10s %18s\n", "Stat", "Value")
            println(io, "────────────────────────────")
                if has_tau; Printf.@printf(io, "%-10s %18.3f\n", "tau", md[:tau_mean]); end
                if has_rho; Printf.@printf(io, "%-10s %18.3f\n", "rho", md[:rho_mean]); end
                if has_beta; Printf.@printf(io, "%-10s %18.3f\n", "beta", md[:beta_mean]); end
            println(io, "────────────────────────────")
        else
            println(io, "───────────────────────────────────────────────────────")
            Printf.@printf(io, "%-10s %10s %10s %10s %10s\n", "Stat", "Mean", "SD", "Min", "Max")
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
                Printf.@printf(io, "%-10s %10.3f %10.3f %10.3f %10.3f\n",
                    "beta", md[:beta_mean], md[:beta_sd], md[:beta_min], md[:beta_max])
            end
            println(io, "───────────────────────────────────────────────────────")
        end
    else
        # Coefficient table
        params = Distributions.params(_copula_of(M))
        θ, nm = _linearize_params(params)
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
