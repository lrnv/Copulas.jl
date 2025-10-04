
###############################################################################
#####  Fitting interface
#####  User-facing function:
#####   - `Distributions.fit(CopulaModel, MyCopulaType, data, method)`
#####   - `Distributions.fit(MyCopulaType, data, method)`
#####
#####  If you want your copula to be fittable byt he default interface, you can overwrite: 
#####   - _available_fitting_methods() to tell the system which method you allow. 
#####   - _fit(MyCopula, data, Val{:mymethod}) to make the fit.  
#####  
#####  Or, for simple models, to get access to a few default bindings, you could also override the following: 
#####   - Distributions.params() yielding a NamedTuple of parameters
#####   - _unbound_params() mappin your parameters to unbounded space
#####   - _rebound_params() doing the reverse
#####   - _example() giving example copula of your type. 
#####   - _example() giving example copula of your type. 
#####  
###############################################################################



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
[`StatsBase.nobs`](@ref), [`StatsBase.coef`](@ref), [`StatsBase.coefnames`](@ref), [`StatsBase.vcov`](@ref),
[`StatsBase.aic`](@ref), [`StatsBase.bic`](@ref), [`StatsBase.deviance`](@ref), etc.

See also [`Distributions.fit`](@ref) and [`_copula_of`](@ref).
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
"""
    Distributions.params(C::Copula)
    Distributions.params(S::SklarDist)

Return the parameters of the given distribution `C`. Our extension gives these parameters in a named tuple format. 

# Arguments
- `C::Distributions.Distribution`: The distribution object whose parameters are to be retrieved. Copulas.jl implements particular bindings for SklarDist and Copula objects. 

# Returns
- A named tuple containing the parameters of the distribution in the order they are defined for that distribution type.
"""
Distributions.params(C::Copula) = throw("You need to specify the Distributions.params() function as returning a named tuple with parameters.")
_example(CT::Type{<:Copula}, d) = throw("You need to specify the `_example(CT::Type{T}, d)` function for your copula type, returning an example of the copula type in dimension d.")
_unbound_params(CT::Type{Copula}, d, θ) = throw("You need to specify the _unbound_param method, that takes the namedtuple returned by `Distributions.params(CT(d, θ))` and trasform it into a raw vector living in R^p.")
_rebound_params(CT::Type{Copula}, d, α) = throw("You need to specify the _rebound_param method, that takes the output of _unbound_params and reconstruct the namedtuple that `Distributions.params(C)` would have returned.")
function _fit(CT::Type{<:Copula}, U, ::Val{:mle})
    # generic MLE routine (agnostic to vcov/inference)
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
    Ĉ   = CT(d, θhat...)
    return Ĉ,
           (; θ̂ = θhat, optimizer = Optim.summary(res), converged = Optim.converged(res), iterations= Optim.iterations(res))
end

"""
    _fit(::Type{<:Copula}, U, ::Val{method}; kwargs...)

Internal entry point for fitting routines.

Each copula family implements `_fit` methods specialized on `Val{method}`.
They must return a pair `(copula, meta)` where:
- `copula` is the fitted copula instance,
- `meta::NamedTuple` holds method–specific metadata to be stored in `method_details`.

This is not intended for direct use by end–users.  
Use [`Distributions.fit(CopulaModel, ...)`] instead.
"""
function _fit(CT::Type{<:Copula}, U, method::Union{Val{:itau}, Val{:irho}, Val{:ibeta}})
    # generic rank-based routine (agnostic to vcov/inference)
    d   = size(U,1)

    cop(α) = CT(d, _rebound_params(CT, d, α)...)
    α₀     = _unbound_params(CT, d, Distributions.params(_example(CT, d)))
    @assert length(α₀) <= d*(d-1)÷2 "Cannot use $method since there are too much parameters."

    fun  = method isa Val{:itau} ? StatsBase.corkendall :
           method isa Val{:irho} ? StatsBase.corspearman : corblomqvist
    est  = fun(U')
    loss(C) = sum(abs2, est .- fun(C))

    res  = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    Ĉ   = CT(d, θhat...)

    return Ĉ, (; θ̂=θhat,
                optimizer  = Optim.summary(res),
                converged  = Optim.converged(res),
                iterations = Optim.iterations(res))
end

"""
    Distributions.fit(CT::Type{<:Copula}, U; kwargs...) -> CT

Quick fit: devuelve solo la cópula ajustada (atajo de `Distributions.fit(CopulaModel, CT, U; summaries=false, kwargs...).result`).
"""
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U, method; kwargs...) = Distributions.fit(T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; copula_method=method, kwargs...)
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U; kwargs...) = Distributions.fit(CopulaModel, T, U; summaries=false, derived_measures=false, vcov=false, kwargs...).result

"""
    _available_fitting_methods(::Type{<:Copula})

Return the tuple of fitting methods available for a given copula family.

This is used internally by [`Distributions.fit`](@ref) to check validity of the `method` argument
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
function Distributions.fit(::Type{CopulaModel}, CT::Type{<:Copula}, U; method=:default, summaries::Bool=true, derived_measures::Bool=true, vcov::Bool=true, vcov_method::Union{Symbol,Nothing}=nothing, kwargs...)
    d, n = size(U)
    method = _find_method(CT, method)

    t = @elapsed (rez = _fit(CT, U, Val{method}(); kwargs...))
    C, meta = rez
    ll = Distributions.loglikelihood(C, U)

    # centralized vcov computation (outside _fit)
    meta2 = meta
    if vcov
        θnt = get(meta2, :θ̂, nothing)
        if θnt !== nothing
            α̂ = _unbound_params(CT, d, θnt)
            chosen = vcov_method !== nothing ? vcov_method : (
                method === :mle ? :hessian :
                (method === :itau || method === :irho || method === :ibeta) ? :godambe : :jackknife)
            Vθ = nothing; vmeta = NamedTuple()
            if chosen === :hessian
                Vθ, vmeta = _vcov_hessian(CT, U, α̂)
                if any(!isfinite, Matrix(Vθ))
                    Vθ, vmeta = _vcov_jackknife_obs(CT, U; estimator=method)
                end
            elseif chosen === :godambe
                Vθ, vmeta = _vcov_godambe_gmm(CT, U, α̂, Val{method}())
            else
                Vθ, vmeta = _vcov_jackknife_obs(CT, U; estimator=method)
            end
            meta2 = merge(meta2, (; vcov=Vθ, vmeta...))
        end
    end

    md = (; d, n, method, meta2..., null_ll=0.0,
           elapsed_sec=t, derived_measures,
           _extra_pairwise_stats(U, !summaries)...)

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
function Distributions.fit(::Type{CopulaModel}, ::Type{SklarDist{CT,TplMargins}}, X;
                           copula_method = :default, sklar_method = :default,
                           summaries::Bool = true, margins_kwargs = NamedTuple(),
                           copula_kwargs = NamedTuple(), 
                           derived_measures::Bool = true, vcov::Bool = true,
                           vcov_method::Union{Symbol,Nothing}=nothing) where
                           {CT<:Copulas.Copula, TplMargins<:Tuple}

    sklar_method  = _find_method(SklarDist, sklar_method)
    copula_method = _find_method(CT, copula_method)
    d, n = size(X)
    marg_types = TplMargins.parameters
    (length(marg_types) == d) || throw(ArgumentError("SklarDist: #marginals $(length(marg_types)) ≠ d=$d"))

    m = ntuple(i -> Distributions.fit(marg_types[i], @view X[i, :]; margins_kwargs...), d)

    # marginal vcov (placeholder: not computed here by default)
    Vm = fill(nothing, d)

    # --- construct U from fit
    ε = eps(Float64)
    U_fit = similar(X); U_ll = similar(X)

    if sklar_method === :ifm
        @inbounds for i in 1:d
            Ui = Distributions.cdf.(m[i], @view X[i,:])
            @views U_fit[i,:] .= clamp.(Ui, ε, 1-ε)
        end
        U_ll .= U_fit
    else
        U_fit .= pseudos(X)
        @inbounds for i in 1:d
            Ui = Distributions.cdf.(m[i], @view X[i,:])
            @views U_ll[i,:] .= clamp.(Ui, ε, 1-ε)
        end
    end

    # fit copula by reusing the Copula fit (avoid redundancy)
    copM = Distributions.fit(CopulaModel, CT, U_fit;
                             method=copula_method,
                             summaries=false,
                             derived_measures=derived_measures,
                             vcov=vcov,
                             vcov_method=vcov_method,
                             copula_kwargs...)
    C = copM.result
    Vfull = StatsBase.vcov(copM)

    # total likelihood
    ll_marg = 0.0
    @inbounds for j in axes(X, 2), i in 1:d
        ll_marg += Distributions.logpdf(m[i], X[i, j])
    end
    ll_cop = Distributions.loglikelihood(C, U_ll)
    ll = ll_marg + ll_cop

    null_ll = ll_marg

    md = (; copM.method_details...,
        vcov_copula   = StatsBase.vcov(copM),
           vcov_margins  = Vm,
           null_ll,
           sklar_method,
           margins       = map(typeof, m),
           has_summaries = summaries,
           d = d, n = n,
           elapsed_sec = copM.elapsed_sec,
           derived_measures,
           X_margins = [copy(@view X[i,:]) for i in 1:d],
           _extra_pairwise_stats(U_fit, !summaries)...)

    S = SklarDist(C, m)
    return CopulaModel(
        S, n, ll, copula_method;
        vcov         = Vfull,
        converged    = copM.converged,
        iterations   = copM.iterations,
        elapsed_sec  = copM.elapsed_sec,
        method_details = md
    )
end

function _uppertriangle_stats(mat)
    # compute the mean and std of the upper triangular part of the matrix (diagonal excluded)
    gen = [mat[idx] for idx in CartesianIndices(mat) if idx[1] < idx[2]]
    return Statistics.mean(gen), length(gen) == 1 ? zero(gen[1]) : Statistics.std(gen), minimum(gen), maximum(gen)
end
function _extra_pairwise_stats(U::AbstractMatrix, bypass::Bool)
    bypass && return (;)
    τm, τs, τmin, τmax = _uppertriangle_stats(StatsBase.corkendall(U'))
    ρm, ρs, ρmin, ρmax = _uppertriangle_stats(StatsBase.corspearman(U'))
    βm, βs, βmin, βmax = _uppertriangle_stats(corblomqvist(U'))
    γm, γs, γmin, γmax = _uppertriangle_stats(corgini(U'))
    return (; tau_mean=τm, tau_sd=τs, tau_min=τmin, tau_max=τmax,
             rho_mean=ρm, rho_sd=ρs, rho_min=ρmin, rho_max=ρmax,
             beta_mean=βm, beta_sd=βs, beta_min=βmin, beta_max=βmax,
             gamma_mean=γm, gamma_sd=γs, gamma_min=γmin, gamma_max=γmax)
end
####### vcov functions...
function _vcov_hessian(CT::Type{<:Copula}, U::AbstractMatrix, α̂::AbstractVector)
    d  = size(U,1)
    cop(α) = CT(d, _rebound_params(CT,d,α)...)
    ℓ(α)   = Distributions.loglikelihood(cop(α), U)
    Hα     = ForwardDiff.hessian(ℓ, α̂)
    infoα  = -Array(Hα)

    if any(!isfinite, infoα)
        return fill(NaN, length(α̂), length(α̂)), (; vcov_method=:hessian_fail, d=d)
    end
    infoα += 1e-8LinearAlgebra.I
    Vα = inv(infoα)

    θvec_of_α = α -> begin
        T  = eltype(α)
        nt = _rebound_params(CT, d, α)
        out = Vector{T}()
        for val in values(nt)
            if val isa Number
                push!(out, T(val))
            elseif val isa AbstractVector
                append!(out, T.(val))
            elseif val isa AbstractMatrix
                append!(out, vec(T.(val)))
            else
                try
                    push!(out, T(val))
                catch
                    # ignored non numerical values
                end
            end
        end
        out
    end

    J  = Array(ForwardDiff.jacobian(θvec_of_α, α̂))

    # Var(θ̂) via delta method
    Vθ = J * Vα * J'
    Vθ = (Vθ + Vθ')/2  # simetrización

    # 🔒 Regularización de autovalores negativos
    λ, Q = LinearAlgebra.eigen(Matrix(Vθ))
    λ_reg = map(x -> max(x, 1e-12), λ)  # fuerza semidefinitud
    Vθ = LinearAlgebra.Symmetric(Q * LinearAlgebra.Diagonal(λ_reg) * Q')

    return Vθ, (; vcov_method=:hessian, d=d)
end

function _vcov_godambe_gmm(CT::Type{<:Copula}, U::AbstractMatrix, α̂::AbstractVector, method::Union{Val{:itau}, Val{:irho}, Val{:ibeta}, Val{:iupper}})
    d, n = size(U)
    φ = method isa Val{:itau}  ? (α -> τ(CT(d, _rebound_params(CT,d,α)...))) :
        method isa Val{:irho}  ? (α -> ρ(CT(d, _rebound_params(CT,d,α)...))) :
        method isa Val{:ibeta} ? (α -> β(CT(d, _rebound_params(CT,d,α)...))) :
                                 (α -> λᵤ(CT(d, _rebound_params(CT,d,α)...)))

    m = method isa Val{:itau} ? τ : method isa Val{:irho} ? ρ : method isa Val{:ibeta} ? β : λᵤ

    g  = ForwardDiff.gradient(φ, α̂)
    Dα = reshape(g, 1, :)

    # Ω = Var(√n m̂) jackknife
    s   = Vector{Float64}(undef, n)
    idx = Vector{Int}(undef, n-1)
    for j in 1:n
        k=1; @inbounds for t in 1:n; if t==j; continue; end; idx[k]=t; k+=1; end
        s[j] = m(@view U[:,idx])
    end
    μ    = Statistics.mean(s)
    Vhat = (n-1)/n * sum((s .- μ).^2) / (n-1)
    Ω    = n * Vhat

    DtD = Dα' * Dα
    Va  = inv(DtD) * (Dα' * Ω * Dα) * inv(DtD) / n

    # Delta method α→θ
    J  = ForwardDiff.jacobian(α -> collect(values(_rebound_params(CT,d,α))), α̂)
    Vθ = (J*Va*J' + (J*Va*J')')/2
    return Vθ, (; vcov_method=:godambe_gmm, estimator=method, d=d, n=n, q=1)
end

function _vcov_jackknife_obs(CT::Type{<:Copula}, U::AbstractMatrix; estimator::Symbol, kw...)
    d = size(U,1)
    n = size(U,2)
    d ≥ 2 || throw(ArgumentError("jackknife requires d≥2."))
    n ≥ 3 || throw(ArgumentError("jackknife requires n≥3."))

    θminus = Matrix{Float64}(undef, n, 0)
    idx = Vector{Int}(undef, n-1)

    for j in 1:n
        k = 1
        for t in 1:n
            if t == j; continue; end
            idx[k] = t; k += 1
        end
        Uminus = @view U[:, idx]
    M = Distributions.fit(CopulaModel, CT, Uminus; method=estimator, summaries=false, vcov=false, derived_measures=false, kw...)
        θj = StatsBase.coef(M)
        if size(θminus,2) == 0
            θminus = Matrix{Float64}(undef, n, length(θj))
        end
        θminus[j, :] .= θj
    end

    θbar = vec(Statistics.mean(θminus, dims=1))
    V = (n-1)/n * (LinearAlgebra.transpose(θminus .- θbar') * (θminus .- θbar')) ./ (n-1)
    return V, (; vcov_method=:jackknife_obs, n=n)
end

function _vcov_safe(CT::Type{<:Copula}, U::AbstractMatrix, α̂::AbstractVector; estimator::Symbol=:mle)
    try
        Vθ, meta = _vcov_hessian(CT, U, α̂)
        if any(!isfinite, Vθ)
            @warn "vcov(hessian) failed (NaN/Inf). Falling back to jackknife."
            Vθ, meta = _vcov_jackknife_obs(CT, U; estimator)
        end
        return Vθ, meta
    catch err
        @warn "vcov(hessian) threw $err. Falling back to jackknife."
        return _vcov_jackknife_obs(CT, U; estimator)
    end
end
#####3
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
StatsBase.deviance(M::CopulaModel) = -2 * M.ll
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
StatsBase.aic(M::CopulaModel) = 2*StatsBase.dof(M) - 2*M.ll

"""
    bic(M::CopulaModel) -> Float64

Bayesian information criterion for the fitted model.
"""
StatsBase.bic(M::CopulaModel) = StatsBase.dof(M)*log(StatsBase.nobs(M)) - 2*M.ll
function aicc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    corr = (n > k + 1) ? (2k*(k+1)) / (n - k - 1) : Inf
    return StatsBase.aic(M) + corr
end
function hqc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    return -2*M.ll + 2k*log(log(max(n, 3)))
end

function StatsBase.nullloglikelihood(M::CopulaModel)
    if hasproperty(M.method_details, :null_ll)
        return getfield(M.method_details, :null_ll)
    else
        throw(ArgumentError("nullloglikelihood not available in method_details."))
    end
end
StatsBase.nulldeviance(M::CopulaModel) = -2 * StatsBase.nullloglikelihood(M)