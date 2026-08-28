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

See also [`Distributions.fit`](@ref).
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

# Fallbacks that throw if the interface is not implemented correctly.
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
_fit_copula(CT::Type{<:Copula}, d, θ, example) = CT(d, θ...)
function _fit(CT::Type{<:Copula}, U, ::Val{:mle})
    # generic MLE routine (agnostic to vcov/inference)
    d   = size(U,1)
    example = _example(CT, d)
    cop(α) = _fit_copula(CT, d, _rebound_params(CT, d, α), example)
    α₀  = _unbound_params(CT, d, Distributions.params(example))
    loss(C) = -Distributions.loglikelihood(C, U)
    res = try
        Optim.optimize(loss ∘ cop, α₀, Optim.LBFGS(); autodiff= ADTypes.AutoForwardDiff())
    catch err
        Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    end
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return _fit_copula(CT, d, θhat, example), (; θ̂=θhat,
                optimizer  = Optim.summary(res),
                converged  = Optim.converged(res),
                iterations = Optim.iterations(res))
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
    example = _example(CT, d)
    cop(α) = _fit_copula(CT, d, _rebound_params(CT, d, α), example)
    α₀ = _unbound_params(CT, d, Distributions.params(example))
    @assert length(α₀) <= d*(d-1)÷2 "Cannot use $method since there are too much parameters."
    fun  = method isa Val{:itau} ? StatsBase.corkendall :
           method isa Val{:irho} ? StatsBase.corspearman : corblomqvist
    est  = fun(U')
    loss(C) = sum(abs2, est .- fun(C))
    res  = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return _fit_copula(CT, d, θhat, example), (; θ̂=θhat,
                optimizer  = Optim.summary(res),
                converged  = Optim.converged(res),
                iterations = Optim.iterations(res))
end


"""
    Distributions.fit(CT::Type{<:Copula}, U; kwargs...) -> CT

Quick fit: devuelve solo la cópula ajustada (atajo de `Distributions.fit(CopulaModel, CT, U; kwargs...)`).
"""
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U, method; kwargs...) = Distributions.fit(T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; copula_method=method, kwargs...)
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U; kwargs...) = Distributions.fit(CopulaModel, T, U; quick_fit=true, kwargs...).result

"""
    _available_fitting_methods(::Type{<:Copula}, d::Int)

Return the tuple of fitting methods available for a given copula family in a given dimension.

This is used internally by [`Distributions.fit`](@ref) to check validity of the `method` argument
and to select a default method when `method=:default`.

# Example
```julia
_available_fitting_methods(GumbelCopula, 3)
# → (:mle, :itau, :irho, :ibeta)
```
"""
_available_fitting_methods(::Type{<:Copula}, d) = (:mle, :itau, :irho, :ibeta)
_available_fitting_methods(C::Copula, d) = _available_fitting_methods(typeof(C), d)

function _find_method(CT, d, method)
    avail = _available_fitting_methods(CT, d)
    isempty(avail) && error("No fitting methods available for $CT.")
    method === :default && return avail[1]
    method ∉ avail && error("Method '$method' not available for $CT. Available: $(join(avail, ", ")).")
    return method
end

"""
    fit(CopulaModel, CT::Type{<:Copula}, U; method=:default, kwargs...)

Fit a copula of type `CT` to pseudo-observations `U`.

# Arguments
- `U::AbstractMatrix` — a `d×n` matrix of data (each column is an observation).
  If the input is raw data, use `SklarDist` fitting instead to estimate both
  margins and copula simultaneously.
- `method::Symbol`    — fitting method; defaults to the first available one
  (see [`_available_fitting_methods`](@ref)).
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
function Distributions.fit(::Type{CopulaModel}, CT::Type{<:Copula}, U;
        method=:default, quick_fit=false, derived_measures=true,
        vcov=true, vcov_method=nothing, kwargs...)
    d, n = size(U)
    method = _find_method(CT, d, method)
    t = @elapsed (rez = _fit(CT, U, Val{method}(); kwargs...))
    C, meta = rez
    quick_fit && return (result=C,) # as soon as possible.
    ll = Distributions.loglikelihood(C, U)

    if vcov && C isa TCopula
        vcov = false
        @info "Setting vcov = false for TCopula since _beta_inc_inv derivative are not implemented"
    end
    if vcov && C isa tEVCopula
        vcov = false
        @info "Setting vcov = false for tEVCopula since _beta_inc_inv derivative are not implemented"
    end
    if vcov && C isa FGMCopula && method==:mle
        vcov = false
        @info "Setting vcov = false for FGMCopula with method=:mle since unimplemented right now"
    end

    if vcov && haskey(meta, :θ̂)
        vcov, vmeta = _vcov(CT, U, meta.θ̂; method=method, override=vcov_method)
        meta = (; meta..., vcov, vmeta...)
    end

    md = (; d, n, method, meta..., null_ll=0.0,
        elapsed_sec=t, derived_measures, U=U)

    return CopulaModel(C, n, ll, method;
        vcov         = get(md, :vcov, nothing),
        converged    = get(md, :converged, true),
        iterations   = get(md, :iterations, 0),
        elapsed_sec  = get(md, :elapsed_sec, NaN),
        method_details = md)
end

_available_fitting_methods(::Type{SklarDist}, d) = (:ifm, :ecdf)
"""
    fit(CopulaModel, SklarDist{CT, TplMargins}, X; copula_method=:default, sklar_method=:default,
                                           margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple())

Joint margin and copula adjustment (Sklar approach).
`sklar_method ∈ (:ifm, :ecdf)` controls whether parametric CDFs (`:ifm`) or pseudo-observations (`:ecdf`) are used.
"""
function Distributions.fit(::Type{CopulaModel}, ::Type{SklarDist{CT,TplMargins}}, X; quick_fit = false,
                           copula_method = :default, sklar_method = :default, margins_kwargs = NamedTuple(),
                           copula_kwargs = NamedTuple(), derived_measures = true, vcov = true,
                           vcov_method=nothing) where {CT<:Copulas.Copula, TplMargins<:Tuple}

    # Get methods:
    d, n = size(X)
    sklar_method  = _find_method(SklarDist, d, sklar_method)
    copula_method = _find_method(CT, d, copula_method)

    # Fit marginals:
    m = ntuple(i -> Distributions.fit(TplMargins.parameters[i], @view X[i, :]; margins_kwargs...), d)

    # Make pseudo-observations
    uniform_type = foldl(
        (T, margin) -> promote_type(T, eltype(margin)),
        m;
        init=float(eltype(X)),
    )
    U = similar(X, uniform_type)
    if sklar_method === :ifm
        lower = nextfloat(zero(uniform_type))
        upper = prevfloat(one(uniform_type))
        @inbounds for j in axes(X, 2), i in axes(X, 1)
            U[i, j] = clamp(Distributions.cdf(m[i], X[i, j]), lower, upper)
        end
    else # :ecdf then
        U .= pseudos(X)
    end

    # Fit the copula
    copM = Distributions.fit(CopulaModel, CT, U; quick_fit=quick_fit,
                method=copula_method, derived_measures=derived_measures,
                vcov=vcov, vcov_method=vcov_method, copula_kwargs...)

    S = SklarDist(copM.result, m)
    quick_fit && return (result=S,)

    # Marginal vcov: compute via θ-Hessian fallback only if vcov=true
    Vm = Vector{Union{Nothing, Matrix{Float64}}}(undef, d)
    if vcov
        for i in 1:d
            p  = length(Distributions.params(m[i]))
            Vm[i] = nothing
            Vg = _vcov_margin_generic(m[i], @view X[i, :])
            if Vg !== nothing && ndims(Vg) == 2 && size(Vg) == (p, p) && all(isfinite, Matrix(Vg))
                Vm[i] = Matrix{Float64}(Vg)
            end
        end
    else
        fill!(Vm, nothing)
    end

    # Copula Vcov:
    Vfull = StatsBase.vcov(copM)

    # total and null loglikelihood
    ll = Distributions.loglikelihood(S, X)
    null_ll = Distributions.loglikelihood(SklarDist(IndependentCopula(d), m), X)
    return CopulaModel(
        S, n, ll, copula_method;
        vcov         = Vfull,
        converged    = copM.converged,
        iterations   = copM.iterations,
        elapsed_sec  = copM.elapsed_sec,
        method_details = (;
            copM.method_details...,
            vcov_copula   = Vfull,
            vcov_margins  = Vm,
            null_ll,
            sklar_method,
            margins       = map(typeof, m),
            d = d, n = n,
            elapsed_sec = copM.elapsed_sec,
            derived_measures,
            # no raw X_margins stored to keep model lightweight
        )
    )
end
####### vcov functions...

# objetive this functions: try get the vcov from marginals...
function _vcov_margin_generic(d::TD, x::AbstractVector) where {TD<:Distributions.UnivariateDistribution}
    # Compute observed information directly on the parameter (θ) scale at current params.
    p_nt = Distributions.params(d)
    θ0 = p_nt isa NamedTuple ? Float64.(collect(values(p_nt))) : Float64.(collect(p_nt))

    # Find the distribution constructor:
    MyDist = TD.name.wrapper
    # Observed information = - Hessian of log-likelihood at θ0
    H = ForwardDiff.hessian(θ -> Distributions.loglikelihood(MyDist(θ...), x), θ0)
    # Small ridge for numerical stability
    Vθ = inv(-H + 1e-8 .* LinearAlgebra.I)
    Vθ = (Vθ + Vθ')/2
    return LinearAlgebra.Symmetric(Matrix{Float64}(Vθ))
end

function _vcov(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple; method::Symbol, override::Union{Symbol,Nothing}=nothing)
    vcovm = !isnothing(override) ? override :
            method === :mle      ? :hessian :
            method === :itau     ? :godambe :
            method === :irho     ? :godambe :
            method === :ibeta    ? :godambe :
            method === :iupper   ? :godambe :  :jackknife

    if vcovm ∉ (:hessian, :godambe, :godambe_pairwise)
        return _vcov(CT, U, θ, Val{vcovm}(), Val{method}()) # you can write new methods through this interface, as the jacknife method below.
    end

    d, n = size(U)
    α  = _unbound_params(CT, d, θ)
    cop(α) = CT(d, _rebound_params(CT,d,α)...)
    _upper_triangle(A) = [A[idx] for idx in CartesianIndices(A) if idx[1] < idx[2]]

    if vcovm === :hessian
        ℓ(α) = Distributions.loglikelihood(cop(α), U)
        H  = ForwardDiff.hessian(ℓ, α)
        Iα = .-H
        if any(!isfinite, Iα)
            @warn "vcov(:hessian): non-finite Fisher information; falling back" Iα
            return _vcov(CT, U, θ, Val{:bootstrap}(), Val{method}())
        end
        Iα = (Iα + Iα')/2
        p   = size(Iα, 1)
        I_p = Matrix{Float64}(LinearAlgebra.I, p, p)
        λ = 1e-8
        Vα = nothing
        @inbounds for _ in 1:8
            A = Iα + λ*I_p
            ch = LinearAlgebra.cholesky(LinearAlgebra.Symmetric(A); check=false)
            if ch.info == 0                     # is p.d.
                Vα = ch \ I_p                   # It is equivalent to inv(A), but stable, we could use pinv but I don't know how optimal it is...
                break
            end
            λ *= 10
        end
        if Vα === nothing || any(!isfinite, Vα)
            @warn "vcov(:hessian): failed to stabilize Fisher; falling back" λ_final=λ
            return _vcov(CT, U, θ, Val{:bootstrap}(), Val{method}())
        end
    else

        pairwise_φ = method isa Val{:itau}  ? StatsBase.corkendall :
            method isa Val{:irho}  ? StatsBase.corspearman :
            method isa Val{:ibeta} ? corblomqvist : coruppertail
        φ = method isa Val{:itau}  ? τ :
                method isa Val{:irho}  ? ρ :
                method isa Val{:ibeta} ? β : λᵤ
        if vcovm === :godambe
            q = 1
            ψ = α -> [φ(cop(α))]
            ψ_emp = u ->[φ(u)]
        else # then :godambe_pairwise
            q = d*(d-1) ÷ 2
            ψ = α -> _upper_triangle(pairwise_φ(cop(α)))
            ψ_emp = U -> _upper_triangle(pairwise_φ(U'))
        end

        Dα = ForwardDiff.jacobian(ψ, α)
        Dα = reshape(Dα, q, length(α))

        # Ω bootstrap
        B   = clamp(Int(floor(sqrt(n))), 10, 200)
        M   = Matrix{Float64}(undef, B, q)
        idx = Vector{Int}(undef, n)
        rng = Random.default_rng()

        @inbounds for b in 1:B
            for i in 1:n
                idx[i] = rand(rng, 1:n)
            end
            Mb = @view U[:, idx]
            M[b, :] = ψ_emp(Mb)
        end
        Ω = n * Statistics.cov(M; corrected=true)
        DtD = Dα' * Dα
        ϵI  = 1e-10LinearAlgebra.I
        Vα  = inv(DtD + ϵI) * (Dα' * Ω * Dα) * inv(DtD + ϵI) / n
    end
    # Delta method Jacobian from α (unbounded) to θ (original params), flattened
    J  = ForwardDiff.jacobian(αv -> _flatten_params(_rebound_params(CT, d, αv))[2], α)
    Vθ = J * Vα * J'
    # <<<<<<< KEY CHANGE >>>>>>>>>
    # Check for finiteness BEFORE calling eigen.
    # If the matrix already contains Inf/NaN, the estimate was unstable.
    # We activate the fallback to jackknife immediately.
    if !all(isfinite, Vθ)
        return _vcov(CT, U, θ, Val{:bootstrap}(), Val{method}())
    end
    Vθ = (Vθ + Vθ')/2
    λ, Q = LinearAlgebra.eigen(Matrix(Vθ))
    λ_reg = map(x -> max(x, 1e-12), λ)
    Vθ = LinearAlgebra.Symmetric(Q * LinearAlgebra.Diagonal(λ_reg) * Q')
    # This final check is now a double security.
    any(!isfinite, Matrix(Vθ)) && return _vcov(CT, U, θ, Val{:jackknife}(), Val{method}())
    return Vθ, (; vcov_method=vcovm)
end
function _vcov(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple, ::Val{:jackknife}, ::Val{method}) where {method}
    d, n = size(U)
    θminus = zeros(n, length(θ))
    idx = Vector{Int}(undef, n-1)

    for j in 1:n
        k = 1; for t in 1:n; if t == j; continue; end; idx[k] = t; k += 1; end
        Uminus = @view U[:, idx]
        θminus[j, :] .= _flatten_params(_fit(CT, Uminus, Val{method}())[2].θ̂)[2]
    end

    θbar = vec(Statistics.mean(θminus, dims=1))
    V = (n-1)/n * (LinearAlgebra.transpose(θminus .- θbar') * (θminus .- θbar')) ./ (n-1)
    return V, (; vcov_method=:jackknife_obs)
end
# Fallback fast: bootstrap refit (B < n)
function _vcov(CT::Type{<:Copula}, U::AbstractMatrix, θ::NamedTuple, ::Val{:bootstrap}, ::Val{method}) where {method}
    d, n = size(U)
    p = length(_flatten_params(θ)[2])
    B = clamp(Int(floor(sqrt(n))), 10, 200)
    Θ   = Matrix{Float64}(undef, B, p)
    idx = Vector{Int}(undef, n)
    rng = Random.default_rng()
    @inbounds for b in 1:B
        for i in 1:n
            idx[i] = rand(rng, 1:n)
        end
        θminus = @view U[:, idx]
        Θ[b, :] .= _flatten_params(_fit(CT, θminus, Val{method}())[2].θ̂)[2]
    end
    V = Statistics.cov(Θ; corrected=true)
    return V, (; vcov_method=:bootstrap, B=B)
end



##### StatsBase interfaces.
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
StatsBase.dof(M::CopulaModel) = length(StatsBase.coef(M))

"""
    _copula_of(M::CopulaModel)

Returns the copula object contained in the model, even if the result is a `SklarDist`.
"""
_copula_of(M::CopulaModel)   = M.result isa SklarDist ? M.result.C : M.result

"""
    coef(M::CopulaModel) -> Vector{Float64}

Vector with the estimated parameters of the copula.
"""
StatsBase.coef(M::CopulaModel) = haskey(M.method_details, :θ̂) ? _flatten_params(M.method_details.θ̂)[2] : Float64[]

"""
coefnames(M::CopulaModel) -> Vector{String}

Names of the estimated copula parameters.
"""
StatsBase.coefnames(M::CopulaModel) = haskey(M.method_details, :θ̂) ? _flatten_params(M.method_details.θ̂)[1] : String[]


# Flatten a NamedTuple of parameters into a Vector{Float64},
# consistent with the generic linearization used in show().
function _flatten_params(params_nt::NamedTuple)
    nm = String[]
    θ = Any[]
    sidx = ["₁", "₂", "₃", "₄", "₅", "₆", "₇", "₈", "₉"]
    for (k, v) in pairs(params_nt)
        if v isa Number
            push!(nm, String(k))
            push!(θ, v)
        elseif v isa AbstractMatrix
            if maximum(size(v)) > 9
                @inbounds for j in 2:size(v,2), i in 1:j-1
                    push!(nm, "$(k)_$(i)_$(j)")
                    push!(θ, v[i,j])
                end
            else
                @inbounds for j in 2:size(v,2), i in 1:j-1
                    push!(nm, "$(k)$(sidx[i])$(sidx[j])")
                    push!(θ, v[i,j])
                end
            end
        elseif v isa AbstractVector
            if length(v) > 9
                for i in eachindex(v)
                    push!(nm, "$(k)_$(i)")
                    push!(θ, v[i])
                end
            else
                for i in eachindex(v)
                    push!(nm, "$(k)$(sidx[i])")
                    push!(θ, v[i])
                end
            end
        else
            try
                push!(nm, String(k))
                push!(θ, v)
            catch
            end
        end
    end
    return nm, [x for x in promote(θ...)]
end



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
"""
    StatsBase.residuals(M::CopulaModel; transform=:uniform)

Compute Rosenblatt residuals of a fitted copula model.

# Arguments
- `transform = :uniform` → returns Rosenblatt residuals in [0,1].
- `transform = :normal`  → applies Φ⁻¹ to obtain pseudo-normal residuals.

# Notes
The residuals should be i.i.d. Uniform(0,1) under a correctly specified model.
"""
StatsBase.residuals(M::CopulaModel; transform=:uniform) = begin
    haskey(M.method_details, :U) || throw(ArgumentError("method_details must contain pseudo-observations :U"))
    U = M.method_details[:U]
    R = rosenblatt(_copula_of(M), U)
    return transform === :normal ? Distributions.quantile.(Distributions.Normal(), R) : R
end
"""
    StatsBase.predict(M::CopulaModel; newdata=nothing, what=:cdf, nsim=0)

Predict or simulate from a fitted copula model.

# Keyword arguments
- `newdata` — matrix of points in [0,1]^d at which to evaluate (`what=:cdf` or `:pdf`).
- `what` — one of `:cdf`, `:pdf`, or `:simulate`.
- `nsim` — number of samples to simulate if `what=:simulate`.

# Returns
- Vector or matrix of predicted probabilities/densities, or simulated samples.
"""
function StatsBase.predict(M::CopulaModel; newdata=nothing, what=:cdf, nsim=0)
    C = _copula_of(M)
    return what === :simulate ? rand(C, nsim > 0 ? nsim : M.n) :
           what === :cdf      ? (newdata === nothing ? throw(ArgumentError("`newdata` required for `:cdf`")) : Distributions.cdf(C, newdata)) :
           what === :pdf      ? (newdata === nothing ? throw(ArgumentError("`newdata` required for `:pdf`")) : Distributions.pdf(C, newdata)) :
           throw(ArgumentError("`what` must be one of :simulate, :cdf, or :pdf. Got `$what`."))
end

###############################################################################
##### Automatic copula-family selection
###############################################################################

"""
    _available_selection_criteria()

Return the information criteria available for automatic copula selection.
The first entry is used when `criterion=:default`.
"""
_available_selection_criteria() = (:bic, :aic, :aicc, :hqc)

struct CopulaSelectionTable{T,R<:AbstractVector{T}} <: AbstractVector{T}
    rows::R
    criterion::Symbol
    selected_family
end

Base.IndexStyle(::Type{<:CopulaSelectionTable}) = IndexLinear()
Base.size(table::CopulaSelectionTable) = size(table.rows)
Base.getindex(table::CopulaSelectionTable, i::Int) = table.rows[i]
Base.sort(table::CopulaSelectionTable; kwargs...) =
    CopulaSelectionTable(sort(table.rows; kwargs...), table.criterion, table.selected_family)

"""
    _default_copula_candidates(d)

Return the built-in parametric copula families considered by automatic
selection in dimension `d` when `candidates=:default`.
"""
function _default_copula_candidates(d::Integer)
    common = (
        GaussianCopula,
        TCopula,
        AMHCopula,
        ClaytonCopula,
        FrankCopula,
        GumbelCopula,
        JoeCopula,
        GalambosCopula,
        HuslerReissCopula,
        LogCopula,
    )
    d == 2 && return (common..., PlackettCopula)
    return common
end

"""
    _all_copula_candidates(d)

Return the broad built-in parametric copula repertoire considered by automatic
selection when `candidates=:all`.
"""
function _all_copula_candidates(d::Integer)
    common = (
        IndependentCopula,
        GaussianCopula,
        TCopula,
        AMHCopula,
        ClaytonCopula,
        FrankCopula,
        GumbelCopula,
        GumbelBarnettCopula,
        InvGaussianCopula,
        JoeCopula,
        BB1Copula,
        BB2Copula,
        BB3Copula,
        BB6Copula,
        BB7Copula,
        BB8Copula,
        BB9Copula,
        BB10Copula,
        RafteryCopula,
        GalambosCopula,
        HuslerReissCopula,
        LogCopula,
        CuadrasAugeCopula,
        MixedCopula,
        MOCopula,
        TawnCopula,
    )
    d == 2 && return (
        common...,
        PlackettCopula,
        FGMCopula,
        BB4Copula,
        BB5Copula,
        AsymGalambosCopula,
        AsymLogCopula,
        AsymMixedCopula,
        BC2Copula,
    )
    return common
end

function _selection_candidates(candidates, d::Integer)
    candidates === :default && return _default_copula_candidates(d)
    candidates === :all && return _all_copula_candidates(d)
    return (candidates isa Type || candidates isa UnionAll) ? (candidates,) : Tuple(candidates)
end

"""
    selectiontable(model::CopulaModel)

Return the candidate-comparison table stored in an automatically selected
copula model.
"""
function selectiontable(M::CopulaModel)
    get(M.method_details, :selection, false) ||
        throw(ArgumentError("The model was not produced by automatic copula selection."))
    return CopulaSelectionTable(
        M.method_details.selection_table,
        M.method_details.criterion,
        M.method_details.selected_family,
    )
end

"""
    fit(CopulaModel, Copula, U; candidates=:default, criterion=:default, method=:default, kwargs...)

Fit candidate copula families to the `d x n` pseudo-observation matrix `U`,
select the family minimizing the requested information criterion, and return
the selected model.

Candidate fits used only for comparison are performed with `vcov=false`. The
winning family is fitted once with the inference options requested by the user.
"""
function Distributions.fit(::Type{CopulaModel}, ::Type{Copula}, U;
        candidates=:default, criterion::Symbol=:default, method::Symbol=:default,
        on_error::Symbol=:skip, require_convergence::Bool=true,
        quick_fit::Bool=false, derived_measures::Bool=true, vcov::Bool=true,
        vcov_method=nothing, kwargs...)
    d, _ = size(U)
    available_criteria = _available_selection_criteria()
    criterion = criterion === :default ? first(available_criteria) : criterion
    criterion in available_criteria ||
        throw(ArgumentError("Criterion '$criterion' is not available. Available: $(join(available_criteria, ", "))."))
    on_error in (:skip, :throw) || throw(ArgumentError("`on_error` must be either `:skip` or `:throw`."))

    candidate_types = _selection_candidates(candidates, d)

    rows = NamedTuple[]
    best_type = nothing
    best_method = nothing
    best_score = Inf
    best_index = 0
    selection_start = time()

    for CT in candidate_types
        CT <: Copula || throw(ArgumentError("Candidate `$CT` is not a subtype of `Copula`."))
        CT === Copula && throw(ArgumentError("`Copula` cannot itself appear inside `candidates`."))
        try
            M = Distributions.fit(CopulaModel, CT, U; method=method,
                quick_fit=false, derived_measures=false, vcov=false, kwargs...)
            aic_value = StatsBase.aic(M)
            aicc_value = aicc(M)
            bic_value = StatsBase.bic(M)
            hqc_value = hqc(M)
            score =
                criterion === :bic  ? bic_value :
                criterion === :aic  ? aic_value :
                criterion === :aicc ? aicc_value : hqc_value
            status =
                !isfinite(M.ll) || !isfinite(score) ? :nonfinite :
                require_convergence && !M.converged ? :not_converged :
                :ok
            if status === :nonfinite
                aic_value = isfinite(aic_value) ? aic_value : Inf
                aicc_value = isfinite(aicc_value) ? aicc_value : Inf
                bic_value = isfinite(bic_value) ? bic_value : Inf
                hqc_value = isfinite(hqc_value) ? hqc_value : Inf
                score = Inf
            end
            push!(rows, (candidate=CT, status=status, method=M.method,
                converged=M.converged, nparams=StatsBase.dof(M),
                loglikelihood=M.ll, aic=aic_value, aicc=aicc_value,
                bic=bic_value, hqc=hqc_value,
                criterion_value=score, elapsed_sec=M.elapsed_sec,
                error=nothing))
            if status === :ok && score < best_score
                best_type = CT
                best_method = M.method
                best_score = score
                best_index = length(rows)
            end
        catch err
            on_error === :throw && rethrow()
            push!(rows, (candidate=CT, status=:failed, method=method,
                converged=false, nparams=0, loglikelihood=NaN,
                aic=Inf, aicc=Inf, bic=Inf, hqc=Inf,
                criterion_value=Inf, elapsed_sec=NaN,
                error=sprint(showerror, err)))
        end
    end

    best_type === nothing && throw(ErrorException("No candidate copula produced an eligible finite fit."))
    selected = Distributions.fit(CopulaModel, best_type, U; method=best_method,
        quick_fit=quick_fit, derived_measures=derived_measures, vcov=vcov,
        vcov_method=vcov_method, kwargs...)
    quick_fit && return selected

    elapsed_sec = time() - selection_start
    return CopulaModel(selected.result, selected.n, selected.ll, selected.method;
        vcov=selected.vcov,
        converged=selected.converged,
        iterations=selected.iterations,
        elapsed_sec=elapsed_sec,
        method_details=(; selected.method_details...,
            selection=true,
            criterion=criterion,
            candidates=candidate_types,
            requested_method=method,
            selection_options=(; on_error, require_convergence, kwargs...),
            selection_table=rows,
            selected_family=best_type,
            selected_index=best_index,
            selected_score=best_score,
            selection_elapsed_sec=elapsed_sec))
end
