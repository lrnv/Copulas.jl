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
    fit(CopulaModel, family, data; kwargs...)

A fitted copula model.

This type stores the result of fitting a copula (or a Sklar distribution) to
pseudo-observations or raw data, together with auxiliary information useful
for statistical inference and model comparison.

Retrieve the fitted copula or Sklar distribution with
[`fitteddistribution`](@ref). `CopulaModel` also implements the documented
`StatsBase.StatisticalModel` interface, including `nobs`, `coef`, `coefnames`,
`vcov`, `stderror`, `confint`, `deviance`, `nulldeviance`,
`nullloglikelihood`, `aic`, `bic`, `predict`, and `residuals`. Use
[`selectiontable`](@ref) for the candidate report produced by automatic family
selection.

The displayed model may additionally report the estimator, convergence state,
iteration count, elapsed time, and method-specific diagnostics when available.
Those diagnostics and the concrete fields used to store them are not public
access interfaces. In particular, `method_details` is internal estimator
metadata and must not be consumed by user code. The complete field layout and
type-parameter order of `CopulaModel` are implementation details.

See also `Distributions.fit`.

See also: [`fitteddistribution`](@ref), [`selectiontable`](@ref),
[`GOFCopulaTest`](@ref),
[`StatsBase.predict`](@ref), [`StatsBase.residuals`](@ref).
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

"""
    fitteddistribution(M::CopulaModel)

Return the fitted copula or `SklarDist` represented by `M`.

This is the supported way to retrieve the fitted distribution from a
`CopulaModel`; the model's concrete storage fields are not public API. The
returned distribution can be passed to the ordinary `Distributions.jl` and
Copulas.jl operations.

See also: [`CopulaModel`](@ref), [`Distributions.fit`](@ref),
[`StatsBase.predict`](@ref), [`StatsBase.residuals`](@ref).
"""
fitteddistribution(M::CopulaModel) = M.result

"""
    _CopulaFitSpec(target, method, kwargs)

Internal, reproducible description of the estimator that produced a
`CopulaModel`. `target` is the family type or runtime structural template passed
to `fit`; `method` selects the estimator; `kwargs` contains only arguments that
change that estimator. Inference controls such as covariance computation are
excluded. Composite goodness-of-fit tests consume this record to replay exactly
the same estimator. Its fields and representation are not stable API.

See also: [`_fit`](@ref), [`_refit`](@ref), [`CopulaModel`](@ref),
[`GOFCopulaTest`](@ref).
"""
struct _CopulaFitSpec{T,K<:NamedTuple}
    target::T
    method::Symbol
    kwargs::K
end

# Bootstrap/refit samples are already on the copula scale. If the original
# estimator accepted raw data through `pseudo_values=false`, replay the same
# estimator on the supplied pseudo-observations without ranking them again.
function _refit_kwargs(kwargs::NamedTuple)
    haskey(kwargs, :pseudo_values) || return kwargs
    return merge(kwargs, (; pseudo_values=true))
end

"""
    _refit(M::CopulaModel, U)

Refit the same estimator specification that produced `M` to pseudo-observations
`U`.

This is an internal inference hook. A model is refittable only when its fitting
entry point recorded a reproducible `_CopulaFitSpec`.

See also: [`_CopulaFitSpec`](@ref), [`_fit`](@ref), [`GOFCopulaTest`](@ref).
"""
function _refit(M::CopulaModel, U::AbstractMatrix)
    spec = get(M.method_details, :_fit_spec, nothing)
    spec isa _CopulaFitSpec || throw(ArgumentError(
        "this fitted model does not store a reproducible fitting specification; " *
        "composite goodness-of-fit refitting is unavailable for this model"))

    kwargs = _refit_kwargs(spec.kwargs)
    # Composite GOF refits receive pseudo-observations already. MPL uses the
    # same numerical likelihood engine as MLE, without ranking these again.
    method = spec.method === :mpl ? :mle : spec.method

    return Distributions.fit(CopulaModel, spec.target, U; method, derived_measures=false, vcov=false, kwargs...,)
end

# Fallbacks that throw if the interface is not implemented correctly.
"""
    Distributions.params(C::Copula)
    Distributions.params(S::SklarDist)

Return the mathematical parameters of a copula or Sklar distribution as a
`NamedTuple`, in canonical constructor order. For an ordinary parametric
copula, splatting `values(params(C))` into its documented typed constructor
reconstructs the same model. Structural and empirical models document any
different reconstruction form explicitly.

Parameter names and values are public; concrete field names, storage-only type
parameters and caches are not. A new in-package family must specialize this
method before it can use generic fitting and display machinery.

See also: [`Copula`](@ref), [`SklarDist`](@ref), [`Distributions.fit`](@ref),
[`CopulaModel`](@ref).
"""
Distributions.params(C::Copula) = throw("You need to specify the Distributions.params() function as returning a named tuple with parameters.")

"""
    _example(CT, d)

Construct an interior representative of copula family `CT` in dimension `d`.
This internal fitting hook supplies parameter names, shapes, numeric types and
an initial point to generic optimization and covariance machinery. The example
must avoid limiting values and must be reconstructible by the family's fitting
protocol; it is not a user-facing default model.

See also: [`_unbound_params`](@ref), [`_rebound_params`](@ref),
[`_available_fitting_methods`](@ref), [`_fit`](@ref).
"""
_example(CT::Type{<:Copula}, d) = throw("You need to specify the `_example(CT::Type{T}, d)` function for your copula type, returning an example of the copula type in dimension d.")

"""
    _unbound_params(CT, d, θ)

Map the parameter `NamedTuple` `θ` of family `CT` to an unconstrained real
vector used by generic optimization and differentiation. This internal fitting
hook must be inverse-compatible with `_rebound_params`, preserve parameter
order, and map interior valid parameters to finite coordinates.

See also: [`_rebound_params`](@ref), [`_example`](@ref), [`_fit`](@ref).
"""
_unbound_params(CT::Type{Copula}, d, θ) = throw("You need to specify the _unbound_param method, that takes the namedtuple returned by `Distributions.params(CT(d, θ))` and trasform it into a raw vector living in R^p.")

"""
    _rebound_params(CT, d, α)

Map an unconstrained optimization vector `α` back to the valid parameter
`NamedTuple` expected by family `CT`. This internal fitting hook must enforce
the mathematical parameter domain, accept automatic-differentiation number
types, and invert `_unbound_params` on interior parameters.

See also: [`_unbound_params`](@ref), [`_example`](@ref), [`_fit`](@ref).
"""
_rebound_params(CT::Type{Copula}, d, α) = throw("You need to specify the _rebound_param method, that takes the output of _unbound_params and reconstruct the namedtuple that `Distributions.params(C)` would have returned.")
_fit_copula(CT, ::Val{d}, θ, example) where {d} = CT(d, θ...)
function _fit(CT::Type{<:Copula}, U, method::Val{:mle})
    return _fit(CT, U, Val(size(U, 1)), method)
end
function _fit(CT::Type{<:Copula}, U, ::Val{d}, ::Val{:mle}) where {d}
    example = _example(CT, d)
    cop(α) = _fit_copula(CT, Val(d), _rebound_params(CT, d, α), example)
    α₀  = _unbound_params(CT, d, Distributions.params(example))
    loss(C) = -Distributions.loglikelihood(C, U)
    res = try
        Optim.optimize(loss ∘ cop, α₀, Optim.LBFGS(); autodiff= ADTypes.AutoForwardDiff())
    catch err
        Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    end
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return _fit_copula(CT, Val(d), θhat, example), (; θ̂=θhat,
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

See also: [`_available_fitting_methods`](@ref), [`_example`](@ref),
[`_unbound_params`](@ref), [`_rebound_params`](@ref).
"""
function _fit(CT::Type{<:Copula}, U, method::Union{Val{:itau},Val{:irho},Val{:ibeta}})
    return _fit(CT, U, Val(size(U, 1)), method)
end
function _fit(CT::Type{<:Copula}, U, ::Val{d}, method::Union{Val{:itau},Val{:irho},Val{:ibeta}}) where {d}
    # generic rank-based routine (agnostic to vcov/inference)
    example = _example(CT, d)
    cop(α) = _fit_copula(CT, Val(d), _rebound_params(CT, d, α), example)
    α₀ = _unbound_params(CT, d, Distributions.params(example))
    @assert length(α₀) <= d*(d-1)÷2 "Cannot use $method since there are too much parameters."
    fun  = method isa Val{:itau} ? StatsBase.corkendall :
           method isa Val{:irho} ? StatsBase.corspearman : corblomqvist
    est  = fun(U')
    loss(C) = sum(abs2, est .- fun(C))
    res  = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    return _fit_copula(CT, Val(d), θhat, example), (; θ̂=θhat,
                optimizer  = Optim.summary(res),
                converged  = Optim.converged(res),
                iterations = Optim.iterations(res))
end


"""
    Distributions.fit(CT::Type{<:Copula}, U; kwargs...) -> CT

Fit `CT` to the `d × n` matrix `U`, whose columns are observations, and return
only the fitted copula or Sklar distribution. This is the concise form of
`fit(CopulaModel, CT, U; kwargs...)`: it uses the same estimator and validation
but discards inference metadata such as covariance estimates, convergence
details and the fitting sample.

Use the `CopulaModel` form when diagnostics, information criteria, uncertainty
quantification, automatic selection, or composite goodness-of-fit testing are
needed. Accepted keywords and the interpretation of `U` depend on the target
and fitting method.
"""
@inline Distributions.fit(T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(T, U; method=method, kwargs...)
@inline Distributions.fit(T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(T, U; copula_method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; copula_method=method, kwargs...)
@inline Distributions.fit(T::Type{<:Union{Copula, SklarDist}}, U; kwargs...) = Distributions.fit(CopulaModel, T, U; quick_fit=true, kwargs...).result

"""
    _available_fitting_methods(::Type{<:Copula}, d::Int)

Return the tuple of fitting methods available for a given copula family in a given dimension.

This is used internally by [`Distributions.fit`](@ref) to check validity of the
`method` argument. Maximum pseudo-likelihood (`:mpl`) is a public semantic alias
of the `:mle` engine and is therefore accepted whenever `:mle` appears in this
tuple, without requiring every family to duplicate the same implementation.

# Example
```julia
_available_fitting_methods(GumbelCopula, 3)
# → (:mle, :itau, :irho, :ibeta)
```

See also: [`_fit`](@ref), [`_example`](@ref),
[`Distributions.fit`](@ref).
"""
_available_fitting_methods(::Type{<:Copula}, d) = (:mle, :itau, :irho, :ibeta)
_available_fitting_methods(C::Copula, d) = _available_fitting_methods(typeof(C), d)

function _supports_fitting_method(CT, d, method)
    available = _available_fitting_methods(CT, d)
    return method in available || (method === :mpl && :mle in available)
end

function _find_method(CT, d, method)
    avail = _available_fitting_methods(CT, d)
    isempty(avail) && throw(ArgumentError("No fitting methods available for $CT."))
    method === :default && return avail[1]
    !_supports_fitting_method(CT, d, method) && throw(ArgumentError(
        "Method '$method' not available for $CT. Available: $(join(avail, ", "))." *
        (:mle in avail ? " Maximum pseudo-likelihood (:mpl) is also available." : ""),
    ))
    return method
end

function _normalize_likelihood_fit(method::Symbol, pseudo_values::Bool)
    if method === :mle && !pseudo_values
        return :mpl
    elseif method === :mpl && pseudo_values
        @warn "method=:mpl requires raw observations; because pseudo_values=true, fitting proceeds as method=:mle"
        return :mle
    end
    return method
end

"""
    fit(CopulaModel, CT::Type{<:Copula}, data;
        method=:mle, pseudo_values=true, kwargs...)

Fit a copula of type `CT` by maximum likelihood or another supported estimator.

# Arguments
- `data::AbstractMatrix` — a `d×n` matrix with observations in columns.
- `pseudo_values::Bool` — whether `data` is already on the copula scale. Pass
  `false` to rank-transform raw observations with [`pseudos`](@ref).
- `method::Symbol` — fitting method, defaulting to `:mle`. `:mpl` denotes
  maximum pseudo-likelihood.
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

Inference controls such as `vcov` and `derived_measures` affect the returned
model metadata, not the point estimator. Covariance availability depends on
the family, method and numerical regularity; `vcov=false` skips that work.

`method=:mle, pseudo_values=false` is normalized silently to `:mpl`, since the
rank transformation changes the statistical estimator. Conversely,
`method=:mpl, pseudo_values=true` is normalized to `:mle` with a warning because
no pseudo-observations are then constructed. Both use the same numerical
copula-likelihood optimizer; the distinction records the input's provenance.

See also: [`CopulaModel`](@ref), [`selectiontable`](@ref),
[`GOFCopulaTest`](@ref).
"""
function Distributions.fit(::Type{CopulaModel}, CT::Type{<:Copula}, U;
        method=:mle, pseudo_values::Union{Nothing,Bool}=nothing, quick_fit=false,
        derived_measures=true, vcov=true, vcov_method=nothing, kwargs...)
    _check_vcov_method(vcov_method)
    d, n = size(U)
    requested_method = method === :default ? :mle : method
    _find_method(CT, d, requested_method)
    likelihood_method = requested_method in (:mle, :mpl)
    input_is_pseudo = something(pseudo_values, true)
    method = likelihood_method ?
        _normalize_likelihood_fit(requested_method, input_is_pseudo) :
        requested_method
    fit_data = method === :mpl ? pseudos(U) : U
    engine_method = method === :mpl ? :mle : method
    engine_kwargs = !likelihood_method && pseudo_values !== nothing ?
        (; pseudo_values=input_is_pseudo, kwargs...) : (; kwargs...)
    fit_kwargs = likelihood_method ?
        (; pseudo_values=input_is_pseudo, kwargs...) : engine_kwargs
    fit_spec = _CopulaFitSpec(CT, method, fit_kwargs)
    t = @elapsed (rez = _fit(CT, fit_data, Val{engine_method}(); engine_kwargs...))
    C, meta = rez
    quick_fit && return (result=C,) # as soon as possible.
    ll = Distributions.loglikelihood(C, fit_data)
    meta = (; meta..., requested_method, pseudo_values=input_is_pseudo,
        fitting_data_pseudo_values=true)

    return _finish_copula_fit(CT, C, fit_data, ll, method, meta, t, fit_spec;
        derived_measures, vcov, vcov_method)
end

function _check_vcov_method(method)
    allowed = (:hessian, :godambe, :godambe_pairwise, :jackknife, :bootstrap)
    isnothing(method) || method in allowed ||
        throw(ArgumentError("unknown vcov method `$method`; expected one of $allowed"))
    return nothing
end

# Assemble inference around an existing fit, without rerunning its estimator.
function _finish_copula_fit(CT, C, U, ll, method, meta, t, fit_spec;
        derived_measures=true, vcov=true, vcov_method=nothing)
    d, n = size(U)

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

    md = (; d, n, method, meta..., null_ll=0.0, elapsed_sec=t, derived_measures, U=U, _fit_spec=fit_spec)

    return CopulaModel(C, n, ll, method;
        vcov         = get(md, :vcov, nothing),
        converged    = get(md, :converged, true),
        iterations   = get(md, :iterations, 0),
        elapsed_sec  = get(md, :elapsed_sec, NaN),
        method_details = md)
end

_available_fitting_methods(::Type{SklarDist}, d) = (:ifm, :ecdf)
"""
    fit(CopulaModel, SklarDist{CT,TplMargins}, X;
        copula_method=:mle, sklar_method=:ifm,
        margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple(), kwargs...)

Fit the margins and dependence structure of a Sklar distribution to a `d × n`
raw-data matrix `X`, with observations in columns. `TplMargins` supplies one
univariate distribution family per row and `CT` supplies the copula family.

With `sklar_method=:ifm`, each fitted marginal CDF transforms its row to the
uniform scale before the copula is fitted. With `:ecdf`, rank
pseudo-observations are used instead, although the requested parametric margins
are still fitted for the returned distribution. `margins_kwargs` are forwarded
to every marginal fit and `copula_kwargs` to the copula fit.

Both routes are sequential estimators, not joint maximum likelihood for the
complete Sklar distribution. Generic joint MLE is deliberately unavailable:
`Distributions.jl` margin families do not expose a common protocol mapping
their positive, bounded, ordered or interdependent parameters to an
unconstrained optimization vector. `params` and a constructor alone cannot
provide that information safely. The default is therefore `sklar_method=:ifm`;
both Sklar routes request `copula_method=:mle` by default, and that copula step
may be replaced by another method supported by `CT`.

The result is a `CopulaModel` whose `result` is the fitted `SklarDist` and whose
coefficient and covariance summaries combine the marginal and copula blocks.
Inference for a block may be unavailable when its estimator does not supply a
usable covariance estimate. Use `fit(SklarDist{...}, X; ...)` when only the
fitted distribution is required.

`SklarDist{CT,TplMargins}` is public here specifically as a fitting target:
`CT` selects the copula family and `TplMargins == Tuple{M₁,...,M_d}` selects the
marginal families. This exception does not expose arbitrary storage type
parameters or the concrete representation of constructed `SklarDist` values.
"""
function Distributions.fit(::Type{CopulaModel}, ::Type{SklarDist{CT,TplMargins}}, X; quick_fit = false,
                           copula_method = :mle, sklar_method = :ifm, margins_kwargs = NamedTuple(),
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

@inline function _vcov_copula(CT, ::Val{d}, α, example) where {d}
    return _fit_copula(CT, Val(d), _rebound_params(CT, d, α), example)
end

function _vcov_upper_triangle(A)
    return [
        A[idx]
        for idx in CartesianIndices(A)
        if idx[1] < idx[2]
    ]
end

_vcov_dependence_measure(::Val{:itau}) = τ
_vcov_dependence_measure(::Val{:irho}) = ρ
_vcov_dependence_measure(::Val{:ibeta}) = β
_vcov_dependence_measure(::Val) = λᵤ

_vcov_pairwise_measure(::Val{:itau}) = StatsBase.corkendall
_vcov_pairwise_measure(::Val{:irho}) = StatsBase.corspearman
_vcov_pairwise_measure(::Val{:ibeta}) = corblomqvist
_vcov_pairwise_measure(::Val) = coruppertail


function _vcov(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple;
    method::Symbol,
    override::Union{Symbol,Nothing}=nothing,
)
    _check_vcov_method(override)

    vcovm =
        !isnothing(override) ? override :
        method === :mle      ? :hessian :
        method === :itau     ? :godambe :
        method === :irho     ? :godambe :
        method === :ibeta    ? :godambe :
        method === :iupper   ? :godambe :
                               :jackknife

    # Compilation barrier: from this point onward the inference method and
    # fitting method are encoded in dispatch instead of runtime Symbol branches.
    return _vcov(
        CT,
        U,
        θ,
        Val(vcovm),
        Val(method),
    )
end


function _vcov(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple,
    vcovv::Val{:hessian},
    methodv::Val{method},
) where {method}
    return _vcov_hessian(CT, U, θ, Val(size(U, 1)), vcovv, methodv)
end

function _vcov_hessian(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple,
    ::Val{d},
    ::Val{:hessian},
    methodv::Val{method},
) where {d,method}
    α = _unbound_params(CT, d, θ)
    example = _example(CT, d)
    vd = Val(d)

    ℓ(αv) = Distributions.loglikelihood(
        _vcov_copula(CT, vd, αv, example),
        U,
    )

    H = ForwardDiff.hessian(ℓ, α)
    Iα = .-H

    if any(!isfinite, Iα)
        @warn "vcov(:hessian): non-finite Fisher information; falling back" Iα
        return _vcov(
            CT,
            U,
            θ,
            Val(:bootstrap),
            methodv,
        )
    end

    Iα = (Iα + Iα') / 2
    p = size(Iα, 1)
    I_p = Matrix{Float64}(LinearAlgebra.I, p, p)

    λ = 1e-8
    Vα = nothing

    @inbounds for _ in 1:8
        A = Iα + λ * I_p
        ch = LinearAlgebra.cholesky(
            LinearAlgebra.Symmetric(A);
            check=false,
        )

        if ch.info == 0
            Vα = ch \ I_p
            break
        end

        λ *= 10
    end

    if Vα === nothing || any(!isfinite, Vα)
        @warn "vcov(:hessian): failed to stabilize Fisher; falling back" λ_final=λ
        return _vcov(
            CT,
            U,
            θ,
            Val(:bootstrap),
            methodv,
        )
    end

    return _vcov_finalize(
        CT,
        U,
        θ,
        d,
        α,
        Vα,
        Val(:hessian),
        methodv,
    )
end


function _vcov(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple,
    ::Val{:godambe},
    methodv::Val{method},
) where {method}
    return _vcov_godambe(
        CT,
        U,
        θ,
        Val(false),
        Val(:godambe),
        methodv,
    )
end


function _vcov(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple,
    ::Val{:godambe_pairwise},
    methodv::Val{method},
) where {method}
    return _vcov_godambe(
        CT,
        U,
        θ,
        Val(true),
        Val(:godambe_pairwise),
        methodv,
    )
end

function _vcov_godambe(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple,
    pairwisev::Val{pairwise},
    vcovv::Val{vcovm},
    methodv::Val{method},
) where {pairwise,vcovm,method}
    return _vcov_godambe(
        CT,
        U,
        θ,
        Val(size(U, 1)),
        pairwisev,
        vcovv,
        methodv,
    )
end

function _vcov_godambe(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple,
    ::Val{d},
    ::Val{pairwise},
    vcovv::Val{vcovm},
    methodv::Val{method},
) where {d,pairwise,vcovm,method}
    n = size(U, 2)
    α = _unbound_params(CT, d, θ)

    example = _example(CT, d)
    vd = Val(d)

    φ = _vcov_dependence_measure(methodv)

    if pairwise
        pairwise_φ = _vcov_pairwise_measure(methodv)
        q = d * (d - 1) ÷ 2

        Dα = ForwardDiff.jacobian(
            αv -> _vcov_upper_triangle(
                pairwise_φ(_vcov_copula(CT, vd, αv, example)),
            ),
            α,
        )

        Dα = reshape(Dα, q, length(α))

        B = clamp(Int(floor(sqrt(n))), 10, 200)
        M = Matrix{Float64}(undef, B, q)
        idx = Vector{Int}(undef, n)
        rng = Random.default_rng()

        @inbounds for b in 1:B
            for i in 1:n
                idx[i] = rand(rng, 1:n)
            end

            Mb = @view U[:, idx]
            M[b, :] .= _vcov_upper_triangle(
                pairwise_φ(Mb'),
            )
        end

    else
        q = 1

        Dα = ForwardDiff.jacobian(
            αv -> [φ(_vcov_copula(CT, vd, αv, example))],
            α,
        )

        Dα = reshape(Dα, q, length(α))

        B = clamp(Int(floor(sqrt(n))), 10, 200)
        M = Matrix{Float64}(undef, B, q)
        idx = Vector{Int}(undef, n)
        rng = Random.default_rng()

        @inbounds for b in 1:B
            for i in 1:n
                idx[i] = rand(rng, 1:n)
            end

            Mb = @view U[:, idx]
            M[b, 1] = φ(Mb)
        end
    end

    Ω = n * Statistics.cov(M; corrected=true)

    DtD = Dα' * Dα
    ϵI = 1e-10LinearAlgebra.I

    stabilized = DtD + ϵI
    stabilized_inv = inv(stabilized)

    Vα =
        stabilized_inv *
        (Dα' * Ω * Dα) *
        stabilized_inv / n

    return _vcov_finalize(
        CT,
        U,
        θ,
        d,
        α,
        Vα,
        vcovv,
        methodv,
    )
end

function _vcov_finalize(
    CT::Type{<:Copula},
    U::AbstractMatrix,
    θ::NamedTuple,
    d::Int,
    α,
    Vα,
    ::Val{vcovm},
    methodv::Val{method},
) where {vcovm,method}
    J = ForwardDiff.jacobian(
        αv -> _flatten_params(
            _rebound_params(CT, d, αv),
        )[2],
        α,
    )

    Vθ = J * Vα * J'

    if !all(isfinite, Vθ)
        return _vcov(
            CT,
            U,
            θ,
            Val(:bootstrap),
            methodv,
        )
    end

    Vθ = (Vθ + Vθ') / 2

    λ, Q = LinearAlgebra.eigen(Matrix(Vθ))
    λ_reg = map(x -> max(x, 1e-12), λ)

    Vθ = LinearAlgebra.Symmetric(
        Q * LinearAlgebra.Diagonal(λ_reg) * Q',
    )

    if any(!isfinite, Matrix(Vθ))
        return _vcov(
            CT,
            U,
            θ,
            Val(:jackknife),
            methodv,
        )
    end

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

Observations are columns of the matrix supplied to `fit`. This value is the
sample size used by likelihood summaries and information criteria.

See also: [`CopulaModel`](@ref), [`StatsBase.dof`](@ref),
[`StatsBase.aic`](@ref), [`StatsBase.bic`](@ref).
"""
StatsBase.nobs(M::CopulaModel)     = M.n

"""
    isfitted(M::CopulaModel) -> Bool

Return `true`: a `CopulaModel` is created only after its fitting procedure has
produced a result. The displayed summary reports method-specific convergence
information when the estimator provides it.

See also: [`CopulaModel`](@ref), [`Distributions.fit`](@ref).
"""
StatsBase.isfitted(::CopulaModel)  = true

"""
    deviance(M::CopulaModel) -> Float64

Return the deviance `-2ℓ`, where `ℓ` is the maximized log-likelihood stored in
the model. For non-likelihood estimators this summary reflects the likelihood
evaluated at the fitted parameters, not the objective that was optimized.

See also: [`StatsBase.nulldeviance`](@ref), [`StatsBase.aic`](@ref),
[`StatsBase.bic`](@ref).
"""
StatsBase.deviance(M::CopulaModel) = -2 * M.ll

"""
    dof(M::CopulaModel) -> Int

Return the number of estimated copula parameters represented by `coef(M)`.
This excludes fixed structural choices and nonparametric components whose
effective degrees of freedom are not defined by the current interface.

See also: [`StatsBase.coef`](@ref), [`StatsBase.coefnames`](@ref),
[`StatsBase.aic`](@ref).
"""
StatsBase.dof(M::CopulaModel) = length(StatsBase.coef(M))

"""
    _copula_of(M::CopulaModel)

Return the copula contained in the fitted result, extracting it from a
`SklarDist` when margins were fitted jointly. This helper is internal; callers
that need the complete public fitted result should use
[`fitteddistribution`](@ref).
"""
_copula_of(M::CopulaModel)   = M.result isa SklarDist ? M.result.C : M.result

"""
    coef(M::CopulaModel) -> Vector{Float64}

Return the estimated copula parameters as a flat vector in the same order as
`coefnames(M)`. Scalars are followed by vector entries and by the strict upper
triangle of matrix parameters. Models without a finite-dimensional parameter
record return an empty vector.

See also: [`StatsBase.coefnames`](@ref), [`StatsBase.vcov`](@ref),
[`StatsBase.confint`](@ref).
"""
StatsBase.coef(M::CopulaModel) = haskey(M.method_details, :θ̂) ? _flatten_params(M.method_details.θ̂)[2] : Float64[]

"""
coefnames(M::CopulaModel) -> Vector{String}

Return names for the flattened parameters in `coef(M)`, in matching order.
Indices are appended to vector and matrix parameter names so each coefficient
can be identified in covariance matrices and printed summaries.

See also: [`StatsBase.coef`](@ref), [`StatsBase.vcov`](@ref),
[`CopulaModel`](@ref).
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

Return the estimated covariance matrix of `coef(M)`, or `nothing` when
covariance estimation was disabled or unavailable. Its rows and columns follow
`coefnames(M)`. The displayed model identifies the estimation method when that
information is available.

See also: [`StatsBase.stderror`](@ref), [`StatsBase.confint`](@ref),
[`StatsBase.coef`](@ref).
"""
StatsBase.vcov(M::CopulaModel) = M.vcov

"""
    stderror(M::CopulaModel)

Return coefficient standard errors computed from the diagonal of `vcov(M)`.
Return `nothing` when no covariance estimate is stored.

See also: [`StatsBase.vcov`](@ref), [`StatsBase.confint`](@ref),
[`StatsBase.coef`](@ref).
"""
function StatsBase.stderror(M::CopulaModel)
    V = StatsBase.vcov(M)
    V === nothing && return nothing
    return sqrt.(LinearAlgebra.diag(V))
end

"""
    confint(M::CopulaModel; level=0.95)

Return lower and upper vectors for pointwise Wald confidence intervals on the
fitted parameter scale. The intervals use a normal approximation and do not
enforce parameter constraints. Return `nothing` when `vcov(M)` is unavailable.

`level` must lie strictly between zero and one. These are marginal intervals;
they are neither simultaneous nor transformed to a family's constrained
parameter space.

See also: [`StatsBase.vcov`](@ref), [`StatsBase.stderror`](@ref),
[`StatsBase.coefnames`](@ref).
"""
function StatsBase.confint(M::CopulaModel; level::Real=0.95)
    V = StatsBase.vcov(M)
    V === nothing && return nothing
    z = Distributions.quantile(Distributions.Normal(), 1 - (1 - level)/2)
    θ = StatsBase.coef(M)
    se = sqrt.(LinearAlgebra.diag(V))
    return θ .- z .* se, θ .+ z .* se
end

"""
    aic(M::CopulaModel) -> Float64

Return Akaike's information criterion `2k - 2ℓ`, using `k = dof(M)` and the
log-likelihood stored in the model. Comparisons are meaningful only for models
fitted to the same observations and likelihood contribution.

See also: [`StatsBase.bic`](@ref), [`StatsBase.deviance`](@ref),
[`selectiontable`](@ref).
"""
StatsBase.aic(M::CopulaModel) = 2*StatsBase.dof(M) - 2*M.ll

"""
    bic(M::CopulaModel) -> Float64

Return the Bayesian information criterion `k log(n) - 2ℓ`, using
`k = dof(M)` and `n = nobs(M)`. Comparisons are meaningful only for models
fitted to the same observations and likelihood contribution.

See also: [`StatsBase.aic`](@ref), [`StatsBase.deviance`](@ref),
[`selectiontable`](@ref).
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

"""
    nullloglikelihood(M::CopulaModel)

Return the null-model log-likelihood recorded by the fitting procedure. This
quantity is available only for estimators that store `:null_ll` in their method
details; otherwise an `ArgumentError` is thrown. The precise null model is part
of that estimator's documented convention and should not be inferred solely
from the fitted family.

See also: [`StatsBase.nulldeviance`](@ref), [`StatsBase.deviance`](@ref),
[`CopulaModel`](@ref).
"""
function StatsBase.nullloglikelihood(M::CopulaModel)
    if hasproperty(M.method_details, :null_ll)
        return getfield(M.method_details, :null_ll)
    else
        throw(ArgumentError("nullloglikelihood is not available for this fitted model."))
    end
end
"""
    nulldeviance(M::CopulaModel)

Return `-2 * nullloglikelihood(M)`. The same availability and null-model
conventions apply, and an `ArgumentError` is propagated when no null
log-likelihood was recorded.

See also: [`StatsBase.nullloglikelihood`](@ref),
[`StatsBase.deviance`](@ref).
"""
StatsBase.nulldeviance(M::CopulaModel) = -2 * StatsBase.nullloglikelihood(M)
"""
    StatsBase.residuals(M::CopulaModel; transform=:uniform)

Compute Rosenblatt residuals of a fitted copula model.

# Arguments
- `transform = :uniform` → returns Rosenblatt residuals in [0,1].
- `transform = :normal`  → applies Φ⁻¹ to obtain pseudo-normal residuals.

# Notes
The residuals should be i.i.d. Uniform(0,1) under a correctly specified model.
Rows correspond to variables and columns to the observations stored by the fit.
Normal residuals can be infinite when a uniform residual is exactly zero or
one. For singular or atomic conditional laws, Rosenblatt residuals need not be
independent uniforms and should not be used as a continuous-model diagnostic.

See also: [`rosenblatt`](@ref), [`GOFCopulaTest`](@ref),
[`StatsBase.predict`](@ref).
"""
StatsBase.residuals(M::CopulaModel; transform=:uniform) = begin
    transform in (:uniform, :normal) ||
        throw(ArgumentError("`transform` must be :uniform or :normal. Got `$transform`."))
    haskey(M.method_details, :U) || throw(ArgumentError("the fitting observations are unavailable for residual computation"))
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
- `what=:cdf` or `:pdf` returns one value per column of `newdata`.
- `what=:simulate` returns a `d × nsim` sample, defaulting to the fitted sample
  size when `nsim <= 0`.

CDF and density evaluation uses the fitted copula component even when the model
result is a `SklarDist`; `newdata` is therefore always on the uniform copula
scale. Density prediction is meaningful only under the fitted copula's
documented measure semantics.

See also: [`CopulaModel`](@ref), [`StatsBase.residuals`](@ref),
[`Distributions.cdf`](@extref Distributions Distributions.cdf) and
[Distributions `pdf`](@extref Distributions Probability-evaluation).
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
    selectiontable(model::CopulaModel)

Return the candidate comparison rows recorded by automatic family selection.

Each row identifies a candidate, whether fitting succeeded, the selected
criterion value when finite, and diagnostic information for skipped failures.
Rows describe only the candidates that were actually considered and retain the
selection order, which makes the table suitable for explaining why the winning
model was chosen.

The returned vector is a copy, so changing its membership does not mutate the
fitted model. An `ArgumentError` is thrown when `model` was not produced by the
automatic-selection form of `fit`.

See also: [`CopulaModel`](@ref), [`Distributions.fit`](@ref),
[`StatsBase.aic`](@ref), [`StatsBase.bic`](@ref).
"""
function selectiontable(M::CopulaModel)
    haskey(M.method_details, :selection_table) ||
        throw(ArgumentError("The model was not produced by automatic copula selection."))
    return copy(M.method_details.selection_table)
end

"""
    fit(CopulaModel, Copula, U; candidates, criterion=:bic, method=:mle, kwargs...)

Fit an explicit collection of candidate families and select the smallest finite
information criterion (`:bic`, `:aic`, `:aicc`, or `:hqc`). The winning fit is
reused; only its requested inference is computed afterwards. Prefer maximum
likelihood fitting when interpreting these as information criteria.

Failed candidates are recorded with `on_error=:skip`, or rethrown with
`on_error=:throw`. Interruptions always propagate. Composite GOF after selection
is not yet supported.
"""
function Distributions.fit(::Type{CopulaModel}, ::Type{Copula}, U;
        candidates, criterion::Symbol=:bic, method::Symbol=:mle,
        on_error::Symbol=:skip, require_convergence::Bool=true,
        quick_fit::Bool=false, derived_measures::Bool=true, vcov::Bool=true,
        vcov_method=nothing, kwargs...)
    criterion in (:bic, :aic, :aicc, :hqc) ||
        throw(ArgumentError("Unknown selection criterion: $criterion"))
    on_error in (:skip, :throw) ||
        throw(ArgumentError("`on_error` must be :skip or :throw."))
    _check_vcov_method(vcov_method)
    candidate_types = collect(candidates)
    isempty(candidate_types) && throw(ArgumentError("at least one candidate is required"))
    all(CT -> CT isa Type && CT <: Copula && CT !== Copula, candidate_types) ||
        throw(ArgumentError("candidates must be concrete copula families, not Copula itself"))
    unique!(candidate_types)

    rows = NamedTuple[]
    best = nothing
    best_index = 0
    best_score = Inf
    started = time()
    for CT in candidate_types
        evaluated = try
            M = Distributions.fit(CopulaModel, CT, U; method,
                derived_measures=false, vcov=false, kwargs...)
            criteria = (; aic=StatsBase.aic(M), aicc=aicc(M),
                bic=StatsBase.bic(M), hqc=hqc(M))
            (M, criteria, StatsBase.dof(M))
        catch err
            (err isa InterruptException || on_error === :throw) && rethrow()
            push!(rows, (candidate=CT, status=:failed, method=method,
                converged=false, nparams=0, loglikelihood=NaN,
                aic=Inf, aicc=Inf, bic=Inf, hqc=Inf,
                error=sprint(showerror, err)))
            continue
        end
        M, criteria, nparams = evaluated
        score = getproperty(criteria, criterion)
        status = !isfinite(M.ll) || !isfinite(score) ? :nonfinite :
            require_convergence && !M.converged ? :not_converged : :ok
        push!(rows, (; candidate=CT, status, method=M.method,
            converged=M.converged, nparams,
            loglikelihood=M.ll, criteria..., error=nothing))
        if status === :ok && score < best_score
            best, best_index, best_score = M, length(rows), score
        end
    end
    best === nothing && throw(ArgumentError("No candidate copula produced an eligible finite fit."))
    quick_fit && return (result=best.result,)

    CT = rows[best_index].candidate
    selected = _finish_copula_fit(CT, best.result, best.method_details.U,
        best.ll, best.method,
        (; best.method_details..., converged=best.converged, iterations=best.iterations),
        best.elapsed_sec, nothing;
        derived_measures, vcov, vcov_method)
    # No single-family fit specification can reproduce model selection. Until
    # selection-aware bootstrap exists, _refit must reject this model.
    return CopulaModel(selected.result, selected.n, selected.ll, selected.method;
        vcov=selected.vcov, converged=selected.converged,
        iterations=selected.iterations, elapsed_sec=time() - started,
        method_details=(; selected.method_details..., criterion,
            selection_table=rows, selected_index=best_index))
end
