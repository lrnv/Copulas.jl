###############################################################################
#####  Fitting interface
#####  User-facing function:
#####   - `Distributions.fit(CopulaModel, MyCopulaType, data, method)`
#####   - `Distributions.fit(MyCopulaType, data, method)`
#####
#####  The fitting machinery below is package-internal.
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

This type stores only the fitted distribution, the data supplied to `fit`, its
cached fitted log-likelihood, and the minimal recipe needed to reproduce the
estimator. Transformed observations and optimizer diagnostics are deliberately
not retained.

Retrieve the fitted copula or Sklar distribution with
[`fitted_distribution`](@ref). `CopulaModel` also implements the documented
`StatsBase.StatisticalModel` interface, including `nobs`, `coef`, `coefnames`,
`deviance`, `nulldeviance`, `nullloglikelihood`, `aic`, `bic`, and `residuals`.
Use
[`selection_table`](@ref) for the candidate report produced by automatic family
selection.

The complete field layout and type-parameter order of `CopulaModel` are
implementation details.

See also `Distributions.fit`.

See also: [`fitted_distribution`](@ref), [`infer`](@ref), [`selection_table`](@ref),
[`GOFCopulaTest`](@ref),
[`StatsBase.residuals`](@ref).
"""
struct CopulaModel{CT,DT,RT} <: StatsBase.StatisticalModel
    result::CT
    data::DT
    loglikelihood::Float64
    recipe::RT
end

"""
    fitted_distribution(M::CopulaModel)

Return the fitted copula or `SklarDist` represented by `M`.

This is the supported way to retrieve the fitted distribution from a
`CopulaModel`; the model's concrete storage fields are not public API. The
returned distribution can be passed to the ordinary `Distributions.jl` and
Copulas.jl operations.

See also: [`CopulaModel`](@ref), [`Distributions.fit`](@ref),
[`StatsBase.residuals`](@ref).
"""
fitted_distribution(M::CopulaModel) = M.result

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
struct _CopulaFitSpec{T,K<:NamedTuple,F<:Tuple}
    target::T
    method::Symbol
    kwargs::K
    fixed::F
end

_CopulaFitSpec(target, method::Symbol, kwargs::NamedTuple) =
    _CopulaFitSpec(target, method, kwargs, ())

"""
    fitting_method(model::CopulaModel) -> Symbol

Return the effective estimator that produced `model`. The value is derived
from the model's reproducible fitting recipe.
"""
fitting_method(M::CopulaModel) = M.recipe.method

# Bootstrap/refit samples are already on the copula scale. If the original
# estimator accepted raw data through `pseudo_values=false`, replay the same
# estimator on the supplied pseudo-observations without ranking them again.
function _refit_kwargs(kwargs::NamedTuple)
    haskey(kwargs, :pseudo_values) || return kwargs
    return merge(kwargs, (; pseudo_values=true))
end

"""
    _refit(M::CopulaModel, data; replay_input=false)

Refit the same estimator specification that produced `M`. Copula models receive
pseudo-observations; Sklar models receive observations on their original scales
so that every marginal and the copula are re-estimated.

With `replay_input=true`, resampling inference replays the estimator from the
same input scale as the original call. The default is reserved for composite
GOF samples that are already pseudo-observations.

This is an internal inference hook. A model is refittable only when its fitting
entry point recorded a reproducible `_CopulaFitSpec`.

See also: [`_CopulaFitSpec`](@ref), [`_fit`](@ref), [`GOFCopulaTest`](@ref).
"""
function _refit(M::CopulaModel, U::AbstractMatrix; replay_input::Bool=false)
    spec = M.recipe
    spec isa _CopulaFitSpec || throw(ArgumentError(
        "this fitted model does not store a reproducible fitting specification; " *
        "composite goodness-of-fit refitting is unavailable for this model"))

    if spec.target isa NamedTuple && haskey(spec.target, :reparam)
        return Distributions.fit(CopulaModel, spec.target.reparam, spec.target.init, U; spec.kwargs...)
    end

    if spec.target isa Type && spec.target <: SklarDist
        return Distributions.fit(CopulaModel, spec.target, U; spec.kwargs...)
    end

    if replay_input && spec.target isa Type
        return Distributions.fit(CopulaModel, spec.target, U; method=spec.method, spec.kwargs...)
    end

    kwargs = _refit_kwargs(spec.kwargs)
    # Composite GOF refits receive pseudo-observations already. MPL uses the
    # same numerical likelihood engine as MLE, without ranking these again.
    method = spec.method === :mpl ? :mle : spec.method

    return Distributions.fit(CopulaModel, spec.target, U; method, kwargs...)
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
_construct_fitted_copula(CT, ::Val{d}, θ, example) where {d} = CT(d, θ...)
function _fit(CT::Type{<:Copula}, U, method::Val{:mle})
    return _fit(CT, U, Val(size(U, 1)), method)
end
function _fit(CT::Type{<:Copula}, U, ::Val{d}, ::Val{:mle}) where {d}
    example = _example(CT, d)
    cop(α) = _construct_fitted_copula(CT, Val(d), _rebound_params(CT, d, α), example)
    α₀  = _unbound_params(CT, d, Distributions.params(example))
    loss(C) = -Distributions.loglikelihood(C, U)
    res = try
        Optim.optimize(loss ∘ cop, α₀, Optim.LBFGS(); autodiff= ADTypes.AutoForwardDiff())
    catch err
        Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    end
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    Optim.converged(res) || throw(ErrorException("maximum-likelihood optimization did not converge"))
    return _construct_fitted_copula(CT, Val(d), θhat, example)
end

"""
    _fit(::Type{<:Copula}, U, ::Val{method}; kwargs...)

Internal entry point for fitting routines.

Each copula family implements `_fit` methods specialized on `Val{method}` and
returns the fitted copula. Temporary optimizer results and diagnostics stay
inside the estimator implementation.

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
    cop(α) = _construct_fitted_copula(CT, Val(d), _rebound_params(CT, d, α), example)
    α₀ = _unbound_params(CT, d, Distributions.params(example))
    @assert length(α₀) <= d*(d-1)÷2 "Cannot use $method since there are too much parameters."
    fun  = method isa Val{:itau} ? StatsBase.corkendall :
           method isa Val{:irho} ? StatsBase.corspearman : corblomqvist
    est  = fun(U')
    loss(C) = sum(abs2, est .- fun(C))
    res  = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    θhat = _rebound_params(CT, d, Optim.minimizer(res))
    Optim.converged(res) || throw(ErrorException("rank-matching optimization did not converge"))
    return _construct_fitted_copula(CT, Val(d), θhat, example)
end


"""
    Distributions.fit(CT::Type{<:Copula}, U; kwargs...) -> CT

Fit `CT` to the `d × n` matrix `U`, whose columns are observations, and return
only the fitted copula or Sklar distribution. This is the concise form of
`fit(CopulaModel, CT, U; kwargs...)`: it uses the same estimator and validation
without first constructing a `CopulaModel`. The same normalized estimator is
used by the model-returning form.

Use the `CopulaModel` form when diagnostics, information criteria, later
uncertainty quantification through `infer`, automatic selection, or composite
goodness-of-fit testing are needed.
"""
@inline Distributions.fit(T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(T, U; method=method, kwargs...)
@inline Distributions.fit(T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(T, U; copula_method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:Copula}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; method=method, kwargs...)
@inline Distributions.fit(::Type{CopulaModel}, T::Type{<:SklarDist}, U, method; kwargs...) = Distributions.fit(CopulaModel, T, U; copula_method=method, kwargs...)
@inline Distributions.fit(T::Type{<:Copula}, U; kwargs...) =
    _run_copula_estimator(T, U; kwargs...).result

"""
    _available_fitting_methods(::Type{<:Copula}, d::Int)

Return the tuple of fitting methods available for a given copula family in a given dimension.

This is used internally by [`Distributions.fit`](@ref) to check validity of the
`method` argument. Maximum pseudo-likelihood (`:mpl`) is handled by the
high-level public fitting entry point whenever `:mle` appears in this tuple;
extensions should continue to advertise and implement only their actual
`_fit(..., Val{method})` dispatches.

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

"""
    fitting_methods(::Type{<:Copula}, ::Val{d}) -> Tuple{Vararg{Symbol}}

Return the estimators supported by a copula fitting target in dimension `d`.
Return the fitting methods registered internally for a copula family. This is
an inspection interface; it does not make the private estimator dispatch an
extension API. Advertise `:mle`, not `:mpl`, for likelihood fitting: maximum
pseudo-likelihood is a high-level input-transformation contract and is
dispatched to the same `Val{:mle}` estimator after pseudo-observations have been
constructed.
"""
fitting_methods(CT::Type{<:Copula}, ::Val{d}) where {d} =
    _available_fitting_methods(CT, d)
fitting_methods(::Type{SklarDist}, ::Val{d}) where {d} =
    _available_fitting_methods(SklarDist, d)

function _reject_inference_fit_keywords(kwargs::NamedTuple)
    for keyword in (:vcov, :vcov_method)
        haskey(kwargs, keyword) && throw(ArgumentError(
            "`$keyword` is an inference option and is no longer accepted by fit; " *
            "fit a CopulaModel first, then call infer(model; method=...)"))
    end
    return nothing
end

function _default_fitting_method(CT, d)
    available = fitting_methods(CT, Val(d))
    isempty(available) && throw(ArgumentError("No fitting methods available for $CT."))
    return :mle in available ? :mle : first(available)
end

function _find_method(CT, d, method)
    avail = fitting_methods(CT, Val(d))
    isempty(avail) && throw(ArgumentError("No fitting methods available for $CT."))
    method === :default && return _default_fitting_method(CT, d)
    method ∉ avail && throw(ArgumentError(
        "Method '$method' not available for $CT. Available: $(join(avail, ", ")).",
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
        method=:default, pseudo_values=true, kwargs...)

Fit a copula of type `CT` by maximum likelihood or another supported estimator.

# Arguments
- `data::AbstractMatrix` — a `d×n` matrix with observations in columns.
- `pseudo_values::Bool` — whether `data` is already on the copula scale. Pass
  `false` to rank-transform raw observations with [`pseudos`](@ref).
- `method::Symbol` — fitting method. `:default` selects `:mle` whenever the
  family advertises it, otherwise the family's first advertised method. `:mpl`
  denotes maximum pseudo-likelihood and is never selected implicitly.
- `kwargs...`         — additional method-specific keyword arguments
  (e.g. `pseudo_values=true`, `grid=401` for extreme-value tails, etc.).

# Returns
A [`CopulaModel`](@ref) containing the fitted copula, the original data, the
fitted log-likelihood, and the minimal recipe needed to replay the estimator.

# Examples
```julia
U = rand(GumbelCopula(2, 3.0), 500)

M = fit(CopulaModel, GumbelCopula, U; method=:mle)
println(M)

# Quick fit: returns only the copula
C = fit(GumbelCopula, U; method=:itau)
```

Fitting performs estimation only. Apply [`infer`](@ref) to the returned model
afterwards when covariance estimates, standard errors, or confidence intervals
are required.

`method=:mle, pseudo_values=false` is normalized silently to `:mpl`, since the
rank transformation changes the statistical estimator. Conversely,
`method=:mpl, pseudo_values=true` is normalized to `:mle` with a warning because
no pseudo-observations are then constructed. Both use the same numerical
copula-likelihood optimizer; the distinction records the input's provenance.

See also: [`CopulaModel`](@ref), [`selection_table`](@ref),
[`GOFCopulaTest`](@ref).
"""
function _run_copula_estimator(CT::Type{<:Copula}, U;
        method=:default, pseudo_values::Union{Nothing,Bool}=nothing, kwargs...)
    _reject_inference_fit_keywords((; kwargs...))
    d = size(U, 1)
    requested_method = method === :default ? _default_fitting_method(CT, d) : method
    # MPL is a public preprocessing contract backed by the MLE
    # engine, not an internal `_fit(..., Val{:mpl})` extension hook.
    validation_method = requested_method === :mpl ? :mle : requested_method
    _find_method(CT, d, validation_method)
    likelihood_method = requested_method in (:mle, :mpl)
    input_is_pseudo = something(pseudo_values, true)
    method = likelihood_method ?
        _normalize_likelihood_fit(requested_method, input_is_pseudo) :
        requested_method
    fit_data = method === :mpl ? pseudos(U) : U
    engine_method = method === :mpl ? :mle : method
    engine_kwargs = !likelihood_method && pseudo_values !== nothing ?
        (; pseudo_values=input_is_pseudo, kwargs...) : (; kwargs...)
    C = _fit(CT, fit_data, Val{engine_method}(); engine_kwargs...)
    C isa Copula{d} || throw(ArgumentError(
        "the fitting implementation returned $(typeof(C)); expected a Copula{$d}"))
    return (; result=C, method, requested_method, input_is_pseudo,
            likelihood_method, fit_data, engine_kwargs)
end

function _estimate_copula(CT::Type{<:Copula}, U;
        method=:default, pseudo_values::Union{Nothing,Bool}=nothing, kwargs...)
    estimate = _run_copula_estimator(CT, U; method, pseudo_values, kwargs...)
    C = estimate.result
    (; method, input_is_pseudo, likelihood_method, fit_data, engine_kwargs) = estimate
    fit_kwargs = likelihood_method ?
        (; pseudo_values=input_is_pseudo, kwargs...) : engine_kwargs
    fit_spec = _CopulaFitSpec(CT, method, fit_kwargs)
    ll = Distributions.loglikelihood(C, fit_data)
    return CopulaModel(C, U, ll, fit_spec)
end

function Distributions.fit(::Type{CopulaModel}, CT::Type{<:Copula}, U;
        method=:default, pseudo_values::Union{Nothing,Bool}=nothing, kwargs...)
    return _estimate_copula(CT, U; method, pseudo_values, kwargs...)
end

_available_fitting_methods(::Type{SklarDist}, d) = (:ifm, :ecdf)
"""
    fit(CopulaModel, SklarDist{CT,TplMargins}, X;
        copula_method=:default, sklar_method=:ifm,
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
both Sklar routes use `copula_method=:mle` by default whenever `CT` supports
MLE, otherwise its first advertised fitting method. That copula step may be
replaced by another method supported by `CT`.

Moreover, calling `Distributions.fit` for a margin does not establish a generic
maximum-likelihood contract. The fitting algorithm is selected by each
distribution family and is not exposed here as a stable estimator protocol; it
may therefore differ between margins and need not be maximum likelihood. A
joint Sklar MLE cannot safely treat those independent calls as MLE building
blocks without a stronger upstream or Copulas.jl-specific interface.

The result is a `CopulaModel` whose `result` is the fitted `SklarDist`. Calling
[`infer`](@ref) with a resampling method repeats the complete sequential
estimator, including every marginal fit and the copula fit. Analytical
covariance is deliberately not inferred for the arbitrary marginal estimators
selected by `Distributions.fit`. Use `fit(SklarDist{...}, X; ...)` when only
the fitted distribution is required.

`SklarDist{CT,TplMargins}` is public here specifically as a fitting target:
`CT` selects the copula family and `TplMargins == Tuple{M₁,...,M_d}` selects the
marginal families. This exception does not expose arbitrary storage type
parameters or the concrete representation of constructed `SklarDist` values.
"""
function _estimate_sklar(T::Type{SklarDist{CT,TplMargins}}, X;
                         copula_method=:default, sklar_method=:ifm,
                         margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple(),
                         model::Bool=true) where {CT<:Copulas.Copula,TplMargins<:Tuple}

    # Get methods:
    d = size(X, 1)
    sklar_method  = _find_method(SklarDist, d, sklar_method)
    copula_method = copula_method === :default ?
        _default_fitting_method(CT, d) : _find_method(CT, d, copula_method)

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
        # Average ranks are the public default. Continuous-data inference still
        # requires users to assess whether ties represent rounding or discreteness.
        U .= pseudos(X)
    end

    # Fit the copula
    cop_estimate = _run_copula_estimator(CT, U; method=copula_method,
                                         copula_kwargs...)

    S = SklarDist(cop_estimate.result, m)
    model || return S
    ll = Distributions.loglikelihood(S, X)
    recipe = _CopulaFitSpec(T, cop_estimate.method,
        (; copula_method=cop_estimate.method, sklar_method,
           margins_kwargs, copula_kwargs))
    return CopulaModel(S, X, ll, recipe)
end

@inline Distributions.fit(T::Type{<:SklarDist}, X; kwargs...) =
    _estimate_sklar(T, X; model=false, kwargs...)

function Distributions.fit(::Type{CopulaModel}, T::Type{<:SklarDist}, X; kwargs...)
    return _estimate_sklar(T, X; kwargs...)
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
StatsBase.nobs(M::CopulaModel) = size(M.data, 2)

"""
    isfitted(M::CopulaModel) -> Bool

Return `true`: a `CopulaModel` exists only after its fitting procedure has
produced an accepted result.

See also: [`CopulaModel`](@ref), [`Distributions.fit`](@ref).
"""
StatsBase.isfitted(::CopulaModel)  = true
Distributions.loglikelihood(M::CopulaModel) = M.loglikelihood

"""
    deviance(M::CopulaModel) -> Float64

Return the deviance `-2ℓ`, where `ℓ` is the maximized log-likelihood stored in
the model. For non-likelihood estimators this summary reflects the likelihood
evaluated at the fitted parameters, not the objective that was optimized.

See also: [`StatsBase.nulldeviance`](@ref), [`StatsBase.aic`](@ref),
[`StatsBase.bic`](@ref).
"""
StatsBase.deviance(M::CopulaModel) = -2 * M.loglikelihood

"""
    dof(M::CopulaModel) -> Int

Return the number of free estimated parameters represented by `coef(M)`. For a
Sklar fit this includes both marginal and copula parameters. Fixed structural
choices and nonparametric components whose effective degrees of freedom are not
defined by the current interface are excluded.

See also: [`StatsBase.coef`](@ref), [`StatsBase.coefnames`](@ref),
[`StatsBase.aic`](@ref).
"""
StatsBase.dof(M::CopulaModel) = length(StatsBase.coef(M))

"""
    _copula_of(M::CopulaModel)

Return the copula contained in the fitted result, extracting it from a
`SklarDist` when margins were fitted jointly. This helper is internal; callers
that need the complete public fitted result should use
[`fitted_distribution`](@ref).
"""
_copula_of(M::CopulaModel)   = M.result isa SklarDist ? M.result.C : M.result

"""
    coef(M::CopulaModel) -> Vector{Float64}

Return the free estimated parameters as a flat vector in the same order as
`coefnames(M)`. For a Sklar fit, copula parameters precede the parameters of
each margin in coordinate order. Scalars are followed by vector entries and by
the strict upper triangle of matrix parameters. Models without a
finite-dimensional parameter record return an empty vector.

See also: [`StatsBase.coefnames`](@ref), [`StatsBase.vcov`](@ref),
[`StatsBase.confint`](@ref).
"""
StatsBase.coef(M::CopulaModel) = _coefficient_data(M)[2]

"""
coefnames(M::CopulaModel) -> Vector{String}

Return names for the flattened parameters in `coef(M)`, in matching order.
Indices are appended to vector and matrix parameter names so each coefficient
can be identified in covariance matrices and printed summaries.

See also: [`StatsBase.coef`](@ref), [`StatsBase.vcov`](@ref),
[`CopulaModel`](@ref).
"""
StatsBase.coefnames(M::CopulaModel) = _coefficient_data(M)[1]


# Flatten natural parameters after a perturbation of optimizer coordinates;
# those coordinates themselves are never exposed as model coefficients.
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

function _append_parameter!(nm, θ, value, name::String)
    if value isa Number
        push!(nm, name)
        push!(θ, value)
    elseif value isa AbstractMatrix
        @inbounds for j in 2:size(value, 2), i in 1:j-1
            push!(nm, maximum(size(value)) > 9 ? "$(name)_$(i)_$(j)" :
                  "$(name)$(Char(0x2080 + i))$(Char(0x2080 + j))")
            push!(θ, value[i, j])
        end
    elseif value isa AbstractVector
        for i in eachindex(value)
            push!(nm, length(value) > 9 ? "$(name)_$(i)" :
                  "$(name)$(Char(0x2080 + i))")
            push!(θ, value[i])
        end
    elseif value isa NamedTuple
        for (key, child) in pairs(value)
            child_name = isempty(name) ? String(key) : "$(name)_$(key)"
            _append_parameter!(nm, θ, child, child_name)
        end
    elseif value isa Tuple
        for (i, child) in pairs(value)
            _append_parameter!(nm, θ, child, "$(name)_$(i)")
        end
    elseif applicable(Distributions.params, value)
        _append_parameter!(nm, θ, Distributions.params(value), name)
    end
    return nothing
end

function _natural_parameters(D)
    nm = String[]
    θ = Any[]
    if D isa SklarDist
        !(hasmethod(StatsBase.dof, Tuple{typeof(D.C)}) && iszero(StatsBase.dof(D.C))) &&
            _append_parameter!(nm, θ, Distributions.params(D.C), "copula")
        for (i, margin) in pairs(D.m)
            _append_parameter!(nm, θ, Distributions.params(margin), "margin_$(i)")
        end
    elseif !(hasmethod(StatsBase.dof, Tuple{typeof(D)}) && iszero(StatsBase.dof(D)))
        _append_parameter!(nm, θ, Distributions.params(D), "")
    end
    values = isempty(θ) ? Float64[] : collect(promote(float.(θ)...))
    return nm, values
end

function _coefficient_data(M::CopulaModel)
    names, values = _natural_parameters(fitted_distribution(M))
    isempty(M.recipe.fixed) && return names, values
    fixed = string.(M.recipe.fixed)
    keep = map(name -> name ∉ fixed, names)
    return names[keep], values[keep]
end

function _parameter_blocks(M::CopulaModel)
    D = fitted_distribution(M)
    if !(D isa SklarDist)
        all = eachindex(StatsBase.coef(M))
        return (; copula=all, margins=())
    end
    names = StatsBase.coefnames(M)
    copula = findall(name -> startswith(name, "copula_"), names)
    margins = ntuple(i -> findall(name -> startswith(name, "margin_$(i)_"), names), length(D.m))
    return (; copula, margins)
end

function _copula_data(M::CopulaModel)
    D, data, spec = fitted_distribution(M), M.data, M.recipe
    if D isa SklarDist
        sklar_method = spec.kwargs.sklar_method
        sklar_method === :ecdf && return pseudos(data)
        T = promote_type(float(eltype(data)), mapreduce(eltype, promote_type, D.m))
        U = Matrix{T}(undef, size(data))
        lower, upper = nextfloat(zero(T)), prevfloat(one(T))
        @inbounds for j in axes(data, 2), i in axes(data, 1)
            U[i, j] = clamp(Distributions.cdf(D.m[i], data[i, j]), lower, upper)
        end
        return U
    end
    if spec.method === :mpl || get(spec.kwargs, :pseudo_values, true) === false
        return pseudos(data)
    end
    return data
end



"""
    aic(M::CopulaModel) -> Float64

Return Akaike's information criterion `2k - 2ℓ`, using `k = dof(M)` and the
log-likelihood stored in the model. Comparisons are meaningful only for models
fitted to the same observations and likelihood contribution.

See also: [`StatsBase.bic`](@ref), [`StatsBase.deviance`](@ref),
[`selection_table`](@ref).
"""
StatsBase.aic(M::CopulaModel) = 2*StatsBase.dof(M) - 2*M.loglikelihood

"""
    bic(M::CopulaModel) -> Float64

Return the Bayesian information criterion `k log(n) - 2ℓ`, using
`k = dof(M)` and `n = nobs(M)`. Comparisons are meaningful only for models
fitted to the same observations and likelihood contribution.

See also: [`StatsBase.aic`](@ref), [`StatsBase.deviance`](@ref),
[`selection_table`](@ref).
"""
StatsBase.bic(M::CopulaModel) = StatsBase.dof(M)*log(StatsBase.nobs(M)) - 2*M.loglikelihood
function aicc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    corr = (n > k + 1) ? (2k*(k+1)) / (n - k - 1) : Inf
    return StatsBase.aic(M) + corr
end
function hqc(M::CopulaModel)
    k, n = StatsBase.dof(M), StatsBase.nobs(M)
    return -2*M.loglikelihood + 2k*log(log(max(n, 3)))
end

"""
    nullloglikelihood(M::CopulaModel)

Compute the null-model log-likelihood lazily. Copula-only models use the
independence copula. Sklar models preserve the fitted margins and replace only
their copula by independence.

See also: [`StatsBase.nulldeviance`](@ref), [`StatsBase.deviance`](@ref),
[`CopulaModel`](@ref).
"""
function StatsBase.nullloglikelihood(M::CopulaModel)
    D = fitted_distribution(M)
    null = D isa SklarDist ?
        SklarDist(IndependentCopula(length(D)), D.m) :
        IndependentCopula(length(D))
    data = D isa SklarDist ? M.data : _copula_data(M)
    return Distributions.loglikelihood(null, data)
end
"""
    nulldeviance(M::CopulaModel)

Return `-2 * nullloglikelihood(M)`. The null likelihood is reconstructed lazily
and follows the same null-model conventions.

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
[`fitted_distribution`](@ref).
"""
StatsBase.residuals(M::CopulaModel; transform=:uniform) = begin
    transform in (:uniform, :normal) ||
        throw(ArgumentError("`transform` must be :uniform or :normal. Got `$transform`."))
    R = rosenblatt(_copula_of(M), _copula_data(M))
    return transform === :normal ? Distributions.quantile.(Distributions.Normal(), R) : R
end
###############################################################################
##### Automatic copula-family selection
###############################################################################

"""
    CopulaSelection

Result of comparing an explicit collection of copula families. It keeps the
winning [`CopulaModel`](@ref) and the candidate comparison separately, so model
selection state does not bloat ordinary fitted models.

Use [`selected_model`](@ref) to retrieve the winner and [`selection_table`](@ref)
to inspect all candidates. Concrete fields are implementation details.
"""
struct CopulaSelection{M,T}
    model::M
    table::T
    criterion::Symbol
    selected_index::Int
end

"""
    selected_model(result::CopulaSelection) -> CopulaModel

Return the winning fitted model from an automatic family-selection result.

See also: [`selection_table`](@ref), [`fitted_distribution`](@ref).
"""
selected_model(S::CopulaSelection) = S.model
fitted_distribution(S::CopulaSelection) = fitted_distribution(selected_model(S))
fitting_method(S::CopulaSelection) = fitting_method(selected_model(S))
Distributions.loglikelihood(S::CopulaSelection) =
    Distributions.loglikelihood(selected_model(S))
"""
    selection_table(result::CopulaSelection)

Return a copy of the candidate comparison rows recorded by automatic family
selection. Each row identifies a candidate, its status and effective method,
its likelihood and information criteria, or the error that prevented fitting.
Rows retain candidate order; changing the returned vector does not mutate the
selection result.

See also: [`CopulaSelection`](@ref), [`selected_model`](@ref),
[`StatsBase.aic`](@ref), [`StatsBase.bic`](@ref).
"""
selection_table(S::CopulaSelection) = copy(S.table)
selection_table(::CopulaModel) = throw(ArgumentError(
    "selection_table is available only for a CopulaSelection result"))

StatsBase.nobs(S::CopulaSelection) = StatsBase.nobs(selected_model(S))
StatsBase.isfitted(S::CopulaSelection) = StatsBase.isfitted(selected_model(S))
StatsBase.deviance(S::CopulaSelection) = StatsBase.deviance(selected_model(S))
StatsBase.dof(S::CopulaSelection) = StatsBase.dof(selected_model(S))
StatsBase.coef(S::CopulaSelection) = StatsBase.coef(selected_model(S))
StatsBase.coefnames(S::CopulaSelection) = StatsBase.coefnames(selected_model(S))
StatsBase.aic(S::CopulaSelection) = StatsBase.aic(selected_model(S))
StatsBase.bic(S::CopulaSelection) = StatsBase.bic(selected_model(S))
StatsBase.nullloglikelihood(S::CopulaSelection) =
    StatsBase.nullloglikelihood(selected_model(S))
StatsBase.nulldeviance(S::CopulaSelection) =
    StatsBase.nulldeviance(selected_model(S))
StatsBase.residuals(S::CopulaSelection; kwargs...) =
    StatsBase.residuals(selected_model(S); kwargs...)

"""
    fit(CopulaModel, Copula, U; candidates, criterion=:bic, method=:mle, kwargs...)

Fit an explicit collection of candidate families and return a
[`CopulaSelection`](@ref) selecting the smallest finite information criterion
(`:bic`, `:aic`, `:aicc`, or `:hqc`). The winning fit is reused without
performing inference. Prefer maximum likelihood fitting when interpreting
these as information criteria.

Failed candidates are recorded with `on_error=:skip`, or rethrown with
`on_error=:throw`. Interruptions always propagate. Composite GOF after selection
is not yet supported.
"""
function Distributions.fit(::Type{CopulaModel}, ::Type{Copula}, U;
        candidates, criterion::Symbol=:bic, method::Symbol=:mle,
        on_error::Symbol=:skip, kwargs...)
    _reject_inference_fit_keywords((; kwargs...))
    criterion in (:bic, :aic, :aicc, :hqc) ||
        throw(ArgumentError("Unknown selection criterion: $criterion"))
    on_error in (:skip, :throw) ||
        throw(ArgumentError("`on_error` must be :skip or :throw."))
    candidate_types = collect(candidates)
    isempty(candidate_types) && throw(ArgumentError("at least one candidate is required"))
    all(CT -> CT isa Type && CT <: Copula && CT !== Copula, candidate_types) ||
        throw(ArgumentError("candidates must be concrete copula families, not Copula itself"))
    unique!(candidate_types)

    rows = NamedTuple[]
    best = nothing
    best_index = 0
    best_score = Inf
    for CT in candidate_types
        evaluated = try
            M = Distributions.fit(CopulaModel, CT, U; method, kwargs...)
            criteria = (; aic=StatsBase.aic(M), aicc=aicc(M),
                bic=StatsBase.bic(M), hqc=hqc(M))
            (M, criteria, StatsBase.dof(M))
        catch err
            (err isa InterruptException || on_error === :throw) && rethrow()
            push!(rows, (candidate=CT, status=:failed, method=method,
                nparams=0, loglikelihood=NaN,
                aic=Inf, aicc=Inf, bic=Inf, hqc=Inf,
                error=sprint(showerror, err)))
            continue
        end
        M, criteria, nparams = evaluated
        score = getproperty(criteria, criterion)
        status = !isfinite(M.loglikelihood) || !isfinite(score) ? :nonfinite : :ok
        push!(rows, (; candidate=CT, status, method=fitting_method(M), nparams,
            loglikelihood=M.loglikelihood, criteria..., error=nothing))
        if status === :ok && score < best_score
            best, best_index, best_score = M, length(rows), score
        end
    end
    best === nothing && throw(ArgumentError("No candidate copula produced an eligible finite fit."))
    return CopulaSelection(best, rows, criterion, best_index)
end

Distributions.fit(::Type{Copula}, U; candidates, kwargs...) =
    fitted_distribution(Distributions.fit(CopulaModel, Copula, U;
                                          candidates, kwargs...))
