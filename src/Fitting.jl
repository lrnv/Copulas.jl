###############################################################################
#####  Fitting interface
#####  User-facing function:
#####   - `Distributions.fit(CopulaModel, MyCopulaType, data, method)`
#####   - `Distributions.fit(MyCopulaType, data, method)`
#####
#####  The fitting machinery below is package-internal. Simple parametric
#####  families opt into the generic routines by defining `Paramorph.param_space`
#####  and a canonical `CT(d, parameters...)` constructor.
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
struct _CopulaFitSpec{T,K<:NamedTuple}
    target::T
    method::Symbol
    kwargs::K
end

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

# A resample of a weighted fit is a sample of the observations in which
# observation j is repeated weights[j] times, so it is refitted unweighted.
_unweighted_kwargs(kwargs::NamedTuple) = Base.structdiff(kwargs, NamedTuple{(:weights,)})

"""
    _refit(M::CopulaModel, data; replay_input=false)

Refit the same estimator specification that produced `M`. Copula models receive
pseudo-observations; Sklar models receive observations on their original scales
so that every marginal and the copula are re-estimated.

With `replay_input=true`, resampling inference replays the estimator from the
same input scale as the original call. The default is reserved for composite
GOF samples that are already pseudo-observations.

A resample of a weighted fit is refitted without the weights: it is a sample
of the observations in which observation `j` is repeated `weights[j]` times,
and the resampling procedures draw it as such.

This is an internal inference hook. A model is refittable only when its fitting
entry point recorded a reproducible `_CopulaFitSpec`.

See also: [`_CopulaFitSpec`](@ref), [`_fit`](@ref), [`GOFCopulaTest`](@ref).
"""
function _refit(M::CopulaModel, U::AbstractMatrix; replay_input::Bool=false)
    spec = M.recipe
    spec isa _CopulaFitSpec || throw(ArgumentError(
        "this fitted model does not store a reproducible fitting specification; " *
        "composite goodness-of-fit refitting is unavailable for this model"))
    spec_kwargs = _unweighted_kwargs(spec.kwargs)

    if spec.target isa NamedTuple && haskey(spec.target, :reparam)
        return Distributions.fit(CopulaModel, spec.target.reparam, spec.target.init, U; spec_kwargs...)
    end

    if spec.target isa Type && spec.target <: SklarDist
        return Distributions.fit(CopulaModel, spec.target, U; spec_kwargs...)
    end

    if replay_input && spec.target isa Type
        return Distributions.fit(CopulaModel, spec.target, U; method=spec.method, spec_kwargs...)
    end

    kwargs = _refit_kwargs(spec_kwargs)
    # Composite GOF refits receive pseudo-observations already. MPL uses the
    # same numerical likelihood engine as MLE, without ranking these again.
    method = spec.method === :mpl ? :mle : spec.method

    return Distributions.fit(CopulaModel, spec.target, U; method, kwargs...)
end

"""
    Distributions.params(C::Copula)
    Distributions.params(S::SklarDist)

Return the mathematical parameters of a copula or Sklar distribution as a
`Tuple`, in canonical constructor order, following the `Distributions.jl`
convention. Parameter names and constraints are supplied independently by
`Paramorph.param_space`; `params` contains values only. For an ordinary
parametric copula, splatting `params(C)` into its documented typed constructor
reconstructs the same model.

See also: [`Copula`](@ref), [`SklarDist`](@ref), [`Distributions.fit`](@ref),
[`CopulaModel`](@ref).
"""


"""
    _fit_weights(weights, n) -> Union{Nothing, Vector}

Validate the observation weights passed to `fit` and normalize them so that
they sum to `n`, the number of observations. `nothing` is returned untouched
and selects the unweighted code path.

A weight is then read as "how many observations this column counts for": the
fitted parameters are invariant to the scale of the weights, uniform weights
reproduce the unweighted fit exactly, and `nobs` keeps the sample size that
the information criteria use. A zero weight counts its column zero times;
[`_weighted_sample`](@ref) drops such a column before any likelihood sees it.

See also: [`_weighted_sample`](@ref), [`_weighted_loglikelihood`](@ref),
[`pseudos`](@ref), [`Distributions.fit`](@ref).
"""
_fit_weights(::Nothing, ::Int) = nothing
function _fit_weights(weights::AbstractVector{<:Real}, n::Int)
    length(weights) == n || throw(DimensionMismatch(
        "weights must have one entry per observation; got $(length(weights)) " *
        "weights for $n observations"))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    all(w -> w >= 0, weights) || throw(ArgumentError("weights must be non-negative"))
    total = sum(weights)
    total > 0 || throw(ArgumentError("weights must not all be zero"))
    return weights .* (n / total)
end
_fit_weights(weights, ::Int) = throw(ArgumentError(
    "weights must be nothing or a vector of non-negative reals; got $(typeof(weights))"))

"""
    _weighted_sample(X, weights) -> (X, weights)

The columns of `X` that carry weight, with their weights. A zero weight removes
its observation, so the pair is what a likelihood engine is given: a removed
observation may sit on the boundary of the unit hypercube, where its score is
infinite and `0 * Inf` would poison a weighted cross-product. Without a zero
weight, and with `weights === nothing`, the inputs are returned untouched.

See also: [`_fit_weights`](@ref), [`_weighted_loglikelihood`](@ref).
"""
_weighted_sample(X::AbstractMatrix, ::Nothing) = (X, nothing)
function _weighted_sample(X::AbstractMatrix, weights::AbstractVector)
    any(iszero, weights) || return (X, weights)
    kept = findall(!iszero, weights)
    return (X[:, kept], weights[kept])
end

"""
    _weighted_loglikelihood(D, X, weights)

Log-likelihood of `D` on the columns of `X`, each column multiplied by its
weight. With `weights === nothing` this is `Distributions.loglikelihood(D, X)`.
The weighted sum runs over the same column views in the same order as the
unweighted reduction, so unit weights reproduce it bit for bit. A zero-weight
column is dropped by [`_weighted_sample`](@ref) rather than summed.

See also: [`_fit_weights`](@ref), [`StatsBase.nobs`](@ref).
"""
_weighted_loglikelihood(D, X, ::Nothing) = Distributions.loglikelihood(D, X)
function _weighted_loglikelihood(D, X::AbstractMatrix, weights::AbstractVector)
    X, weights = _weighted_sample(X, weights)
    return sum(j -> weights[j] * Distributions.logpdf(D, view(X, :, j)), axes(X, 2))
end

# Weights recorded by the estimator that produced a model, or `nothing`.
_model_weights(M::CopulaModel) =
    M.recipe isa _CopulaFitSpec ? get(M.recipe.kwargs, :weights, nothing) : nothing

# Average-rank pseudo-observations for weights that `_fit_weights` has already
# normalized, so that the fit and the model's `_copula_data` rank by exactly
# the same weights instead of normalizing them a second time.
_normalized_pseudos(X::AbstractMatrix, weights) =
    _pseudos(X, Val(:average), Random.default_rng(), weights)

_parameter_arguments(η::Tuple) = η
_parameter_arguments(η) = (η,)
_parameter_space_copula(CT, d, p, α) =
    CT(d, _parameter_arguments(Paramorph.constrain(p, α))...)

function _fit(CT::Type{<:Copula}, U, method::Val{:mle}; kwargs...)
    return _fit(CT, U, Val(size(U, 1)), method; kwargs...)
end
function _fit(CT::Type{<:Copula}, U, ::Val{d}, ::Val{:mle}; weights=nothing) where {d}
    p = Paramorph.param_space(CT, d)
    α₀ = zeros(Paramorph.dimension(p))
    cop(α) = _parameter_space_copula(CT, d, p, α)
    loss(C) = -_weighted_loglikelihood(C, U, weights)
    res = Optim.optimize(loss ∘ cop, α₀, Optim.LBFGS();
                         autodiff=ADTypes.AutoForwardDiff())
    return cop(Optim.minimizer(res))
end

"""
    _fit(::Type{<:Copula}, U, ::Val{method}; kwargs...)

Internal entry point for fitting routines.

Each copula family implements `_fit` methods specialized on `Val{method}` and
returns the fitted copula. Temporary optimizer results and diagnostics stay
inside the estimator implementation.

Simple parametric families can use the generic implementations by defining
`Paramorph.param_space(CT, d)` and a canonical `CT(d, parameters...)`
constructor. This is not intended for direct use by end-users; use
[`Distributions.fit(CopulaModel, ...)`] instead.

See also: [`_available_fitting_methods`](@ref), [`Distributions.fit`](@ref).
"""
function _fit(CT::Type{<:Copula}, U, method::Union{Val{:itau},Val{:irho},Val{:ibeta}}; weights=nothing)
    return _fit(CT, U, Val(size(U, 1)), method; weights)
end
function _fit(CT::Type{<:Copula}, U, ::Val{d}, method::Union{Val{:itau},Val{:irho},Val{:ibeta}}; weights=nothing) where {d}
    p = Paramorph.param_space(CT, d)
    α₀ = zeros(Paramorph.dimension(p))
    length(α₀) <= d*(d-1)÷2 || throw(ArgumentError(
        "cannot use $method in dimension $d with $(length(α₀)) free parameters; " *
        "only $(d*(d-1)÷2) pairwise rank constraints are available"))
    cop(α) = _parameter_space_copula(CT, d, p, α)
    fun = method isa Val{:itau} ? StatsBase.corkendall :
          method isa Val{:irho} ? StatsBase.corspearman : corblomqvist
    est = _rank_measure(method, U, weights)
    loss(C) = sum(abs2, est .- fun(C))
    res = Optim.optimize(loss ∘ cop, α₀, Optim.NelderMead())
    return cop(Optim.minimizer(res))
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

See also: [`_fit`](@ref), [`Distributions.fit`](@ref).
"""
_available_fitting_methods(::Type{<:Copula}, d) = (:mle, :itau, :irho, :ibeta)
_available_fitting_methods(C::Copula, d) = _available_fitting_methods(typeof(C), d)

function _reject_inference_fit_keywords(kwargs::NamedTuple)
    for keyword in (:vcov, :vcov_method)
        haskey(kwargs, keyword) && throw(ArgumentError(
            "`$keyword` is an inference option and is no longer accepted by fit; " *
            "fit a CopulaModel first, then call infer(model; method=...)"))
    end
    return nothing
end

function _default_fitting_method(CT, d)
    available = _available_fitting_methods(CT, d)
    isempty(available) && throw(ArgumentError("No fitting methods available for $CT."))
    return :mle in available ? :mle : first(available)
end

function _find_method(CT, d, method)
    avail = _available_fitting_methods(CT, d)
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
- `weights` — optional vector of one non-negative, finite weight per
  observation, not all zero. See *Weighted observations* below.
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

# Weighted observations

`weights=w` fits a weighted pseudo-likelihood: the log-likelihood contribution
of column `j` is multiplied by `w[j]`. The weights are normalized once so that
they sum to the number of observations `n`, so a weight reads as "how many
observations this column counts for", the fitted parameters are invariant to
the scale of `w`, and uniform weights reproduce the unweighted fit exactly. A
zero weight removes its observation: its column is dropped before the
likelihood is evaluated, so it may lie on the boundary of the unit hypercube
where the density is not defined. The stored
`loglikelihood`, and hence `aic`, `bic` and `deviance`, are the weighted ones;
`nobs` stays `n`. With `pseudo_values=false` the rank transformation is the
weighted one of [`pseudos`](@ref).

Weights are accepted by the likelihood estimators `:mle` and `:mpl` and by the
rank inversions `:itau`, `:irho`, `:ibeta` and `:itau_irho`, which invert the
weighted sample measure: Kendall's tau-b, Spearman's rho and Blomqvist's beta
of the sample in which observation `j` is repeated `weights[j]` times. Every
family shipped with one of these methods takes them; a family whose own method
has no keyword arguments is fitted by the generic driver instead, as it is for
any keyword. The tail estimator `:iupper` and the nonparametric estimators
refuse them. The Sklar route `fit(SklarDist{...}, X; weights)` takes them too.
[`infer`](@ref) reads the same weights; composite goodness-of-fit tests refuse
a weighted model.

See also: [`CopulaModel`](@ref), [`selection_table`](@ref),
[`GOFCopulaTest`](@ref).
"""
# The rank inversions whose sample measure has a weighted form.
const _WEIGHTED_RANK_METHODS = (:itau, :irho, :ibeta, :itau_irho)

function _run_copula_estimator(CT::Type{<:Copula}, U;
        method=:default, pseudo_values::Union{Nothing,Bool}=nothing,
        weights=nothing, kwargs...)
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
    weights = _fit_weights(weights, size(U, 2))
    weights === nothing || likelihood_method ||
        requested_method in _WEIGHTED_RANK_METHODS || throw(ArgumentError(
        "`weights` are supported by the likelihood estimators :mle and :mpl and by " *
        "the rank inversions $(_WEIGHTED_RANK_METHODS) only; method=$requested_method " *
        "takes none"))
    fit_data = method === :mpl ? _normalized_pseudos(U, weights) : U
    engine_method = method === :mpl ? :mle : method
    engine_kwargs = !likelihood_method && pseudo_values !== nothing ?
        (; pseudo_values=input_is_pseudo, kwargs...) : (; kwargs...)
    # The engine sees the weighted columns only; the model keeps every column.
    engine_data, engine_weights = _weighted_sample(fit_data, weights)
    weights === nothing || (engine_kwargs = (; weights=engine_weights, engine_kwargs...))
    C = _fit(CT, engine_data, Val{engine_method}(); engine_kwargs...)
    C isa Copula{d} || throw(ArgumentError(
        "the fitting implementation returned $(typeof(C)); expected a Copula{$d}"))
    return (; result=C, method, requested_method, input_is_pseudo,
            likelihood_method, fit_data, engine_kwargs, weights)
end

function _estimate_copula(CT::Type{<:Copula}, U;
        method=:default, pseudo_values::Union{Nothing,Bool}=nothing,
        weights=nothing, kwargs...)
    estimate = _run_copula_estimator(CT, U; method, pseudo_values, weights, kwargs...)
    C = estimate.result
    (; method, input_is_pseudo, likelihood_method, fit_data, engine_kwargs) = estimate
    fit_kwargs = likelihood_method ?
        (; pseudo_values=input_is_pseudo, kwargs...) : engine_kwargs
    # The normalized weights are part of the estimator, so the recipe carries them.
    estimate.weights === nothing ||
        (fit_kwargs = (; fit_kwargs..., weights=estimate.weights))
    fit_spec = _CopulaFitSpec(CT, method, fit_kwargs)
    ll = _weighted_loglikelihood(C, fit_data, estimate.weights)
    return CopulaModel(C, U, ll, fit_spec)
end

function Distributions.fit(::Type{CopulaModel}, CT::Type{<:Copula}, U;
        method=:default, pseudo_values::Union{Nothing,Bool}=nothing,
        weights=nothing, kwargs...)
    return _estimate_copula(CT, U; method, pseudo_values, weights, kwargs...)
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

`weights` gives one non-negative, finite weight per observation, not all zero,
normalized to sum to `n` as in the copula-only `fit`, and every step reads the
same vector: margin `i` is fitted by `Distributions.fit(Mᵢ, xᵢ, w)`, which is
the weighted maximum-likelihood fit `fit_mle(Mᵢ, xᵢ, w)` that Distributions.jl
defines for the families with weighted sufficient statistics, and a margin
family without one is refused by name, and a zero-weight observation is
dropped before the margin sees it, as [`_weighted_sample`](@ref) drops it
before the copula engine; `:ecdf` ranks by weighted mass as
[`pseudos`](@ref) does; the copula is fitted with the same `weights`; the
stored log-likelihood is the weighted one. Unit weights reproduce the
unweighted margins up to rounding, since Distributions.jl reduces its weighted
sufficient statistics in another order, and integer weights summing to `n`
reproduce the fit of the sample in which each observation is repeated that
many times. `weights` is a keyword of `fit` itself, not of `copula_kwargs`.
"""
function _estimate_sklar(T::Type{SklarDist{CT,TplMargins}}, X;
                         copula_method=:default, sklar_method=:ifm,
                         margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple(),
                         weights=nothing,
                         model::Bool=true) where {CT<:Copulas.Copula,TplMargins<:Tuple}

    # Get methods:
    d = size(X, 1)
    haskey(copula_kwargs, :weights) && throw(ArgumentError(
        "pass `weights` to `fit` itself rather than through `copula_kwargs`: the " *
        "Sklar route fits its margins, ranks its observations and fits its copula " *
        "by the same weights"))
    # The copula estimator normalizes the same raw weights itself, so that the
    # margins, the ranks, the copula and the recipe all carry one vector.
    w = _fit_weights(weights, size(X, 2))
    sklar_method  = _find_method(SklarDist, d, sklar_method)
    copula_method = copula_method === :default ?
        _default_fitting_method(CT, d) : _find_method(CT, d, copula_method)

    # Fit marginals. A zero-weight observation is dropped before a margin
    # sees it, as it is before the copula engine: it may lie outside the
    # margin's support, where its weighted sufficient statistic is `0 * -Inf`.
    Xm, wm = _weighted_sample(X, w)
    m = ntuple(i -> _fit_margin(TplMargins.parameters[i], (@view Xm[i, :]), wm; margins_kwargs...), d)

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
        U .= _normalized_pseudos(X, w)
    end

    # Fit the copula
    cop_estimate = _run_copula_estimator(CT, U; method=copula_method, weights,
                                         copula_kwargs...)

    S = SklarDist(cop_estimate.result, m)
    model || return S
    ll = _weighted_loglikelihood(S, X, w)
    fit_kwargs = (; copula_method=cop_estimate.method, sklar_method,
                    margins_kwargs, copula_kwargs)
    w === nothing || (fit_kwargs = (; fit_kwargs..., weights=w))
    recipe = _CopulaFitSpec(T, cop_estimate.method, fit_kwargs)
    return CopulaModel(S, X, ll, recipe)
end

# One margin of the Sklar route. With weights it is the weighted
# maximum-likelihood fit that Distributions.jl defines through
# `suffstats(D, x, w)` or a `fit_mle(D, x, w)` method; a family with neither
# is refused by name rather than by the error Distributions.jl raises inside
# `fit`, a `MethodError` or the "not implemented" fallback of `suffstats`.
_fit_margin(D, x, ::Nothing; kwargs...) = Distributions.fit(D, x; kwargs...)
function _fit_margin(D, x, w::AbstractVector; kwargs...)
    try
        return Distributions.fit(D, x, w; kwargs...)
    catch err
        _is_missing_weighted_fit(err) || rethrow()
        throw(ArgumentError(
            "Distributions.jl defines no weighted fit for the margin family $D " *
            "(neither `suffstats($D, x, w)` nor `fit_mle($D, x, w)` has a method), " *
            "so the Sklar route cannot take `weights` with this margin; fit the " *
            "margins yourself, then rank with `pseudos(X; weights)` and fit the " *
            "copula with `fit(CT, U; weights)`"))
    end
end
function _is_missing_weighted_fit(err::MethodError)
    f = err.f === Core.kwcall ? err.args[2] : err.f
    return f in (Distributions.fit, Distributions.fit_mle, Distributions.suffstats)
end
_is_missing_weighted_fit(err::ErrorException) =
    startswith(err.msg, "suffstats is not implemented")
_is_missing_weighted_fit(::Exception) = false

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
sample size used by likelihood summaries and information criteria. A weighted
fit normalizes its weights to sum to this number, so it reports the same value.

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


function _parameter_blocks(M::CopulaModel)
    all = eachindex(StatsBase.coef(M))
    return (; copula=all, margins=())
end

function _copula_data(M::CopulaModel)
    D, data, spec = fitted_distribution(M), M.data, M.recipe
    if D isa SklarDist
        sklar_method = spec.kwargs.sklar_method
        sklar_method === :ecdf && return _normalized_pseudos(data, _model_weights(M))
        T = promote_type(float(eltype(data)), mapreduce(eltype, promote_type, D.m))
        U = Matrix{T}(undef, size(data))
        lower, upper = nextfloat(zero(T)), prevfloat(one(T))
        @inbounds for j in axes(data, 2), i in axes(data, 1)
            U[i, j] = clamp(Distributions.cdf(D.m[i], data[i, j]), lower, upper)
        end
        return U
    end
    if spec.method === :mpl || get(spec.kwargs, :pseudo_values, true) === false
        return _normalized_pseudos(data, _model_weights(M))
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
    return _weighted_loglikelihood(null, data, _model_weights(M))
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
end

"""
    selected_model(result::CopulaSelection) -> CopulaModel

Return the winning fitted model from an automatic family-selection result.

See also: [`selection_table`](@ref), [`fitted_distribution`](@ref).
"""
selected_model(S::CopulaSelection) = S.model
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
            best, best_score = M, score
        end
    end
    best === nothing && throw(ArgumentError("No candidate copula produced an eligible finite fit."))
    return CopulaSelection(best, rows, criterion)
end

Distributions.fit(::Type{Copula}, U; candidates, kwargs...) =
    fitted_distribution(selected_model(Distributions.fit(
        CopulaModel, Copula, U; candidates, kwargs...)))