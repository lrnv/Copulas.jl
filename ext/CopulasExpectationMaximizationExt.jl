module CopulasExpectationMaximizationExt

using ADTypes
using Copulas
using Distributions
using ExpectationMaximization
using Optim

import Distributions: fit_mle

const _LOCATION_SCALE_MARGINS = Union{
    Distributions.Cauchy,
    Distributions.Gumbel,
    Distributions.Laplace,
    Distributions.Logistic,
    Distributions.LogitNormal,
    Distributions.LogNormal,
    Distributions.Normal,
}

const _POSITIVE_MARGINS = Union{
    Distributions.Beta,
    Distributions.BetaPrime,
    Distributions.FDist,
    Distributions.Gamma,
    Distributions.InverseGaussian,
    Distributions.Pareto,
    Distributions.Weibull,
}

const _POSITIVE_SCALAR_MARGINS = Union{
    Distributions.Chisq,
    Distributions.Exponential,
    Distributions.Rayleigh,
    Distributions.TDist,
}

function _check_sample(C::Copulas.Copula, U::AbstractMatrix)
    size(U, 1) == length(C) || throw(DimensionMismatch(
        "the copula has dimension $(length(C)), but the sample has $(size(U, 1)) rows",
    ))
    return size(U, 2)
end

function _check_weights(weights::AbstractVector, n::Integer)
    length(weights) == n || throw(DimensionMismatch(
        "the sample contains $n observations, but $(length(weights)) weights were provided",
    ))
    all(isfinite, weights) || throw(ArgumentError("weights must be finite"))
    all(>=(zero(eltype(weights))), weights) || throw(ArgumentError("weights must be nonnegative"))
    any(>(zero(eltype(weights))), weights) || throw(ArgumentError("at least one weight must be positive"))
    return nothing
end

_distribution_wrapper(D::Distributions.Distribution) = Base.typename(typeof(D)).wrapper

function _margin_unbound(D::_LOCATION_SCALE_MARGINS)
    location, scale = Distributions.params(D)
    return [float(location), log(float(scale))]
end

function _margin_unbound(D::_POSITIVE_MARGINS)
    return [log(float(parameter)) for parameter in Distributions.params(D)]
end

function _margin_unbound(D::_POSITIVE_SCALAR_MARGINS)
    return [log(float(only(Distributions.params(D))))]
end

function _margin_unbound(D::Distributions.Uniform)
    lower, upper = Distributions.params(D)
    return [float(lower), log(float(upper - lower))]
end

function _margin_unbound(
    D::Distributions.MixtureModel{Distributions.Univariate,Distributions.Continuous},
)
    probabilities = Distributions.probs(D)
    all(>(zero(eltype(probabilities))), probabilities) || throw(ArgumentError(
        "joint Sklar MLE requires strictly positive initial mixture probabilities",
    ))
    component_parameters = mapreduce(
        _margin_unbound,
        vcat,
        Distributions.components(D),
    )
    reference_probability = last(probabilities)
    logits = [
        log(float(probabilities[k])) - log(float(reference_probability))
        for k in 1:(length(probabilities) - 1)
    ]
    return vcat(component_parameters, logits)
end


function _margin_unbound(D::Distributions.DiscreteUnivariateDistribution)
    throw(ArgumentError(
        "joint Sklar MLE is only defined for continuous margins; got $(typeof(D))",
    ))
end


function _margin_unbound(D::Distributions.ContinuousUnivariateDistribution)
    throw(ArgumentError(
        "joint Sklar MLE does not yet provide an unconstrained parameterization for $(typeof(D))",
    ))
end


function _take_parameters!(alpha::AbstractVector, cursor::Base.RefValue{Int}, count::Int)
    first_index = cursor[]
    last_index = first_index + count - 1
    cursor[] = last_index + 1
    return @view alpha[first_index:last_index]
end


function _margin_rebound(
    template::_LOCATION_SCALE_MARGINS,
    alpha::AbstractVector,
    cursor::Base.RefValue{Int},
)
    parameters = _take_parameters!(alpha, cursor, 2)
    return _distribution_wrapper(template)(parameters[1], exp(parameters[2]))
end


function _margin_rebound(
    template::_POSITIVE_MARGINS,
    alpha::AbstractVector,
    cursor::Base.RefValue{Int},
)
    count = length(Distributions.params(template))
    parameters = _take_parameters!(alpha, cursor, count)
    return _distribution_wrapper(template)(exp.(parameters)...)
end


function _margin_rebound(
    template::_POSITIVE_SCALAR_MARGINS,
    alpha::AbstractVector,
    cursor::Base.RefValue{Int},
)
    parameter = only(_take_parameters!(alpha, cursor, 1))
    return _distribution_wrapper(template)(exp(parameter))
end


function _margin_rebound(
    template::Distributions.Uniform,
    alpha::AbstractVector,
    cursor::Base.RefValue{Int},
)
    parameters = _take_parameters!(alpha, cursor, 2)
    lower = parameters[1]
    return Distributions.Uniform(lower, lower + exp(parameters[2]))
end


function _margin_rebound(
    template::Distributions.MixtureModel{
        Distributions.Univariate,
        Distributions.Continuous,
    },
    alpha::AbstractVector,
    cursor::Base.RefValue{Int},
)
    fitted_components = Any[
        _margin_rebound(component, alpha, cursor)
        for component in Distributions.components(template)
    ]
    initial_component_type = eltype(Distributions.components(template))
    typed_components = if all(component -> component isa initial_component_type, fitted_components)
        initial_component_type[fitted_components...]
    else
        Distributions.ContinuousUnivariateDistribution[fitted_components...]
    end

    component_count = Distributions.ncomponents(template)
    if component_count == 1
        probabilities = [one(eltype(alpha))]
    else
        free_logits = _take_parameters!(alpha, cursor, component_count - 1)
        logits = vcat(free_logits, zero(eltype(alpha)))
        shift = maximum(logits)
        probabilities = exp.(logits .- shift)
        probabilities ./= sum(probabilities)
    end
    return Distributions.MixtureModel(typed_components, probabilities)
end


function _optimize_unconstrained(objective, alpha0::AbstractVector)
    initial_loss = objective(alpha0)
    isfinite(initial_loss) || throw(ArgumentError(
        "the initialized model has a non-finite weighted log-likelihood",
    ))
    options = Optim.Options(; iterations=250)
    result = try
        Optim.optimize(
            objective,
            alpha0,
            Optim.LBFGS(),
            options;
            autodiff=ADTypes.AutoForwardDiff(),
        )
    catch
        try
            Optim.optimize(
                objective,
                alpha0,
                Optim.LBFGS(),
                options;
                autodiff=ADTypes.AutoFiniteDiff(; fdtype=Val(:central)),
            )
        catch
            Optim.optimize(objective, alpha0, Optim.NelderMead(), options)
        end
    end

    fitted_alpha = Optim.minimizer(result)
    fitted_loss = objective(fitted_alpha)
    return isfinite(fitted_loss) && fitted_loss <= initial_loss ? fitted_alpha : alpha0
end

"""Fit an initialized copula through the ordinary Copulas.jl interface."""
function fit_mle(
    C::Copulas.Copula,
    U::AbstractMatrix;
    method::Symbol=:mle,
    kwargs...,
)
    _check_sample(C, U)
    method === :mle || throw(ArgumentError(
        "fit_mle only supports method=:mle; use fit for other copula fitting methods",
    ))
    return Distributions.fit(typeof(C), U; method=method, kwargs...)
end

"""Weighted copula MLE used by ExpectationMaximization's M-step."""
function fit_mle(
    C::Copulas.Copula,
    U::AbstractMatrix,
    weights::AbstractVector;
    method::Symbol=:mle,
    kwargs...,
)
    n = _check_sample(C, U)
    _check_weights(weights, n)
    method === :mle || throw(ArgumentError(
        "weighted copula fitting is only implemented for method=:mle",
    ))
    isempty(kwargs) || throw(ArgumentError(
        "unsupported keyword arguments for weighted copula MLE: $(keys(kwargs))",
    ))

    isempty(Distributions.params(C)) && return C

    CT = typeof(C)
    d = length(C)
    copula(alpha) = CT(d, Copulas._rebound_params(CT, d, alpha)...)
    alpha0 = Copulas._unbound_params(CT, d, Distributions.params(C))
    function objective(alpha)
        fitted = copula(alpha)
        loss = zero(eltype(alpha))
        @inbounds for j in axes(U, 2)
            weight = weights[j]
            iszero(weight) && continue
            loss -= weight * Distributions.logpdf(fitted, view(U, :, j))
        end
        return loss
    end

    return copula(_optimize_unconstrained(objective, alpha0))
end

function _check_sklar_sample(S::Copulas.SklarDist, X::AbstractMatrix, weights)
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "the Sklar distribution has dimension $d, but the sample has $(size(X, 1)) rows",
    ))
    size(X, 2) > 0 || throw(ArgumentError("the sample must contain at least one observation"))
    all(isfinite, X) || throw(ArgumentError("the sample must contain only finite values"))
    isnothing(weights) || _check_weights(weights, size(X, 2))
    for margin in S.m
        margin isa Distributions.ContinuousUnivariateDistribution || throw(ArgumentError(
            "joint Sklar MLE is only defined for continuous margins; got $(typeof(margin))",
        ))
    end
    return nothing
end


function _sklar_unbound(S::Copulas.SklarDist)
    margin_parameters = mapreduce(_margin_unbound, vcat, S.m)
    copula_parameters = isempty(Distributions.params(S.C)) ? Float64[] :
        Copulas._unbound_params(typeof(S.C), length(S), Distributions.params(S.C))
    return vcat(margin_parameters, copula_parameters)
end


function _sklar_rebound(S::Copulas.SklarDist, alpha::AbstractVector)
    cursor = Ref(1)
    margins = ntuple(length(S)) do i
        _margin_rebound(S.m[i], alpha, cursor)
    end

    if isempty(Distributions.params(S.C))
        copula = S.C
    else
        copula_parameters = @view alpha[cursor[]:end]
        CT = typeof(S.C)
        copula = CT(
            length(S),
            Copulas._rebound_params(CT, length(S), copula_parameters)...,
        )
    end
    return Copulas.SklarDist(copula, margins)
end


function _joint_logpdf(
    S::Copulas.SklarDist,
    observation::AbstractVector,
    ::Type{T},
) where {T}
    value = zero(T)
    uniforms = Vector{T}(undef, length(S))
    boundary = zero(T) + eps(Float64)
    upper_boundary = one(T) - boundary
    @inbounds for i in eachindex(S.m)
        margin = S.m[i]
        x = observation[i]
        value += Distributions.logpdf(margin, x)
        uniforms[i] = clamp(
            Distributions.cdf(margin, x),
            boundary,
            upper_boundary,
        )
    end
    return value + Distributions.logpdf(S.C, uniforms)
end


function _fit_sklar_mle(
    S::Copulas.SklarDist,
    X::AbstractMatrix,
    weights,
)
    _check_sklar_sample(S, X, weights)
    alpha0 = _sklar_unbound(S)
    function objective(alpha)
        fitted = _sklar_rebound(S, alpha)
        loss = zero(eltype(alpha))
        work_type = promote_type(eltype(alpha), float(eltype(X)))
        @inbounds for j in axes(X, 2)
            weight = isnothing(weights) ? one(eltype(alpha)) : weights[j]
            iszero(weight) && continue
            value = _joint_logpdf(fitted, view(X, :, j), work_type)
            isfinite(value) || return loss + Inf
            loss -= weight * value
        end
        return loss
    end
    return _sklar_rebound(S, _optimize_unconstrained(objective, alpha0))
end


"""Jointly fit an initialized Sklar distribution by maximum likelihood."""
fit_mle(S::Copulas.SklarDist, X::AbstractMatrix) = _fit_sklar_mle(S, X, nothing)


"""Joint weighted Sklar MLE used when a Sklar distribution is an EM component."""
fit_mle(S::Copulas.SklarDist, X::AbstractMatrix, weights::AbstractVector) =
    _fit_sklar_mle(S, X, weights)

end
