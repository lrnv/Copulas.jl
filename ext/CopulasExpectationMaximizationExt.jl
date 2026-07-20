module CopulasExpectationMaximizationExt

using ADTypes
using Copulas
using Distributions
using ExpectationMaximization
using Optim

import Distributions: fit_mle

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

    result = try
        Optim.optimize(
            objective,
            alpha0,
            Optim.LBFGS();
            autodiff=ADTypes.AutoForwardDiff(),
        )
    catch
        Optim.optimize(objective, alpha0, Optim.NelderMead())
    end
    return copula(Optim.minimizer(result))
end

function _fit_margins(S::Copulas.SklarDist, X::AbstractMatrix, weights; kwargs...)
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "the Sklar distribution has dimension $d, but the sample has $(size(X, 1)) rows",
    ))
    isempty(weights) || _check_weights(only(weights), size(X, 2))
    return ntuple(d) do i
        fit_mle(S.m[i], view(X, i, :), weights...; kwargs...)
    end
end

function _fit_sklar_mle(
    S::Copulas.SklarDist,
    X::AbstractMatrix,
    weights;
    margins_kwargs::NamedTuple=NamedTuple(),
    copula_kwargs::NamedTuple=NamedTuple(),
)
    margins = _fit_margins(S, X, weights; margins_kwargs...)
    U = similar(X)
    @inbounds for i in axes(X, 1)
        U[i, :] .= Distributions.cdf.(Ref(margins[i]), view(X, i, :))
    end
    copula = fit_mle(S.C, U, weights...; copula_kwargs...)
    return Copulas.SklarDist(copula, margins)
end

"""Fit a Sklar distribution while retaining initialized structured margins."""
function fit_mle(
    S::Copulas.SklarDist,
    X::AbstractMatrix;
    margins_kwargs::NamedTuple=NamedTuple(),
    copula_kwargs::NamedTuple=NamedTuple(),
)
    return _fit_sklar_mle(
        S,
        X,
        ();
        margins_kwargs=margins_kwargs,
        copula_kwargs=copula_kwargs,
    )
end

"""Weighted Sklar MLE used when a Sklar distribution is an EM component."""
function fit_mle(
    S::Copulas.SklarDist,
    X::AbstractMatrix,
    weights::AbstractVector;
    margins_kwargs::NamedTuple=NamedTuple(),
    copula_kwargs::NamedTuple=NamedTuple(),
)
    return _fit_sklar_mle(
        S,
        X,
        (weights,);
        margins_kwargs=margins_kwargs,
        copula_kwargs=copula_kwargs,
    )
end

end
