# Public fitting-target validation kept separate from the estimator machinery.
# `SklarDist{CT,Tuple{M₁,...,M_d}}` is documented public syntax, so malformed
# target types should fail before any marginal fit starts.
function _validate_sklar_fitting_target(
    ::Type{SklarDist{CT,TplMargins}}, d::Integer,
) where {CT<:Copula,TplMargins<:Tuple}
    margins = TplMargins.parameters
    length(margins) == d || throw(DimensionMismatch(
        "SklarDist fitting target specifies $(length(margins)) margin families " *
        "for data with $d rows",
    ))
    all(M -> M <: Distributions.UnivariateDistribution, margins) ||
        throw(ArgumentError(
            "every SklarDist fitting-target margin must be a univariate distribution family",
        ))
    return nothing
end

@inline function Distributions.fit(
    T::Type{SklarDist{CT,TplMargins}}, X::AbstractMatrix; kwargs...,
) where {CT<:Copula,TplMargins<:Tuple}
    _validate_sklar_fitting_target(T, size(X, 1))
    return _estimate_sklar(T, X; model=false, kwargs...)
end

function Distributions.fit(
    ::Type{CopulaModel}, T::Type{SklarDist{CT,TplMargins}}, X::AbstractMatrix;
    kwargs...,
) where {CT<:Copula,TplMargins<:Tuple}
    _validate_sklar_fitting_target(T, size(X, 1))
    return _estimate_sklar(T, X; kwargs...)
end
