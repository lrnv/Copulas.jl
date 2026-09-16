# Scalar multivariate concordance normalizations are defined for d >= 2.
# Copula objects satisfy that invariant through their public constructors; data
# methods need an explicit guard because a 1×n matrix is otherwise admissible.
@inline function _require_dependence_dimension(U::AbstractMatrix)
    d = size(U, 1)
    d >= 2 || throw(DimensionMismatch(
        "scalar multivariate dependence summaries require at least two rows; got d=$d",
    ))
    return nothing
end

function τ(U::AbstractMatrix{T}) where {T<:Real}
    _require_dependence_dimension(U)
    return invoke(τ, Tuple{AbstractMatrix}, U)
end
function ρ(U::AbstractMatrix{T}) where {T<:Real}
    _require_dependence_dimension(U)
    return invoke(ρ, Tuple{AbstractMatrix}, U)
end
function β(U::AbstractMatrix{T}) where {T<:Real}
    _require_dependence_dimension(U)
    return invoke(β, Tuple{AbstractMatrix}, U)
end
function γ(U::AbstractMatrix{T}) where {T<:Real}
    _require_dependence_dimension(U)
    return invoke(γ, Tuple{AbstractMatrix}, U)
end
