# 1.0 public exception taxonomy for the ordinary real-valued distribution paths.
# More-specific methods keep the existing numerical behavior while ensuring
# public shape mismatches consistently report DimensionMismatch.

function Distributions.cdf(C::Copula{d}, u::AbstractVector{T}) where {d,T<:Real}
    length(u) == d || throw(DimensionMismatch(
        "input vector has length $(length(u)); expected copula dimension $d",
    ))
    if any(x -> x <= zero(x), u)
        return zero(u[1])
    elseif all(x -> x >= one(x), u)
        return one(u[1])
    end
    bounded = any(x -> x > one(x), u) ? min.(u, one(eltype(u))) : u
    return _cdf(C, bounded)
end

function Distributions.cdf(C::Copula{d}, A::AbstractMatrix{T}) where {d,T<:Real}
    size(A, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(A, 1)) rows; expected copula dimension $d",
    ))
    return [Distributions.cdf(C, u) for u in eachcol(A)]
end

function Distributions.logpdf(C::Copula{d}, A::AbstractMatrix{T}) where {d,T<:Real}
    size(A, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(A, 1)) rows; expected copula dimension $d",
    ))
    return [Distributions.logpdf(C, u) for u in eachcol(A)]
end

function Distributions.cdf(S::SklarDist, x::AbstractVector{T}) where {T<:Real}
    d = length(S)
    length(x) == d || throw(DimensionMismatch(
        "input vector has length $(length(x)); expected distribution dimension $d",
    ))
    WT = _sklar_work_eltype(S, x)
    u = Vector{WT}(undef, d)
    @inbounds for i in 1:d
        u[i] = Distributions.cdf(S.m[i], x[i])
    end
    return Distributions.cdf(S.C, u)
end

function Distributions.cdf(S::SklarDist, X::AbstractMatrix{T}) where {T<:Real}
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(X, 1)) rows; expected distribution dimension $d",
    ))
    return [Distributions.cdf(S, x) for x in eachcol(X)]
end

function Distributions.pdf(S::SklarDist, X::AbstractMatrix{T}) where {T<:Real}
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(X, 1)) rows; expected distribution dimension $d",
    ))
    return [Distributions.pdf(S, x) for x in eachcol(X)]
end

function Distributions.logpdf(S::SklarDist, X::AbstractMatrix{T}) where {T<:Real}
    d = length(S)
    size(X, 1) == d || throw(DimensionMismatch(
        "input matrix has $(size(X, 1)) rows; expected distribution dimension $d",
    ))
    return [Distributions.logpdf(S, x) for x in eachcol(X)]
end

function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    S::SklarDist,
    A::AbstractMatrix{T},
) where {T<:Real}
    d = length(S)
    size(A, 1) == d || throw(DimensionMismatch(
        "output matrix has $(size(A, 1)) rows; expected distribution dimension $d",
    ))
    Random.rand!(rng, S.C, A)
    lo, hi = nextfloat(T(0)), prevfloat(T(1))
    @inbounds for col in axes(A, 2), row in axes(A, 1)
        A[row, col] = Distributions.quantile(S.m[row], clamp(A[row, col], lo, hi))
    end
    return A
end

function Distributions._logpdf(S::SklarDist, u::AbstractVector{T}) where {T<:Real}
    d = length(S)
    length(u) == d || throw(DimensionMismatch(
        "input vector has length $(length(u)); expected distribution dimension $d",
    ))
    _has_atoms(S.m) && return _sklar_logpdf_atoms(S, u)
    WT = _sklar_work_eltype(S, u)
    s = zero(WT)
    @inbounds for i in 1:d
        s += Distributions.logpdf(S.m[i], u[i])
    end
    U = Vector{WT}(undef, d)
    @inbounds for i in 1:d
        U[i] = clamp(Distributions.cdf(S.m[i], u[i]), zero(WT), one(WT))
    end
    return s + Distributions.logpdf(S.C, U)
end

# EmpiricalCopula-backed Sklar models have an atomic likelihood specialization
# whose first argument is more specific than the generic method above while its
# second argument is less specific. Define the intersection explicitly so the
# 1.0 dimension check does not introduce a dispatch ambiguity.
function Distributions._logpdf(
    S::SklarDist{CT},
    u::AbstractVector{T},
) where {CT<:EmpiricalCopula,T<:Real}
    d = length(S)
    length(u) == d || throw(DimensionMismatch(
        "input vector has length $(length(u)); expected distribution dimension $d",
    ))
    return invoke(Distributions._logpdf, Tuple{SklarDist{CT},Any}, S, u)
end
