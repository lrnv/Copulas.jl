
"""
    DiscreteSpectralTail(B)

Finite discrete spectral representation of a multivariate extreme-value copula.

`B` is a nonnegative `d × m` matrix whose rows sum to one. Its stable tail
dependence function is

    ℓ(x) = sum(maximum(B[:, k] .* x) for k in 1:m).

Equivalently, if `h[k] = sum(B[:,k])` and `v[:,k] = B[:,k] / h[k]`,
the associated spectral measure is `sum(h[k] δ_{v[:,k]})`.

The row-sum constraints are exactly the spectral moment constraints required
for unit-Fréchet / uniform margins.
"""
abstract type DiscreteSpectralBackedTail <: Tail end

# Combined capability for tails that expose both a native bivariate Pickands
# representation and a finite discrete spectral representation in any valid
# dimension.
abstract type DiscreteSpectralPickandsTail <: BivariatePickandsTail end
const DiscreteSpectralCapableTail = Union{
    DiscreteSpectralBackedTail,
    DiscreteSpectralPickandsTail,
}

struct DiscreteSpectralTail{T} <: DiscreteSpectralBackedTail
    B::Matrix{T}
    function DiscreteSpectralTail{T}(B::Matrix{T}) where {T}
        return new{T}(B)
    end
end

_spectral_tail(tail::DiscreteSpectralTail) = tail
_spectral_tail(tail::DiscreteSpectralCapableTail) = tail.spectral

function DiscreteSpectralTail(B::AbstractMatrix)
    d, m = size(B)
    d >= 2 || throw(ArgumentError("a discrete spectral tail requires dimension d ≥ 2",))
    m >= 1 || throw(ArgumentError("a discrete spectral tail requires at least one spectral atom",))

    vals = collect(B)
    T = promote_type(Float64, map(typeof, vals)...)
    BB = Matrix{T}(B)

    all(isfinite, BB) || throw(ArgumentError("all discrete spectral coefficients must be finite",))
    all(v -> v >= zero(T), BB) || throw(ArgumentError("all discrete spectral coefficients must be nonnegative",))
  
    tol = sqrt(eps(T))
    @inbounds for i in 1:d
        s = sum(@view BB[i, :])
        isapprox(s, one(T); atol=tol, rtol=tol) || throw(ArgumentError("row $i of B must sum to one; got $s",))
    end

    return DiscreteSpectralTail{T}(BB)
end

Base.eltype(::DiscreteSpectralTail{T}) where {T} = T
Distributions.params(tail::DiscreteSpectralTail) = (B = tail.B,)
_is_valid_in_dim(tail::DiscreteSpectralTail, d::Int) = size(tail.B, 1) == d

"""
    DiscreteSpectralCopula(B)

Construct the extreme-value copula associated with the discrete spectral
coefficient matrix `B`.
"""
function DiscreteSpectralCopula(B::AbstractMatrix)
    tail = DiscreteSpectralTail(B)
    return ExtremeValueCopula(size(tail.B, 1), tail)
end

DiscreteSpectralCopula(tail::DiscreteSpectralTail) =
    ExtremeValueCopula(size(tail.B, 1), tail)

function ℓ(tail::DiscreteSpectralTail, x)
    d, m = size(tail.B)
    length(x) == d || throw(DimensionMismatch("input dimension does not match discrete spectral tail dimension",))

    T = promote_type(eltype(tail.B), typeof(first(x)))
    out = zero(T)

    @inbounds for k in 1:m
        best = zero(T)
        for i in 1:d
            candidate = tail.B[i, k] * x[i]
            if candidate > best
                best = candidate
            end
        end
        out += best
    end

    return out
end

ℓ(tail::DiscreteSpectralCapableTail, x) = ℓ(_spectral_tail(tail), x)

function _discrete_spectral_rand!(rng::Distributions.AbstractRNG, tail::DiscreteSpectralTail, X::AbstractMatrix{T},) where {T<:Real}
    d, n = size(X)
    d == size(tail.B, 1) || throw(DimensionMismatch("output dimension does not match discrete spectral tail dimension",))

    fill!(X, zero(T))
    m = size(tail.B, 2)

    @inbounds for col in 1:n
        for k in 1:m
            invE = inv(Random.randexp(rng))
            for i in 1:d
                b = tail.B[i, k]
                iszero(b) && continue
                candidate = T(b * invE)
                if candidate > X[i, col]
                    X[i, col] = candidate
                end
            end
        end
    end

    @inbounds for i in 1:d, col in 1:n
        zi = X[i, col]
        zi > zero(T) || throw(ArgumentError("invalid zero max-linear factor for margin $i",))
        X[i, col] = exp(-inv(zi))
    end

    return X
end

function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d,<:DiscreteSpectralCapableTail},
    X::AbstractMatrix{T},
) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch(
        "output dimension does not match copula dimension",
    ))
    return _discrete_spectral_rand!(rng, _spectral_tail(C.tail), X)
end

function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{2,<:DiscreteSpectralPickandsTail},
    X::AbstractMatrix{T},
) where {T<:Real}
    return _discrete_spectral_rand!(rng, _spectral_tail(C.tail), X)
end

function Distributions._logpdf(::ExtremeValueCopula{d,<:DiscreteSpectralCapableTail}, u,) where {d}
    throw(ArgumentError("a discrete-spectral extreme-value copula can contain singular components; " *
        "a global Lebesgue log-density is not defined in general",))
end

Distributions._logpdf(C::ExtremeValueCopula{2,<:DiscreteSpectralPickandsTail}, u) =
    _bivariate_pickands_logpdf(C, u)
