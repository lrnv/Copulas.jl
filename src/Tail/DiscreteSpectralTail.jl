
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
struct DiscreteSpectralTail{T} <: Tail
    B::Matrix{T}
    function DiscreteSpectralTail{T}(B::Matrix{T}) where {T}
        return new{T}(B)
    end
end

function DiscreteSpectralTail(B::AbstractMatrix)
    d, m = size(B)
    d >= 2 || throw(ArgumentError(
        "a discrete spectral tail requires dimension d ≥ 2",
    ))
    m >= 1 || throw(ArgumentError(
        "a discrete spectral tail requires at least one spectral atom",
    ))

    vals = collect(B)
    T = promote_type(Float64, map(typeof, vals)...)
    BB = Matrix{T}(B)

    all(isfinite, BB) || throw(ArgumentError(
        "all discrete spectral coefficients must be finite",
    ))
    all(v -> v >= zero(T), BB) || throw(ArgumentError(
        "all discrete spectral coefficients must be nonnegative",
    ))

    tol = sqrt(eps(T))
    @inbounds for i in 1:d
        s = sum(@view BB[i, :])
        isapprox(s, one(T); atol=tol, rtol=tol) || throw(ArgumentError(
            "row $i of B must sum to one; got $s",
        ))
    end

    return DiscreteSpectralTail{T}(BB)
end

Base.eltype(::DiscreteSpectralTail{T}) where {T} = T
Distributions.params(tail::DiscreteSpectralTail) = (B = tail.B,)
_is_valid_in_dim(tail::DiscreteSpectralTail, d::Int) =
    size(tail.B, 1) == d

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
    length(x) == d || throw(DimensionMismatch(
        "input dimension does not match discrete spectral tail dimension",
    ))

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

function _spectral_subsets(d::Int)
    d >= 1 || throw(ArgumentError("dimension must be positive"))
    out = Vector{Vector{Int}}()
    for k in 1:d
        for S in Combinatorics.combinations(1:d, k)
            push!(out, collect(S))
        end
    end
    return out
end

function _discrete_spectral_rand!(
    rng::Distributions.AbstractRNG,
    tail::DiscreteSpectralTail,
    X::AbstractMatrix{T},
) where {T<:Real}
    d, n = size(X)
    d == size(tail.B, 1) || throw(DimensionMismatch(
        "output dimension does not match discrete spectral tail dimension",
    ))

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
        zi > zero(T) || throw(ArgumentError(
            "invalid zero max-linear factor for margin $i",
        ))
        X[i, col] = exp(-inv(zi))
    end

    return X
end

_rand_ev_multivariate!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d,<:DiscreteSpectralTail},
    X::AbstractMatrix{T},
) where {d,T<:Real} =
    _discrete_spectral_rand!(rng, C.tail, X)

function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{2,<:DiscreteSpectralTail},
    X::AbstractMatrix{T},
) where {T<:Real}
    size(X, 1) == 2 || throw(DimensionMismatch(
        "output must have two rows for a bivariate discrete spectral copula",
    ))
    return _discrete_spectral_rand!(rng, C.tail, X)
end

function Distributions._logpdf(
    ::ExtremeValueCopula{d,<:DiscreteSpectralTail},
    u,
) where {d}
    throw(ArgumentError(
        "DiscreteSpectralCopula can contain singular components; " *
        "a global Lebesgue log-density is not defined in general",
    ))
end
