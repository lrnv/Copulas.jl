module CopulasPartitionedDistributionsExt

using Copulas
using Distributions
using PartitionedDistributions

import Copulas: condition, inverse_rosenblatt, rosenblatt, subsetdims
import PartitionedDistributions: conditional, marginal


const CopulaLike = Union{Copulas.Copula,Copulas.SklarDist}


###############################################################################
# Utilities
###############################################################################

# PartitionedDistributions follows ordinary array indexing semantics.
# Copulas.jl, on the other hand, represents selected dimensions internally
# as tuples of integer indices.
_normalize_keep(keep::Tuple{Vararg{Integer}}) = collect(keep)
_normalize_keep(keep) = keep


"""
Return the selected linear indices and the shape requested by the
PartitionedDistributions selector.

For a scalar selector, `shape === nothing`; for array selectors, `shape`
records the shape that PartitionedDistributions expects the returned
distribution to have.
"""
function _selected_indices(dist::CopulaLike, keep)
    linear = LinearIndices(axes(dist))
    selected = linear[_normalize_keep(keep)]

    if selected isa Integer
        return (Int(selected),), nothing
    end

    inds = Tuple(Int(i) for i in vec(collect(selected)))

    isempty(inds) &&
        throw(ArgumentError("At least one element must be selected."))

    allunique(inds) ||
        throw(ArgumentError("Indices must be unique."))

    return inds, size(selected)
end


_restore_shape(result, ::Nothing) = result

function _restore_shape(result, shape::Tuple)
    size(result) == shape && return result
    return reshape(result, shape)
end


function _copulas_indices(dist, dims; proper_subset::Bool=false)
    inds = if dims isa Integer
        (Int(dims),)
    else
        Tuple(Int(i) for i in dims)
    end

    d = length(dist)

    isempty(inds) &&
        throw(ArgumentError("At least one dimension must be selected."))

    all(i -> 1 <= i <= d, inds) ||
        throw(ArgumentError("Dimension indices must lie in 1:$d."))

    allunique(inds) ||
        throw(ArgumentError("Dimension indices must be unique."))

    if proper_subset && length(inds) == d
        throw(ArgumentError(
            "Conditioning indices must be a non-empty proper subset of 1:$d.",
        ))
    end

    return inds
end


_pdist_selector(inds::Tuple) =
    length(inds) == 1 ? only(inds) : collect(inds)


###############################################################################
# PartitionedDistributions API on Copulas.jl distributions
###############################################################################

"""
Implement `PartitionedDistributions.marginal` for Copula and SklarDist objects
through Copulas.jl's native `subsetdims` machinery.
"""
function marginal(
    dist::CopulaLike,
    keep,
)
    inds, shape = _selected_indices(dist, keep)

    result = Copulas.subsetdims(
        dist,
        inds,
    )

    return _restore_shape(result, shape)
end


"""
Implement `PartitionedDistributions.conditional` for Copula and SklarDist
objects through Copulas.jl's native conditioning machinery.

`PartitionedDistributions` specifies the coordinates to keep, whereas
`Copulas.condition` specifies the coordinates on which to condition.
"""
function conditional(
    dist::CopulaLike,
    x::AbstractVector,
    keep,
)
    d = length(dist)

    length(x) == d || throw(DimensionMismatch(
        "the distribution has dimension $d, but the supplied point has " *
        "length $(length(x))",
    ))

    keepinds, shape = _selected_indices(dist, keep)

    conditioned = Tuple(
        i for i in 1:d
        if i ∉ keepinds
    )

    # Keeping every coordinate is just a marginal/permutation operation.
    if isempty(conditioned)
        result = Copulas.subsetdims(
            dist,
            keepinds,
        )
        return _restore_shape(result, shape)
    end

    observed = Tuple(
        x[j] for j in conditioned
    )

    result = Copulas.condition(
        dist,
        conditioned,
        observed,
    )

    # Copulas.condition returns the remaining coordinates in their original
    # order. PartitionedDistributions permits selectors that reorder them.
    natural_order = Tuple(
        i for i in 1:d
        if i ∉ conditioned
    )

    if length(keepinds) > 1 && keepinds != natural_order
        permutation = ntuple(length(keepinds)) do k
            something(
                findfirst(==(keepinds[k]), natural_order),
            )
        end

        result = Copulas.subsetdims(
            result,
            permutation,
        )
    end

    return _restore_shape(result, shape)
end


###############################################################################
# Copulas.jl API on PartitionedDistributions-compatible distributions
###############################################################################

"""
Use PartitionedDistributions' marginal implementation as the generic
`subsetdims` fallback for vector-variate Distributions.jl distributions.

More-specific Copula and SklarDist methods continue to dispatch to Copulas.jl's
native implementations.
"""
function subsetdims(
    dist::Distributions.Distribution{
        Distributions.ArrayLikeVariate{1}
    },
    dims,
)
    inds = _copulas_indices(dist, dims)

    return PartitionedDistributions.marginal(
        dist,
        _pdist_selector(inds),
    )
end


"""
Use PartitionedDistributions' conditional implementation as the generic
`condition` fallback for vector-variate Distributions.jl distributions.

Copulas.condition receives only the observed coordinates, whereas
PartitionedDistributions.conditional takes a complete point. The coordinates
that are kept are therefore filled with placeholders; a conditional law can
depend only on the observed coordinates.
"""
function condition(
    dist::Distributions.Distribution{
        Distributions.ArrayLikeVariate{1}
    },
    js,
    xjs,
)
    conditioned = _copulas_indices(
        dist,
        js;
        proper_subset=true,
    )

    observed = if xjs isa Number
        (xjs,)
    else
        Tuple(xjs)
    end

    length(conditioned) == length(observed) ||
        throw(DimensionMismatch(
            "conditioning indices and conditioning values must have " *
            "the same length",
        ))

    all(x -> x isa Number && isfinite(x), observed) ||
        throw(ArgumentError("conditioning values must be finite numbers"))

    d = length(dist)

    keep = Tuple(
        i for i in 1:d
        if i ∉ conditioned
    )

    # PartitionedDistributions' public API takes a full support point even
    # though only the observed coordinates define the conditional law. Complete
    # retained coordinates with deterministic points from their own marginal
    # supports, then validate the assembled point against the joint support.
    x = Any[]
    for i in 1:d
        margin = subsetdims(dist, (i,))
        placeholder = Distributions.quantile(margin, 0.5)
        Distributions.insupport(margin, placeholder) || throw(ArgumentError(
            "the median of marginal $i is not in its support; call " *
            "PartitionedDistributions.conditional directly with a complete " *
            "support point",
        ))
        push!(x, placeholder)
    end

    @inbounds for k in eachindex(conditioned)
        x[conditioned[k]] = observed[k]
    end

    x = collect(promote(x...))

    Distributions.insupport(dist, x) || throw(ArgumentError(
        "cannot construct a full in-support point from the supplied " *
        "conditioning values; call PartitionedDistributions.conditional " *
        "directly with a complete support point",
    ))

    return PartitionedDistributions.conditional(
        dist,
        x,
        _pdist_selector(keep),
    )
end


###############################################################################
# Rosenblatt transforms for PartitionedDistributions-compatible distributions
###############################################################################

function _rosenblatt_output(x)
    return similar(x, float(eltype(x)))
end


"""
Extend `Copulas.rosenblatt` to compatible vector-valued distributions supported
by PartitionedDistributions, using successive marginals and conditionals.
"""
function rosenblatt(
    dist::Distributions.Distribution{
        Distributions.ArrayLikeVariate{1}
    },
    x::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
)
    d = length(dist)
    size(x, 1) == d || throw(DimensionMismatch(
        "the distribution has dimension $d, but the input has " *
        "$(size(x, 1)) rows",
    ))

    isvector = x isa AbstractVector
    X = isvector ? reshape(x, d, 1) : x
    S = _rosenblatt_output(X)

    first_marginal = subsetdims(dist, (1,))
    @inbounds for j in axes(X, 2)
        S[1, j] = cdf(first_marginal, X[1, j])
    end

    for k in 2:d
        prefix = subsetdims(dist, ntuple(identity, k))
        observed_dims = ntuple(identity, k - 1)

        @inbounds for j in axes(X, 2)
            observed = ntuple(i -> X[i, j], k - 1)
            conditional_k = condition(prefix, observed_dims, observed)
            S[k, j] = cdf(conditional_k, X[k, j])
        end
    end

    return isvector ? vec(S) : S
end


"""
Extend `Copulas.inverse_rosenblatt` to compatible vector-valued distributions
supported by PartitionedDistributions, using successive conditional quantiles.
"""
function inverse_rosenblatt(
    dist::Distributions.Distribution{
        Distributions.ArrayLikeVariate{1}
    },
    s::Union{AbstractVector{<:Real},AbstractMatrix{<:Real}},
)
    d = length(dist)
    size(s, 1) == d || throw(DimensionMismatch(
        "the distribution has dimension $d, but the input has " *
        "$(size(s, 1)) rows",
    ))

    isvector = s isa AbstractVector
    S = isvector ? reshape(s, d, 1) : s
    X = _rosenblatt_output(S)

    first_marginal = subsetdims(dist, (1,))
    @inbounds for j in axes(S, 2)
        X[1, j] = quantile(first_marginal, clamp(float(S[1, j]), 0.0, 1.0))
    end

    for k in 2:d
        prefix = subsetdims(dist, ntuple(identity, k))
        observed_dims = ntuple(identity, k - 1)

        @inbounds for j in axes(S, 2)
            observed = ntuple(i -> X[i, j], k - 1)
            conditional_k = condition(prefix, observed_dims, observed)
            X[k, j] = quantile(
                conditional_k,
                clamp(float(S[k, j]), 0.0, 1.0),
            )
        end
    end

    return isvector ? vec(X) : X
end


end
