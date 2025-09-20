"""
    pseudos(sample)

Compute the pseudo-observations of a multivariate sample. Note that the sample has to be given in wide format (d,n), where d is the dimension and n the number of observations.

Warning: the order used is ordinal ranking like https://en.wikipedia.org/wiki/Ranking#Ordinal_ranking_.28.221234.22_ranking.29, see `StatsBase.ordinalrank` for the ordering we use. If you want more flexibility, checkout `NormalizeQuantiles.sampleranks`.
"""
function pseudos(sample::AbstractMatrix)
    # Fast pseudo-observations (d×n) using per-row ordinal ranks without allocations per row
    d, n = size(sample)
    U = Matrix{Float64}(undef, d, n)
    tmp_idx = Vector{Int}(undef, n)
    @inbounds for i in 1:d
        # compute ordinal ranks for row i
        x = @view sample[i, :]
        # sortperm is stable; ordinal ranks from positions in sorted order
        sortperm!(tmp_idx, x; by=identity, alg=Base.Sort.DEFAULT_STABLE)
        # ranks: position in sorted order; ties preserve order of appearance
        for (rank, idx) in enumerate(tmp_idx)
            U[i, idx] = rank / (n + 1)
        end
    end
    return U
end



@inline _δ(t) = oftype(t, 1e-12)
@inline _safett(t) = clamp(t, _δ(t), one(t) - _δ(t))
@inline _as_tuple(x) = x isa Tuple ? x : (x,)


function _kendall_sample(u::AbstractMatrix)

    # Compute the empirical Kendall sample `W` with entries `W[i] = C_n(U[:,i])`,
    # where `C_n` is the Deheuvels empirical copula built from the same `u`.

    # Input and tie handling
    # - `u` is expected as a `d×n` matrix (columns are observations). This routine first
    #     applies per-margin ordinal ranks (same policy as `pseudos`) so that the result is
    #     invariant under strictly increasing marginal transformations and robust to ties.
    #     Consequently, `_kendall_sample(u) ≡ _kendall_sample(pseudos(u))` (same tie policy).

    # Returns
    # - `Vector{Float64}` of length `n` with values in `(0,1)`.

    d, n = size(u)
    # Apply ordinal ranks per margin to remove ties consistently with `pseudos`
    R = Matrix{Int}(undef, d, n)
    @inbounds for i in 1:d
        R[i, :] = StatsBase.ordinalrank(@view u[i, :])
    end
    W = zeros(Float64, n)
    @inbounds for i in 1:n
        ri = @view R[:, i]
        count_le = 0
        for j in 1:n
            count_le += all(@view(R[:, j]) .≤ ri)
        end
        W[i] = count_le / (n + 1)
    end
    return W
end

# TODO (performance): Replace the O(n^2) double loop with a Fenwick tree (Binary Indexed Tree) approach.
# Idea: for each observation i, process points in an order compatible with the partial order,
# updating a multi-dimensional count of how many prior points are ≤ current in all margins.
# In 2D, this reduces to sorting by one rank and querying a 1D BIT over the other rank in O(log n) per point.
# For d > 2, one can use a sweep over one margin and a (d-1)-dimensional BIT or recursively reduced structures;
# practical variants often use coordinate compression and nested BIT/segment trees to achieve ~O(n log^{d-1} n).
# We keep the simple O(n^2) version for clarity and modest n; contributions welcome to add a fast path.

@inline function _winsorize_tau_vclib(τ::Float64)
    s = τ < 0 ? -1.0 : 1.0
    a = abs(τ)
    a = a < 0.01 ? 0.01 : (a > 0.9 ? 0.9 : a)
    return s*a
end

# β̂ multivariate (Hofert–Mächler–McNeil, ec. (7))
function blomqvist_beta(U::AbstractMatrix)
    d, n = size(U)
    c = 2.0^(d-1) / (2.0^(d-1) - 1.0)
    acc = 0.0
    @inbounds for i in 1:n
        ui = view(U, :, i)
        q1 = all(ui .<= 0.5)
        q3 = all(ui .>  0.5)
        acc += (q1 || q3) ? 1.0 : 0.0
    end
    return c * (acc/n - 2.0^(1-d))
end

_uppertriangle_flat(mat) = [mat[idx] for idx in CartesianIndices(mat) if idx[1] < idx[2]]
function _uppertriangle_mean_var(mat)
    # compute the mean and std of the upper triangular part of the matrix (diagonal excluded)
    gen = _uppertriangle_flat(mat)
    return Statistics.mean(gen), length(gen) == 1 ? zero(gen[1]) : Statistics.std(gen), minimum(gen), maximum(gen)
end
