"""
    pseudos(sample)

Compute the pseudo-observations of a multivariate sample. Note that the sample has to be given in wide format (d,n), where d is the dimension and n the number of observations.

Warning: the order used is ordinal ranking like https://en.wikipedia.org/wiki/Ranking#Ordinal_ranking_.28.221234.22_ranking.29, see `StatsBase.ordinalrank` for the ordering we use. If you want more flexibility, checkout `NormalizeQuantiles.sampleranks`.
"""
pseudos(sample) = transpose(hcat([StatsBase.ordinalrank(sample[i,:])./(size(sample,2)+1) for i in axes(sample,1)]...))



@inline _δ(t) = oftype(t, 1e-12)
@inline _safett(t) = clamp(t, _δ(t), one(t) - _δ(t))
@inline _as_tuple(x) = x isa Tuple ? x : (x,)
@inline _as_pxn(p::Integer, U::AbstractMatrix) = (size(U,1) == p) ? U : permutedims(U)


"""
        _kendall_sample(u::AbstractMatrix)

Compute the empirical Kendall sample `W` with entries `W[i] = C_n(U[:,i])`,
where `C_n` is the Deheuvels empirical copula built from the same `u`.

Input and tie handling
- `u` is expected as a `d×n` matrix (columns are observations). This routine first
    applies per-margin ordinal ranks (same policy as `pseudos`) so that the result is
    invariant under strictly increasing marginal transformations and robust to ties.
    Consequently, `_kendall_sample(u) ≡ _kendall_sample(pseudos(u))` (same tie policy).

Returns
- `Vector{Float64}` of length `n` with values in `(0,1)`.
"""
function _kendall_sample(u::AbstractMatrix)
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