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

_uppertriangle_flat(mat) = [mat[idx] for idx in CartesianIndices(mat) if idx[1] < idx[2]]
function _uppertriangle_stats(mat)
    # compute the mean and std of the upper triangular part of the matrix (diagonal excluded)
    gen = _uppertriangle_flat(mat)
    return Statistics.mean(gen), length(gen) == 1 ? zero(gen[1]) : Statistics.std(gen), minimum(gen), maximum(gen)
end
#SCHMIDT1 and STADTMULLER (2006)
function tail(U::Matrix{T}; t::Symbol=:upper) where T <: Real
    size(U, 1) == 2 || throw(ArgumentError("U must be a 2×n matrix"))
    m = size(U, 2)
    m ≥ 4 || throw(ArgumentError("At least 4 observations are required"))
    ranks1 = StatsBase.ordinalrank(U[1, :])
    ranks2 = StatsBase.ordinalrank(U[2, :])
    k = max(4, floor(Int, sqrt(m)))
    count_extreme = if t === :upper
        threshold = m - k
        sum((ranks1 .> threshold) .& (ranks2 .> threshold))
    elseif t === :lower
        sum((ranks1 .<= k) .& (ranks2 .<= k))
    else
        throw(ArgumentError("t must be :upper or :lower"))
    end
    return count_extreme / k
end

_invmono(f; tol=1e-8, θmax=1e6, a=0.0, b=1.0) = begin
    fa,fb = f(0.0), f(1.0)
    while fb ≤ 0 && b < θmax
        b = min(2b, θmax); fb = f(b)
        !isfinite(fb) && (b = θmax; break)
    end
    (fa < 0 && fb > 0) || error("Could not bound root at [0, $θmax].")
    Roots.find_zero(f, (a,b), Roots.Brent(); atol=tol, rtol=tol)
end



function β(U::AbstractMatrix)
    # Assumes psuedo-data given. β multivariate (Hofert–Mächler–McNeil, ec. (7))
    d, n = size(U)
    count = sum(j -> all(U[:, j] .<= 0.5) || all(U[:, j] .> 0.5), 1:n)
    h_d = 2.0^(d-1) / (2.0^(d-1) - 1.0)
    return h_d * (count/n - 2.0^(1-d))
end
function τ(U::AbstractMatrix)
    # Sample version of multivariate Kendall's tau for pseudo-data
    d, n = size(U)
    comp = 0
    @inbounds for j in 2:n, i in 1:j-1
        uᵢ = @view U[:, i]; uⱼ = @view U[:, j]
        comp += (all(uᵢ .<= uⱼ) || all(uᵢ .>= uⱼ))
    end
    pc = comp / (n*(n-1)/2)
    return (2.0^d * pc - 2.0) / (2.0^d - 2.0)
end
function ρ(U::AbstractMatrix)
    # Sample version of multivariate Spearman's tau for pseudo-data
    d, n = size(U)
    R = hcat((StatsBase.tiedrank(U[k, :]) for k in 1:d)...)   # n×d
    μ = Statistics.mean(prod(R, dims=2)) / (n + 1)^d          # ≈ E[∏ U_i]
    h = (d + 1) / (2.0^d - (d + 1))
    return h * (2.0^d * μ - 1.0)
end
function γ(U::AbstractMatrix)
    # Assumes pseudo-data given. Multivariate Gini’s gamma (Behboodian–Dolati–Úbeda, 2007)
    d, n = size(U)
    if d == 2
    # Schechtman–Yitzhaki symmetric Gini over ranks (copular invariant)
        r1 = StatsBase.tiedrank(@view U[1, :])
        r2 = StatsBase.tiedrank(@view U[2, :])
        m  = n
        h  = m + 1
        acc = 0.0
        @inbounds @simd for k in 1:m
            acc += abs(r1[k] + r2[k] - h) - abs(r1[k] - r2[k])
        end
        return 2*acc / (m*h)
    else
        @inline _A(u)    = (minimum(u) + max(sum(u) - d + 1, 0.0)) / 2
        @inline _Abar(u) = (1 - maximum(u) + max(1 - sum(u), 0.0)) / 2
        invfac(k::Integer) = exp(-SpecialFunctions.logfactorial(k))
        s = 0.0
        binomf(d,i) = exp(SpecialFunctions.loggamma(d+1) - SpecialFunctions.loggamma(i+1) - SpecialFunctions.loggamma(d-i+1))
        @inbounds for i in 0:d
            s += (isodd(i) ? -1.0 : 1.0) * binomf(d,i) * invfac(i + 1)
        end
        a_d = 1/(d + 1) + 0.5*invfac(d + 1) + 0.5*s
        b_d = 2/3 + 4.0^(1 - d) / 3
        m = 0.0
        @inbounds for j in 1:n
            u = @view U[:, j]
            m += _A(u) + _Abar(u)
        end
        m /= n
        return (m - a_d) / (b_d - a_d)
    end
end
function entropy(U::AbstractMatrix; k::Int=5, p::Real=Inf, leafsize::Int=32)
    # Assumes pseudo-data given. Multivariate copula entropy (L.F. Kozachenko and N.N. Leonenko., 1987)
    d, n = size(U)
    n ≥ k+1 || throw(ArgumentError("n ≥ k+1 is required"))
    (p ≥ 1 || isinf(p)) || throw(ArgumentError("invalid Minkowski norm: p ∈ [1,∞]"))
    any(isnan, U) && return NaN
    X = Array{Float64}(U)
    lp(u, v, p) = (sum(abs.(u .- v) .^ p))^(1/p)
    cheb(u, v)  = maximum(abs.(u .- v))
    function lb_lp(q, lo, hi, p)
        s = 0.0
        @inbounds for t in eachindex(q)
            δ = q[t] < lo[t] ? (lo[t]-q[t]) : (q[t] > hi[t] ? q[t]-hi[t] : 0.0)
            s += δ^p
        end
        return s^(1/p)
    end
    function lb_inf(q, lo, hi)
        m = 0.0
        @inbounds for t in eachindex(q)
            δ = q[t] < lo[t] ? (lo[t]-q[t]) : (q[t] > hi[t] ? q[t]-hi[t] : 0.0)
            m = δ > m ? δ : m
        end
        return m
    end
    nodes = Vector{Vector{Any}}()
    function build(idxs::Vector{Int})
        lo = fill( Inf, d); hi = fill(-Inf, d)
        @inbounds for j in idxs, r in 1:d
            v = X[r, j]
            lo[r] = v < lo[r] ? v : lo[r]
            hi[r] = v > hi[r] ? v : hi[r]
        end
        if length(idxs) ≤ leafsize
            push!(nodes, Any[copy(idxs), 0, 0.0, 0, 0, lo, hi])  # hoja
            return length(nodes)
        end
        spans = hi .- lo
        sd = findmax(spans)[2]
        sv = (lo[sd] + hi[sd]) / 2
        left = Int[]; right = Int[]
        @inbounds for j in idxs
            (X[sd, j] ≤ sv ? push!(left, j) : push!(right, j))
        end
        if isempty(left) || isempty(right)
            ord = sort(idxs; by = j -> X[sd, j]); m = length(ord) ÷ 2
            left  = ord[1:m]; right = ord[m+1:end]; sv = X[sd, ord[m]]
        end
        L = build(left); R = build(right)
        push!(nodes, Any[Int[], sd, sv, L, R, lo, hi])
        return length(nodes)
    end
    root = build(collect(1:n))
    function knn!(q::AbstractVector{<:Real}, selfidx::Int, K::Int,
                  D::Vector{Float64}, I::Vector{Int}, node::Int)
        nd = nodes[node]
        idxs, sd, sv, L, R, lo, hi = nd[1], nd[2], nd[3], nd[4], nd[5], nd[6], nd[7]
        worst = isempty(D) ? Inf : maximum(D)
        lb = isinf(p) ? lb_inf(q, lo, hi) : lb_lp(q, lo, hi, p)
        if length(D) == K && lb ≥ worst
            return
        end
        if sd == 0
            @inbounds for j in idxs
                j == selfidx && continue
                xj = @view X[:, j]
                dj = isinf(p) ? cheb(q, xj) : lp(q, xj, p)
                dj = ifelse(iszero(dj), eps(Float64), dj)
                if length(D) < K
                    push!(D, dj); push!(I, j)
                elseif dj < worst
                    t = findmax(D)[2]; D[t] = dj; I[t] = j
                end
                worst = length(D) == K ? maximum(D) : Inf
            end
            return
        end
        near = (q[sd] ≤ sv) ? L : R
        far  = (q[sd] ≤ sv) ? R : L
        knn!(q, selfidx, K, D, I, near)
        worst = length(D) == K ? maximum(D) : Inf
        if length(D) < K || abs(q[sd] - sv) < worst
            knn!(q, selfidx, K, D, I, far)
        end
    end
    ρ = Vector{Float64}(undef, n)
    @inbounds for j in 1:n
        D = Float64[]; I = Int[]
        q = @view X[:, j]
        knn!(q, j, k, D, I, root)
        ρ[j] = maximum(D)
    end
    ρ .= max.(ρ, eps(Float64))

    #KL: H = -ψ(k)+ψ(n)+log c_{d,p} + (d/n)∑log ρ  ; for L∞, we absorb log c_{d,∞}=d log 2
    H = -SpecialFunctions.digamma(k) + SpecialFunctions.digamma(n)
    if isinf(p)
        H += (d / n) * sum(log.(2 .* ρ))
    else
        logcd = d*log(2*SpecialFunctions.gamma(1 + 1/p)) - SpecialFunctions.loggamma(1 + d/p)
        H += logcd + (d / n) * sum(log.(ρ))
    end
    return H
end

function corblomqvist(X::AbstractMatrix{<:Real})
    # We expect the number of dimension to be the second axes here, 
    # contrary to the whole package but to be coherent with 
    # StatsBase.corspearman and StatsBase.corkendall. 
    n = size(X, 2) 
    C = Matrix{Float64}(LinearAlgebra.I, n, n)
    anynan = Vector{Bool}(undef, n)
    m = size(X, 1)
    h = (m + 1) / 2
    for j = 1:n
        Xj = view(X, :, j)
        anynan[j] = any(isnan, Xj)
        if anynan[j]
            C[:,j] .= NaN
            C[j,:] .= NaN
            C[j,j] = 1
            continue
        end
        xrj = StatsBase.tiedrank(Xj)
        for i = 1:(j-1)
            Xi = view(X, :, i)
            if anynan[i]
                C[i,j] = C[j,i] = NaN
            else
                xri = StatsBase.tiedrank(Xi)
                c = 0
                @inbounds for k in 1:m
                    c += ( (xri[k] <= h) == (xrj[k] <= h) )
                end
                C[i,j] = C[j,i] = 2c/m - 1
            end
        end
    end
    return C
end
function corgini(X::AbstractMatrix{<:Real})
    # We expect the number of dimension to be the second axes here, 
    # contrary to the whole package but to be coherent with 
    # StatsBase.corspearman and StatsBase.corkendall. 
    m, n = size(X)
    C = Matrix{Float64}(LinearAlgebra.I, n, n)
    anynan = Vector{Bool}(undef, n)
    ranks  = Vector{Vector{Float64}}(undef, n)
    for j in 1:n
        Xj = view(X, :, j)
        anynan[j] = any(isnan, Xj)
        ranks[j]  = anynan[j] ? Float64[] : StatsBase.tiedrank(Xj)
        if anynan[j]
            C[:, j] .= NaN
            C[j, :] .= NaN
            C[j, j]  = 1.0
        end
    end
    h = m + 1
    for j in 2:n
        anynan[j] && continue
        rj = ranks[j]
        for i in 1:j-1
            if anynan[i]
                C[i, j] = C[j, i] = NaN
            else
                ri  = ranks[i]
                acc = 0.0
                @inbounds @simd for k in 1:m
                    acc += abs(ri[k] + rj[k] - h) - abs(ri[k] - rj[k])
                end
                C[i, j] = C[j, i] = 2*acc / (m*h)
            end
        end
    end
    return C
end