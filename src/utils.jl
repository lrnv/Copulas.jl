# Three small utility functions.

@inline _δ(t::AbstractFloat) = eps(one(t))
@inline _δ(t::ForwardDiff.Dual) = oftype(t, eps(one(ForwardDiff.value(t))))
@inline _δ(t::Real) = eps(float(one(t)))
@inline _safett(t) = clamp(t, _δ(t), one(t) - _δ(t))

# Replace one coordinate while preserving the container shape and
# promoting the element type when ForwardDiff introduces a Dual.
@inline _replace_coordinate(x::Tuple, i::Int, xi) = ntuple(j -> j == i ? xi : x[j], length(x))

function _replace_coordinate(x::AbstractVector, i::Int, xi)
    T = promote_type(eltype(x), typeof(xi))
    y = similar(x, T)
    copyto!(y, x)
    y[i] = xi
    return y
end

"""
    _mixed_partial(f, x, I)

Evaluate the mixed partial derivative of `f` at `x` with respect to the ordered
coordinates in `I`, nesting `ForwardDiff.derivative` once per coordinate. An
empty index set evaluates `f(x)`. This internal shared fallback powers generic
conditioning and stable-tail derivatives; callers requiring non-AD numerical
kernels or singular derivatives must provide a specialized route.

Repeated indices request repeated differentiation in that coordinate. Indices
must address coordinates of `x`; this low-level helper performs no semantic
validation of a derivative's existence.

See also: [`_partial_cdf`](@ref), [`_ellpartial_signlog`](@ref),
[`ellpartial`](@ref).
"""
function _mixed_partial(f, x, I::Tuple{Vararg{Int}})
    isempty(I) && return f(x)
    i = first(I)
    return ForwardDiff.derivative(
        xi -> _mixed_partial(f, _replace_coordinate(x, i, xi), Base.tail(I),),
        x[i],
    )
end

_mixed_partial(f, x, I::AbstractVector{<:Integer}) = _mixed_partial(f, x, Tuple(I))

function _nonempty_subsets(d::Int)
    d >= 1 || throw(ArgumentError("dimension must be positive"))
    return [collect(S) for k in 1:d for S in Combinatorics.combinations(1:d, k)]
end

_invmono(f; tol=1e-8, θmax=1e6, a=0.0, b=1.0) = begin
    fa,fb = f(a), f(b)
    iszero(fa) && return a
    iszero(fb) && return b
    while fb ≤ 0 && b < θmax
        b = min(2b, θmax); fb = f(b)
        iszero(fb) && return b
        !isfinite(fb) && (b = θmax; break)
    end
    (fa < 0 && fb > 0) || error("Could not bound root at [0, $θmax].")
    Roots.find_zero(f, (a,b), Roots.Brent(); atol=tol, rtol=tol)
end

"""
    _falling_factorial(x, k)

Compute `x * (x - 1) * ⋯ * (x - k + 1)` without forming factorials or
using `loggamma`. The implementation also applies to non-integer `x`.
"""
function _falling_factorial(x, k::Integer)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    result = one(x)
    @inbounds for j in 0:(k - 1)
        result *= x - j
    end
    return result
end

"""Multiply `x` by `k!` without first forming an integer factorial."""
function _mul_factorial(x, k::Integer)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    @inbounds for j in 2:k
        x *= j
    end
    return x
end

"""Divide `x` by `k!` without first forming an integer factorial."""
function _div_factorial(x, k::Integer)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    @inbounds for j in 2:k
        x /= j
    end
    return x
end

"""Compute `x * (x + 1) * ⋯ * (x + k - 1)` directly."""
function _rising_factorial(x, k::Integer)
    k ≥ 0 || throw(ArgumentError("k must be non-negative"))
    result = one(x)
    @inbounds for j in 0:(k - 1)
        result *= x + j
    end
    return result
end

"""
    taylor(f::F, x₀, d::Int) where {F}

Compute the Taylor series expansion of the function `f` around the point `x₀` up to order `d`, and gives you back the derivatives as a vector of length d+1. (first value is f(x₀)).

# Arguments
- `f`: A function to be expanded.
- `x₀`: The point around which to expand the Taylor series.
- `d`: The order up to which the Taylor series is computed.

# Returns
A tuple with value ``(f(x₀), f'(x₀),...,f^{(d)}(x₀))``.
"""
function taylor(f::F, x₀, d::Int) where {F}
    x = if x₀ isa Real
        seed = fill(zero(x₀), d + 1)
        seed[1] = x₀
        d > 0 && (seed[2] = one(x₀))
        TaylorSeries.Taylor1(seed)
    else
        x₀ + TaylorSeries.Taylor1(eltype(x₀), d)
    end
    rez = f(x).coeffs
    p = length(rez)
    p == d + 1 && return rez
    if p < d + 1
        v = fill(zero(eltype(rez)), d + 1)
        v[1:p] .= rez
        return v
    end
    return rez[1:(d + 1)]
end

# Stable evaluations of W(exp(logx)) and W₋₁(-exp(logx)). They avoid forming
# arguments that overflow or underflow close to independence in generators
# whose first-derivative inverses have a Lambert-W closed form.
function _lambertw_exp(logx::T) where {T<:AbstractFloat}
    isinf(logx) && return logx > 0 ? T(Inf) : zero(T)
    logx <= log(floatmax(T)) && return LambertW.lambertw(exp(logx))

    w = logx - log(logx)
    for _ in 1:4
        w -= (w + log(w) - logx) / (one(T) + inv(w))
    end
    return w
end

function _lambertwm1_negexp(logabsx::T) where {T<:AbstractFloat}
    logabsx == T(-Inf) && return T(-Inf)
    logabsx = min(logabsx, -one(T))
    logabsx >= log(floatmin(T)) && return LambertW.lambertw(-exp(logabsx), -1)

    target = -logabsx
    y = target + log(target)
    for _ in 1:4
        y -= (y - log(y) - target) / (one(T) - inv(y))
    end
    return -y
end



"""
    pseudos(sample; ties=:average, rng=Random.default_rng(), weights=nothing)

Compute pseudo-observations from a `d×n` sample, with variables in rows and
observations in columns.

Each row is replaced by its ranks divided by `n+1`, so every output is strictly
inside `(0,1)`. The transformation is invariant under strictly increasing
changes of each margin and returns a newly allocated floating-point matrix.

The `ties` keyword controls equal observations. Supported values are `:average`
(the default), `:first`, `:last`, `:min`, `:max`, and `:random`. Average ranks
preserve ties and make the result invariant to observation order. `:first` is
the ordinal convention used by Copulas.jl before version 1.0; `:last` reverses
that order within each tied group. `:random` randomly assigns the available
ordinal ranks within each tied group and uses `rng`; deterministic methods do
not consume it.

Choosing a rank convention does not by itself make continuous-margin fitting
or hypothesis-testing procedures valid for genuinely discrete data.

`weights` gives one non-negative, finite weight per observation, not all zero,
and ranks each margin by weighted mass. The weights are normalized to sum to
the number of observations `n` first, so the result is invariant to their
scale and unit weights reproduce the unweighted ranks exactly. The ranks are
computed and returned in the floating-point type that the sample and the
weights promote to. Observation `i`
then receives the mean rank of its copies in the sample where every observation
`j` is repeated `weights[j]` times, under the same tie convention, divided by
`n + 1`. A zero-weight observation contributes no mass and is placed at the
weighted empirical distribution function of its value.

# Example
```julia
X = [30 10 20; 4 6 5]
pseudos(X) == [0.75 0.25 0.5; 0.25 0.75 0.5]

# Integer weights that sum to `n` are counts: the ranks are those of the
# sample where each observation is repeated that many times.
X = [30 10 20 40; 4 6 5 7]
kept = [1, 3, 4]
pseudos(X; weights=[2, 0, 1, 1])[:, kept] == pseudos([30 30 20 40; 4 4 5 7])[:, kept]
```

See also: [`EmpiricalCopula`](@ref), [`BetaCopula`](@ref),
[`CheckerboardCopula`](@ref).
"""
function pseudos(sample::AbstractMatrix; ties::Symbol=:average,
        rng::Random.AbstractRNG=Random.default_rng(), weights=nothing)
    ties in _PSEUDO_TIE_METHODS || throw(ArgumentError(
        "unsupported tie method :$ties; expected one of $(_PSEUDO_TIE_METHODS)"))
    return _pseudos(sample, Val(ties), rng, _fit_weights(weights, size(sample, 2)))
end

_pseudos(sample::AbstractMatrix, tie_method::Val, rng::Random.AbstractRNG, ::Nothing) =
    _pseudos(sample, tie_method, rng)
function _pseudos(sample::AbstractMatrix, tie_method::Val, rng::Random.AbstractRNG,
        weights::AbstractVector)
    d, n = size(sample)
    T = float(promote_type(eltype(sample), eltype(weights)))
    U = Matrix{T}(undef, d, n)
    tmp_idx = Vector{Int}(undef, n)
    @inbounds for i in 1:d
        x = @view sample[i, :]
        ranks = @view U[i, :]
        sortperm!(tmp_idx, x; by=identity, alg=Base.Sort.DEFAULT_STABLE)
        _assign_weighted_pseudoranks!(ranks, x, tmp_idx, weights, tie_method, rng)
    end
    return U
end

function _pseudos(sample::AbstractMatrix, tie_method::Val,
        rng::Random.AbstractRNG)
    d, n = size(sample)
    T = float(eltype(sample))
    U = Matrix{T}(undef, d, n)
    tmp_idx = Vector{Int}(undef, n)
    @inbounds for i in 1:d
        x = @view sample[i, :]
        ranks = @view U[i, :]
        _pseudoranks!(ranks, x, tie_method, rng, tmp_idx)
    end
    return U
end

const _PSEUDO_TIE_METHODS = (:average, :first, :last, :min, :max, :random)
const _GROUPED_PSEUDO_TIE_METHOD = Union{
    Val{:average}, Val{:last}, Val{:min}, Val{:max}, Val{:random},
}

@inline function _pseudoranks!(ranks::AbstractVector, x::AbstractVector, tie_method::Val,
        rng::Random.AbstractRNG, order::Vector{Int})
    sortperm!(order, x; by=identity, alg=Base.Sort.DEFAULT_STABLE)
    return _assign_pseudoranks!(ranks, x, order, tie_method, rng)
end

@inline function _assign_pseudoranks!(ranks::AbstractVector{T}, ::AbstractVector,
        order::Vector{Int}, ::Val{:first}, ::Random.AbstractRNG) where {T}
    denominator = T(length(order) + 1)
    @inbounds for (rank, index) in enumerate(order)
        ranks[index] = T(rank) / denominator
    end
    return ranks
end

@inline function _assign_pseudoranks!(ranks::AbstractVector, x::AbstractVector,
        order::Vector{Int}, tie_method::_GROUPED_PSEUDO_TIE_METHOD,
        rng::Random.AbstractRNG)
    n = length(order)
    denominator = eltype(ranks)(n + 1)
    first = 1
    while first <= n
        index = @inbounds order[first]
        if first == n || @inbounds(x[order[first + 1]] != x[index])
            @inbounds ranks[index] = eltype(ranks)(first) / denominator
            first += 1
            continue
        end

        last = first + 1
        @inbounds while last < n && x[order[last + 1]] == x[index]
            last += 1
        end
        _assign_tie_group!(ranks, order, first, last, denominator, tie_method, rng)
        first = last + 1
    end
    return ranks
end

@inline function _assign_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        first::Int, last::Int, denominator::T, ::Val{:average},
        ::Random.AbstractRNG) where {T}
    rank = ((T(first) + T(last)) / T(2)) / denominator
    @inbounds for k in first:last
        ranks[order[k]] = rank
    end
    return nothing
end

@inline function _assign_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        first::Int, last::Int, denominator::T, ::Val{:min},
        ::Random.AbstractRNG) where {T}
    rank = T(first) / denominator
    @inbounds for k in first:last
        ranks[order[k]] = rank
    end
    return nothing
end

@inline function _assign_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        first::Int, last::Int, denominator::T, ::Val{:max},
        ::Random.AbstractRNG) where {T}
    rank = T(last) / denominator
    @inbounds for k in first:last
        ranks[order[k]] = rank
    end
    return nothing
end

@inline function _assign_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        first::Int, last::Int, denominator::T, ::Val{:last},
        ::Random.AbstractRNG) where {T}
    @inbounds for k in first:last
        ranks[order[k]] = T(last - (k - first)) / denominator
    end
    return nothing
end

@inline function _assign_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        first::Int, last::Int, denominator::T, ::Val{:random},
        rng::Random.AbstractRNG) where {T}
    Random.shuffle!(rng, @view order[first:last])
    @inbounds for k in first:last
        ranks[order[k]] = T(k) / denominator
    end
    return nothing
end

# Weighted ranks. Observation `k` takes the mean rank of its copies in the
# sample where every observation `j` is repeated `w[j]` times, under the same
# tie convention: `before + (w[k] + 1) / 2`, where `before` is the weight of
# the copies that the convention places below it. The denominator is the total
# weight plus one. Unit weights reproduce the unweighted ranks exactly. A
# zero-weight tie group has no copies, so `:min` and `:max` place it at the
# weighted empirical distribution function of its value, which keeps every
# rank strictly inside (0, 1).
function _assign_weighted_pseudoranks!(ranks::AbstractVector{T}, x::AbstractVector,
        order::Vector{Int}, w::AbstractVector, tie_method::Val,
        rng::Random.AbstractRNG) where {T}
    n = length(order)
    denominator = T(sum(w)) + one(T)
    before = zero(T)
    first = 1
    @inbounds while first <= n
        index = order[first]
        last = first
        while last < n && x[order[last + 1]] == x[index]
            last += 1
        end
        block = zero(T)
        for k in first:last
            block += T(w[order[k]])
        end
        _assign_weighted_tie_group!(ranks, order, w, first, last, before, block,
                                    denominator, tie_method, rng)
        before += block
        first = last + 1
    end
    return ranks
end

@inline function _assign_weighted_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        ::AbstractVector, first::Int, last::Int, before::T, block::T, denominator::T,
        ::Val{:average}, ::Random.AbstractRNG) where {T}
    rank = (before + (block + one(T)) / T(2)) / denominator
    @inbounds for k in first:last
        ranks[order[k]] = rank
    end
    return nothing
end

@inline function _assign_weighted_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        ::AbstractVector, first::Int, last::Int, before::T, block::T, denominator::T,
        ::Val{:min}, ::Random.AbstractRNG) where {T}
    rank = (before + (iszero(block) ? one(T) / T(2) : one(T))) / denominator
    @inbounds for k in first:last
        ranks[order[k]] = rank
    end
    return nothing
end

@inline function _assign_weighted_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        ::AbstractVector, first::Int, last::Int, before::T, block::T, denominator::T,
        ::Val{:max}, ::Random.AbstractRNG) where {T}
    rank = (before + (iszero(block) ? one(T) / T(2) : block)) / denominator
    @inbounds for k in first:last
        ranks[order[k]] = rank
    end
    return nothing
end

# Ordinal conventions: the copies of one observation are consecutive, in the
# convention's order within the tie group.
@inline function _assign_weighted_ordinal!(ranks::AbstractVector{T}, order::Vector{Int},
        w::AbstractVector, positions, before::T, denominator::T) where {T}
    @inbounds for k in positions
        wk = T(w[order[k]])
        ranks[order[k]] = (before + (wk + one(T)) / T(2)) / denominator
        before += wk
    end
    return nothing
end

@inline function _assign_weighted_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        w::AbstractVector, first::Int, last::Int, before::T, ::T, denominator::T,
        ::Val{:first}, ::Random.AbstractRNG) where {T}
    return _assign_weighted_ordinal!(ranks, order, w, first:last, before, denominator)
end

@inline function _assign_weighted_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        w::AbstractVector, first::Int, last::Int, before::T, ::T, denominator::T,
        ::Val{:last}, ::Random.AbstractRNG) where {T}
    return _assign_weighted_ordinal!(ranks, order, w, last:-1:first, before, denominator)
end

@inline function _assign_weighted_tie_group!(ranks::AbstractVector{T}, order::Vector{Int},
        w::AbstractVector, first::Int, last::Int, before::T, ::T, denominator::T,
        ::Val{:random}, rng::Random.AbstractRNG) where {T}
    Random.shuffle!(rng, @view order[first:last])
    return _assign_weighted_ordinal!(ranks, order, w, first:last, before, denominator)
end



function _require_tie_free_rows(sample::AbstractMatrix, operation::AbstractString)
    for row in axes(sample, 1)
        allunique(@view sample[row, :]) || throw(ArgumentError(
            "$operation requires tie-free margins; ties were detected in margin $row. " *
            "If deliberate tie breaking is justified, preprocess the data with pseudos " *
            "using :first, :last, or :random before construction."))
    end
    return nothing
end

# Pairwise component metrics applied to (n,d)-shaped matrices:
"""
    corblomqvist(X::AbstractMatrix)
    corblomqvist(C::Copula)

Return the symmetric matrix of pairwise Blomqvist beta coefficients. Data follow
the `StatsBase` convention (`n × d`, observations in rows); a copula is reduced
to each bivariate margin. Diagonal entries are one.

Data columns are ranked before observations are split at their median. A column
containing `NaN` produces `NaN` in all corresponding off-diagonal entries.
"""
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

##### Weighted pairwise rank measures.
#
# A fit weight is "how many observations this column counts for" (see
# `_fit_weights`), so each weighted measure below is the measure of the sample
# in which observation j is repeated w[j] times, written so that it extends to
# real weights and is invariant to their scale. They take the package's `d × n`
# orientation and return the `d × d` matrix that `corkendall(U')`,
# `corspearman(U')` and `corblomqvist(U')` return. `_rank_measure` selects the
# unweighted `StatsBase` function or the weighted one from the weights.
_rank_measure(::Val{:itau}, U::AbstractMatrix, ::Nothing) = StatsBase.corkendall(U')
_rank_measure(::Val{:irho}, U::AbstractMatrix, ::Nothing) = StatsBase.corspearman(U')
_rank_measure(::Val{:ibeta}, U::AbstractMatrix, ::Nothing) = corblomqvist(U')
_rank_measure(::Val{:itau}, U::AbstractMatrix, w::AbstractVector) = _weighted_corkendall(U, w)
_rank_measure(::Val{:irho}, U::AbstractMatrix, w::AbstractVector) = _weighted_corspearman(U, w)
_rank_measure(::Val{:ibeta}, U::AbstractMatrix, w::AbstractVector) = _weighted_corblomqvist(U, w)

# Prefix sums over the measure's element type, for the Kendall sweep.
struct _Fenwick{T}
    bit::Vector{T}
end
@inline function _fenwick_add!(F::_Fenwick, i::Int, δ)
    n = length(F.bit)
    @inbounds while i <= n
        F.bit[i] += δ
        i += i & -i
    end
    return nothing
end
@inline function _fenwick_sum(F::_Fenwick{T}, i::Int) where {T}
    s = zero(T)
    @inbounds while i > 0
        s += F.bit[i]
        i -= i & -i
    end
    return s
end

# Dense ranks of `x` under `==`, and their number.
function _dense_ranks(x::AbstractVector)
    order = sortperm(x)
    ranks = Vector{Int}(undef, length(x))
    k = 0
    @inbounds for (position, index) in enumerate(order)
        (position == 1 || x[index] != x[order[position - 1]]) && (k += 1)
        ranks[index] = k
    end
    return ranks, k
end

# Weighted number of pairs tied in `x`: Σ_g (T_g² − Σ_{i∈g} w_i²) / 2 over the
# tie groups g of total weight T_g. The pairs formed by the copies of one
# observation are left out, as they are from the total number of pairs, so
# for integer weights this is the tie count of the replicated sample.
function _weighted_tie_pairs(x::AbstractVector, w::AbstractVector)
    T = promote_type(eltype(x), eltype(w))
    order = sortperm(x)
    n = length(order)
    ties = zero(T)
    first = 1
    @inbounds while first <= n
        last = first
        while last < n && x[order[last + 1]] == x[order[first]]
            last += 1
        end
        if last > first
            total = zero(T)
            squares = zero(T)
            for k in first:last
                wk = w[order[k]]
                total += wk
                squares += abs2(wk)
            end
            ties += (abs2(total) - squares) / 2
        end
        first = last + 1
    end
    return ties
end

# Weighted Kendall's tau-b of one pair,
#
#     τ = Σ_{i<j} w_i w_j sgn(x_i − x_j) sgn(y_i − y_j) / √((W₀ − W₁)(W₀ − W₂)),
#
# with W₀ = Σ_{i<j} w_i w_j and W₁, W₂ the weighted tie pairs of x and y. The
# concordance sum is one O(n log n) sweep over x with a Fenwick tree over the
# ranks of y, every x-tie block being queried before it is inserted. The
# integer arithmetic of `StatsBase.corkendall` is reproduced exactly for unit
# weights, so the two agree bit for bit.
function _weighted_kendall(x::AbstractVector, y::AbstractVector, w::AbstractVector)
    T = promote_type(eltype(x), eltype(y), eltype(w))
    n = length(x)
    (any(isnan, x) || any(isnan, y)) && return T(NaN)
    n <= 1 && return T(NaN)
    order = sortperm(x)
    yranks, m = _dense_ranks(y)
    F = _Fenwick(zeros(T, m))
    seen = zero(T)
    S = zero(T)
    first = 1
    @inbounds while first <= n
        last = first
        while last < n && x[order[last + 1]] == x[order[first]]
            last += 1
        end
        for k in first:last
            r = yranks[order[k]]
            less = _fenwick_sum(F, r - 1)
            greater = seen - _fenwick_sum(F, r)
            S += w[order[k]] * (less - greater)
        end
        for k in first:last
            wk = w[order[k]]
            _fenwick_add!(F, yranks[order[k]], wk)
            seen += wk
        end
        first = last + 1
    end
    W₀ = (abs2(sum(w)) - sum(abs2, w)) / 2
    return S / sqrt((W₀ - _weighted_tie_pairs(x, w)) * (W₀ - _weighted_tie_pairs(y, w)))
end

function _weighted_corkendall(U::AbstractMatrix, w::AbstractVector)
    d = size(U, 1)
    C = Matrix{promote_type(eltype(U), eltype(w))}(LinearAlgebra.I, d, d)
    for j in 1:d, i in 1:(j - 1)
        C[i, j] = C[j, i] = _weighted_kendall(view(U, i, :), view(U, j, :), w)
    end
    return C
end

# Weighted average ranks: a tie block of total weight T preceded by weight B
# takes the rank B + (T + 1) / 2, the mean rank of the block in the sample
# where each observation is repeated w[i] times, which is `tiedrank` for unit
# weights.
function _weighted_average_ranks(x::AbstractVector, w::AbstractVector)
    T = promote_type(eltype(x), eltype(w))
    order = sortperm(x)
    n = length(order)
    ranks = Vector{T}(undef, n)
    before = zero(T)
    first = 1
    @inbounds while first <= n
        last = first
        while last < n && x[order[last + 1]] == x[order[first]]
            last += 1
        end
        block = zero(T)
        for k in first:last
            block += w[order[k]]
        end
        rank = before + (block + one(T)) / 2
        for k in first:last
            ranks[order[k]] = rank
        end
        before += block
        first = last + 1
    end
    return ranks
end

# Weighted Pearson correlation of two vectors.
function _weighted_pearson(x::AbstractVector, y::AbstractVector, w::AbstractVector)
    W = sum(w)
    x̄ = sum(i -> w[i] * x[i], eachindex(w)) / W
    ȳ = sum(i -> w[i] * y[i], eachindex(w)) / W
    sxy = sum(i -> w[i] * (x[i] - x̄) * (y[i] - ȳ), eachindex(w))
    sxx = sum(i -> w[i] * abs2(x[i] - x̄), eachindex(w))
    syy = sum(i -> w[i] * abs2(y[i] - ȳ), eachindex(w))
    return sxy / sqrt(sxx * syy)
end

# Weighted Spearman's rho: the weighted Pearson correlation of the weighted
# average ranks, which for integer weights is the Spearman correlation of the
# replicated sample. It is another reduction than `StatsBase.corspearman`, so
# unit weights reproduce it within an ulp rather than bit for bit.
function _weighted_corspearman(U::AbstractMatrix, w::AbstractVector)
    d = size(U, 1)
    C = Matrix{promote_type(eltype(U), eltype(w))}(LinearAlgebra.I, d, d)
    anynan = [any(isnan, view(U, i, :)) for i in 1:d]
    ranks = [anynan[i] ? eltype(C)[] : _weighted_average_ranks(view(U, i, :), w) for i in 1:d]
    for j in 1:d, i in 1:(j - 1)
        C[i, j] = C[j, i] = (anynan[i] || anynan[j]) ? NaN :
            _weighted_pearson(ranks[i], ranks[j], w)
    end
    return C
end

# Weighted Blomqvist's beta: the weighted average ranks are split at the
# median rank (W + 1) / 2 of the replicated sample and the concordant mass is
# counted, as `corblomqvist` counts concordant observations.
function _weighted_corblomqvist(U::AbstractMatrix, w::AbstractVector)
    d = size(U, 1)
    C = Matrix{promote_type(eltype(U), eltype(w))}(LinearAlgebra.I, d, d)
    W = sum(w)
    h = (W + 1) / 2
    anynan = [any(isnan, view(U, i, :)) for i in 1:d]
    ranks = [anynan[i] ? eltype(C)[] : _weighted_average_ranks(view(U, i, :), w) for i in 1:d]
    for j in 1:d, i in 1:(j - 1)
        if anynan[i] || anynan[j]
            C[i, j] = C[j, i] = NaN
            continue
        end
        c = zero(W)
        @inbounds for k in eachindex(w)
            c += w[k] * ((ranks[i][k] <= h) == (ranks[j][k] <= h))
        end
        C[i, j] = C[j, i] = 2c / W - 1
    end
    return C
end
"""
    corgini(X::AbstractMatrix)
    corgini(C::Copula)

Return the symmetric matrix of pairwise Gini gamma coefficients. Data follow
the `StatsBase` `n × d` orientation; copula entries are computed from bivariate
margins. Diagonal entries are one.

The data estimator is rank based. A column containing `NaN` produces `NaN` in
all corresponding off-diagonal entries.
"""
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
"""
    corentropy(X::AbstractMatrix; k=5, p=Inf, leafsize=32)
    corentropy(C::Copula)

Return pairwise copula entropy as a symmetric matrix with zero diagonal. The
data method uses `n × d` observations and a nearest-neighbor estimator; the
copula method requires bivariate margins with ordinary densities.

For data, `k`, `p` and `leafsize` have the same interpretation and caveats as in
`ι`; columns containing `NaN` propagate `NaN` to the corresponding pairs. The
zero diagonal is conventional and does not invoke a degenerate self-copula
entropy calculation.
"""
function corentropy(X::AbstractMatrix{<:Real}; k::Int=5, p::Real=Inf, leafsize::Int=32)
    # We expect the number of dimension to be the second axes here,
    # contrary to the whole package but to be coherent with
    # StatsBase.corspearman and StatsBase.corkendall.
    m, n = size(X)
    Cnan = Vector{Bool}(undef, n)
    for j in 1:n
        Cnan[j] = any(isnan, @view X[:, j])
    end
    Ucol = [Cnan[j] ? Float64[] : collect(@view X[:, j]) for j in 1:n]
    H  = zeros(Float64, n, n)
    H  = zeros(Float64, n, n)
    Ub = Array{Float64}(undef, 2, m)
    @inbounds for j in 2:n
        if Cnan[j]
            H[:, j] .= NaN
            H[j, :] .= NaN
            H[j, j] = 0.0
            continue
        end
        uj = Ucol[j]
        for i in 1:j-1
            if Cnan[i]
                H[i, j] = NaN
                H[j, i] = NaN
                continue
            end
            ui = Ucol[i]
            Ub[1, :] .= ui; Ub[2, :] .= uj
            H[i, j] = ι(Ub; k=k, p=p, leafsize=leafsize)
        end
    end
    return H
    return H
end
function _cortail(X::AbstractMatrix{<:Real}; t = :lower, method = :SchmidtStadtmueller, p = nothing)
    # We expect the number of dimension to be the second axes here,
    # contrary to the whole package but to be coherent with
    # StatsBase.corspearman and StatsBase.corkendall.
    m, n = size(X)
    n ≥ 2 || throw(ArgumentError("≥ 2 variables (columns) are required."))
    (t === :lower || t === :upper) || throw(ArgumentError("t ∈ {:lower,:upper}"))
    U = t === :upper ? (1 .- Float64.(X)) : Float64.(X)
    anynan = [any(isnan, @view U[:, j]) for j in 1:n]
    p === nothing && (p = 1 / sqrt(m))
    (0 < p < 1) || throw(ArgumentError("p must be in (0,1); hint: p = 1/√m"))

    Lam = Matrix{Float64}(LinearAlgebra.I, n, n)

    if method === :SchmidtStadtmueller
        B = U .<= p
        @inbounds @views for j in 2:n
            anynan[j] && continue
            bj = B[:, j]
            for i in 1:j-1
                if anynan[i]
                    Lam[i,j] = Lam[j,i] = NaN
                else
                    bi = B[:, i]
                    c  = sum(bi .& bj)
                    Lam[i,j] = Lam[j,i] = clamp((c / m) / p, 0.0, 1.0)
                end
            end
        end

    elseif method === :SchmidSchmidt
        pmu = max.(0.0, p .- U)
        S   = Matrix{Float64}(LinearAlgebra.I, n, n)
        @inbounds @views for j in 2:n
            anynan[j] && continue
            y = pmu[:, j]
            for i in 1:j-1
                if anynan[i]
                    S[i,j] = S[j,i] = NaN
                else
                    x = pmu[:, i]
                    S[i,j] = S[j,i] = LinearAlgebra.dot(x, y) / m
                end
            end
        end
        int_over_Pi = (p^2 / 2)^2
        int_over_M  = p^3 / 3
        scale = int_over_M - int_over_Pi
        @inbounds for j in 2:n, i in 1:j-1
            if isfinite(S[i,j])
                Lam[i,j] = Lam[j,i] = clamp((S[i,j] - int_over_Pi) / scale, 0.0, 1.0)
            else
                Lam[i,j] = Lam[j,i] = NaN
            end
        end

    else
        throw(ArgumentError("method must be :SchmidtStadtmueller or :SchmidSchmidt"))
    end
    @inbounds for j in 1:n
        if anynan[j]
            Lam[:, j] .= NaN
            Lam[j, :] .= NaN
            Lam[j, j]  = 1.0
        end
    end
    return Lam
end
"""
    corlowertail(X::AbstractMatrix, method=:SchmidtStadtmueller, p=nothing)
    corlowertail(C::Copula)

Return the symmetric matrix of pairwise lower-tail dependence coefficients.
Data use the `StatsBase` `n × d` orientation and either the
`:SchmidtStadtmueller` threshold estimator or `:SchmidSchmidt` integral
estimator. Copula entries use their bivariate margins.

`p` controls the lower-tail threshold and defaults to `1/√n`. Smaller values
focus farther into the tail but increase sampling variability. Diagonal entries
are one; a data column containing `NaN` propagates `NaN` to its off-diagonal
pairs. Data must already be represented on the uniform scale; apply `pseudos`
to raw continuous margins. Many ties can make a continuous-tail interpretation
unreliable.
"""
corlowertail(X::AbstractMatrix{<:Real}, method = :SchmidtStadtmueller, p=nothing) = _cortail(X; t=:lower, method=method, p=p)

"""
    coruppertail(X::AbstractMatrix, method=:SchmidtStadtmueller, p=nothing)
    coruppertail(C::Copula)

Return the symmetric matrix of pairwise upper-tail dependence coefficients.
Data use the `StatsBase` `n × d` orientation and either the
`:SchmidtStadtmueller` threshold estimator or `:SchmidSchmidt` integral
estimator. Copula entries use their bivariate margins.

`p` controls the upper-tail threshold and defaults to `1/√n`. Smaller values
focus farther into the tail but increase sampling variability. Diagonal entries
are one; a data column containing `NaN` propagates `NaN` to its off-diagonal
pairs. Data must already be represented on the uniform scale; apply `pseudos`
to raw continuous margins. Many ties can make a continuous-tail interpretation
unreliable.
"""
coruppertail(X::AbstractMatrix{<:Real}, method = :SchmidtStadtmueller, p=nothing) = _cortail(X; t=:upper, method=method, p=p)


"""
    _kendall_sample(u::AbstractMatrix)

Compute the empirical Kendall sample `W` with entries `W[i] = C_n(U[:,i])`,
where `C_n` is the Deheuvels empirical copula built from the same `u`.

Input and tie handling
- `u` is expected as a `d×n` matrix (columns are observations).
- Dominance is evaluated directly on `u`, preserving equal values. The result is
  therefore invariant under strictly increasing marginal transformations and
  under permutations of the observations, including when ties are present.
- For any rank transformation that preserves ties, including the default
  `pseudos(u; ties=:average)`, applying that transformation first leaves the
  empirical Kendall sample unchanged.

Returns
- `Vector{Float64}` of length `n` with values in `(0,1)`.
"""
function _kendall_sample(u::AbstractMatrix)
    _, n = size(u)
    W = zeros(Float64, n)
    @inbounds for i in 1:n
        ui = @view u[:, i]
        count_le = 0
        for j in 1:n
            count_le += all(@view(u[:, j]) .≤ ui)
        end
        W[i] = count_le / (n + 1)
    end
    return W
end
# Numeric type carried by a parameter object. Integer and Boolean values are
# structural unless they occur in an array whose element type is the actual
# stored numeric representation.
_parameter_eltype(::Integer) = Union{}
_parameter_eltype(::Bool) = Union{}
_parameter_eltype(x::Real) = typeof(float(x))
_parameter_eltype(x::AbstractArray{<:Real}) = float(eltype(x))
function _parameter_eltype(x::AbstractArray)
    T = Union{}
    for value in x
        S = _parameter_eltype(value)
        S === Union{} && continue
        T = T === Union{} ? S : promote_type(T, S)
    end
    return T
end
_parameter_eltype(x::Distributions.Distribution) = float(Distributions.partype(x))
_parameter_eltype(x::NamedTuple) = _parameter_eltype(values(x))
_parameter_eltype(x::Tuple) = _parameter_eltype(x...)
_parameter_eltype() = Union{}
function _parameter_eltype(x, xs...)
    T = _parameter_eltype(x)
    S = _parameter_eltype(xs...)
    T === Union{} && return S
    S === Union{} && return T
    return promote_type(T, S)
end
_parameter_eltype(x) = _parameter_eltype(Distributions.params(x))
function _sample_eltype(x)
    T = _parameter_eltype(x)
    return T === Union{} ? Float64 : T
end