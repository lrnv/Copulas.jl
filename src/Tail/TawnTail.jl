"""
    TawnTail(d, dep, asy)
    TawnTail(α, weights)

Multivariate asymmetric-logistic stable tail dependence function. The full
subset representation follows Tawn's multivariate extreme-value construction
[tawn1990multivariate](@cite):

```math
\\ell(x)
=
\\sum_{\\varnothing\\ne C\\subseteq\\{1,\\ldots,d\\}}
\\left[
\\sum_{i\\in C}(\\beta_{i,C}x_i)^{\\alpha_C}
\\right]^{1/\\alpha_C},
```

with `α_C ≥ 1`, `β_{i,C} ≥ 0`, `β_{i,C}=0` for `i ∉ C`, and

```math
\\sum_{C\\ni i}\\beta_{i,C}=1
```

for every margin.

`TawnTail(d, dep, asy)` exposes the full subset model. `TawnTail(α, weights)`
is a Copulas.jl convenience parameterization with one full-set logistic
component plus singleton remainders; it is a structured submodel of the same
valid Tawn representation, not a separate literature family.

References:

* [tawn1988bivariate](@cite) for the bivariate precursor.
* [tawn1990multivariate](@cite) for the multivariate model.
"""
struct TawnTail{T} <: Tail
    d::Int
    α::Vector{T}
    β::Matrix{T}
end

"""
    TawnCopula(α, weights)
    TawnCopula(d, dep, asy)

Construct a Tawn asymmetric-logistic extreme-value copula.

`TawnCopula(α, weights)` is a convenience submodel with one full-set logistic
component plus singleton remainders; `length(weights)` determines `d`.

`TawnCopula(d, dep, asy)` exposes the full subset model. It requires one
dependence parameter for each non-singleton subset,
`length(dep) = 2^d-d-1`, and one asymmetry-weight vector for each nonempty
subset, `length(asy) = 2^d-1`. The weights involving each margin must sum to
one.
"""
const TawnCopula{d,T} = ExtremeValueCopula{d,TawnTail{T}}

function _tawn_subsets(d::Int)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    subsets = Vector{Vector{Int}}()
    for k in 1:d
        for C in Combinatorics.combinations(1:d, k)
            push!(subsets, collect(C))
        end
    end
    return subsets
end

function TawnTail(d::Int, dep::AbstractVector, asy::AbstractVector)
    subsets = _tawn_subsets(d)
    m = length(subsets)

    length(dep) == m - d || throw(DimensionMismatch("dep must contain one parameter for each non-singleton subset: expected $(m-d)",))
    length(asy) == m || throw(DimensionMismatch("asy must contain one weight vector for each nonempty subset: expected $m",))

    vals = Any[1.0]
    append!(vals, dep)
    for w in asy
        w isa AbstractVector || throw(ArgumentError("each asymmetry component must be an AbstractVector",))
        append!(vals, w)
    end
    T = promote_type(Float64, map(typeof, vals)...)

    α = ones(T, m)
    @inbounds for j in (d + 1):m
        a = T(dep[j - d])
        a >= one(T) || throw(ArgumentError("each non-singleton dependence parameter must be ≥ 1",))
        α[j] = a
    end

    β = zeros(T, d, m)
    @inbounds for (j, C) in enumerate(subsets)
        w = asy[j]
        length(w) == length(C) || throw(DimensionMismatch("asy[$j] must have length $(length(C)) for subset $(Tuple(C))",))
        for (a, i) in enumerate(C)
            wij = T(w[a])
            zero(T) <= wij <= one(T) || throw(ArgumentError("all asymmetry weights must lie in [0,1]",))
            β[i, j] = wij
        end
    end

    tol = 64 * eps(T)
    @inbounds for i in 1:d
        rowsum = sum(@view β[i, :])
        abs(rowsum - one(T)) <= tol * max(one(T), abs(rowsum)) ||
            throw(ArgumentError(
                "asymmetry weights for margin $i must sum to one; got $rowsum",
            ))
    end

    return TawnTail{T}(d, α, β)
end

# Convenience submodel: one full-set logistic component plus singleton remainders.
function TawnTail(α::Real, weights::AbstractVector)
    d = length(weights)
    d >= 2 || throw(ArgumentError("weights must contain at least two entries"))

    subsets = _tawn_subsets(d)
    m = length(subsets)
    T = promote_type(Float64, typeof(α), eltype(weights))
    a = T(α)
    a >= one(T) || throw(ArgumentError("α must be ≥ 1"))

    w = T.(weights)
    all(v -> zero(T) <= v <= one(T), w) || throw(ArgumentError("all full-set weights must lie in [0,1]"))

    dep = ones(T, m - d)
    dep[end] = a

    asy = Vector{Vector{T}}(undef, m)
    for (j, C) in enumerate(subsets)
        asy[j] = zeros(T, length(C))
    end

    for i in 1:d
        asy[i][1] = one(T) - w[i]
    end
    asy[end] .= w

    return TawnTail(d, dep, asy)
end

function (::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(α::Real, weights::AbstractVector,)
    tail = TawnTail(α, weights)
    return ExtremeValueCopula(tail.d, tail)
end
function (::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(α::Int, weights::AbstractVector,)
    tail = TawnTail(α, weights)
    return ExtremeValueCopula(tail.d, tail)
end

(::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(d::Int, dep::AbstractVector, asy::AbstractVector,) = ExtremeValueCopula(d, TawnTail(d, dep, asy))

Distributions.params(tail::TawnTail) = (α = tail.α, β = tail.β)
_is_valid_in_dim(tail::TawnTail, d::Int) = d == tail.d

function _tawn_component_stdf(α, βcol, C, x)
    T = promote_type(typeof(α), eltype(x), eltype(βcol))
    scale = zero(T)

    @inbounds for i in C
        scale = max(scale, βcol[i] * x[i])
    end
    iszero(scale) && return zero(scale)

    s = zero(scale)
    @inbounds for i in C
        y = βcol[i] * x[i] / scale
        s += y^α
    end
    return scale * s^(inv(α))
end

function ℓ(tail::TawnTail, x)
    length(x) == tail.d || throw(DimensionMismatch("input dimension does not match Tawn tail dimension",))

    subsets = _tawn_subsets(tail.d)
    T = promote_type(eltype(x), eltype(tail.α), eltype(tail.β))
    out = zero(T)

    @inbounds for j in eachindex(subsets)
        out += _tawn_component_stdf(
            tail.α[j],
            @view(tail.β[:, j]),
            subsets[j],
            x,
        )
    end
    return out
end

@inline function _tawn_logsumexp(logs::AbstractVector)
    isempty(logs) && return -Inf
    m = maximum(logs)
    isinf(m) && return m
    return m + log(sum(exp(v - m) for v in logs))
end

function _tawn_component_partial_signlog(α::Real, βcol, C, x, I::Tuple{Vararg{Int}},)
    k = length(I)
    k > 0 || throw(ArgumentError("partial block must be nonempty"))

    all(i -> i in C && βcol[i] > 0, I) || return 0, -Inf

    if α == 1
        return k == 1 ? (1, log(float(βcol[only(I)]))) : (0, -Inf)
    end

    all(i -> x[i] > 0, I) || return 0, -Inf

    logterms = Float64[]
    @inbounds for i in C
        yi = float(βcol[i]) * float(x[i])
        yi > 0 && push!(logterms, float(α) * log(yi))
    end
    isempty(logterms) && return 0, -Inf
    logS = _tawn_logsumexp(logterms)

    logcoef = 0.0
    @inbounds for j in 1:(k - 1)
        c = 1.0 - j * float(α)
        iszero(c) && return 0, -Inf
        logcoef += log(abs(c))
    end

    logprod = 0.0
    @inbounds for i in I
        logprod += float(α) * log(float(βcol[i]))
        logprod += (float(α) - 1.0) * log(float(x[i]))
    end

    logabs = logcoef + (inv(float(α)) - k) * logS + logprod
    sign = isodd(k) ? 1 : -1
    return sign, logabs
end

function _ellpartial_signlog(tail::TawnTail, x, I::Tuple{Vararg{Int}},)
    isempty(I) && return 1, log(float(ℓ(tail, x)))

    subsets = _tawn_subsets(tail.d)
    logs = Float64[]
    expected_sign = isodd(length(I)) ? 1 : -1

    @inbounds for j in eachindex(subsets)
        sign, logabs = _tawn_component_partial_signlog(tail.α[j], @view(tail.β[:, j]), subsets[j], x, I,)
        sign == 0 && continue
        sign == expected_sign || throw(ArgumentError("unexpected Tawn component partial sign",))
        push!(logs, logabs)
    end

    isempty(logs) && return 0, -Inf
    return expected_sign, _tawn_logsumexp(logs)
end

_ellpartial_signlog(tail::TawnTail, x, I::AbstractVector{<:Integer}) = _ellpartial_signlog(tail, x, Tuple(I))

function ellpartial(tail::TawnTail, x, I::Tuple{Vararg{Int}})
    isempty(I) && return ℓ(tail, x)
    sign, logabs = _ellpartial_signlog(tail, x, I)
    sign == 0 && return zero(float(first(x)))
    return sign * exp(logabs)
end

ellpartial(tail::TawnTail, x, I::AbstractVector{<:Integer}) = ellpartial(tail, x, Tuple(I))

function _tawn_rand_multivariate!(rng::Distributions.AbstractRNG, tail::TawnTail, X::AbstractMatrix{T},) where {T<:Real}
    d, n = size(X)
    d == tail.d || throw(DimensionMismatch("output dimension does not match Tawn tail dimension",))

    subsets = _tawn_subsets(d)

    # Work on unit-Fréchet margins. Independent max-stable components add
    # their exponent functions; componentwise maxima therefore recover the
    # complete Tawn exponent.
    Z = zeros(Float64, d, n)

    @inbounds for j in eachindex(subsets)
        C = subsets[j]
        α = tail.α[j]
        active = [i for i in C if tail.β[i, j] > 0]
        isempty(active) && continue

        # α = 1 is a linear exponent contribution and hence independent
        # across the active coordinates. A one-coordinate component is also
        # simply unit-Fréchet regardless of α.
        if α == 1 || length(active) == 1
            for i in active
                βij = Float64(tail.β[i, j])
                for col in 1:n
                    candidate = βij / Random.randexp(rng)
                    if candidate > Z[i, col]
                        Z[i, col] = candidate
                    end
                end
            end
            continue
        end

        k = length(active)
        Clog = ExtremeValueCopula(k, LogTail(α))
        U = Random.rand(rng, Clog, n)

        for (a, i) in enumerate(active)
            βij = Float64(tail.β[i, j])
            for col in 1:n
                candidate = βij / (-log(Float64(U[a, col])))
                if candidate > Z[i, col]
                    Z[i, col] = candidate
                end
            end
        end
    end

    @inbounds for i in 1:d, col in 1:n
        zi = Z[i, col]
        zi > 0 || throw(ArgumentError("Tawn weights leave margin $i without a positive spectral component",))
        X[i, col] = T(exp(-inv(zi)))
    end

    return X
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:TawnTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch("output dimension does not match copula dimension",))
    return _tawn_rand_multivariate!(rng, C.tail, X)
end
