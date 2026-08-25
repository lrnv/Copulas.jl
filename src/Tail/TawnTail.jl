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
    TawnCopula{d}(α, weights)
    TawnCopula(d, α, weights)
    TawnCopula(α, weights)
    TawnCopula{d}(dep, asy)
    TawnCopula(d, dep, asy)
    TawnCopula(dep, asy)

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

function TawnTail(d::Int, dep::AbstractVector, asy::AbstractVector)
    α, β = _normalize_asymmetric_subset_components(
        d, dep, asy;
        singleton_parameter=1.0,
        valid_parameter=parameter -> parameter >= one(parameter),
        family="Tawn",
    )
    return TawnTail{eltype(α)}(d, α, β)
end

# Convenience submodel: one full-set logistic component plus singleton remainders.
function TawnTail(α::Real, weights::AbstractVector)
    α >= one(α) || throw(ArgumentError("α must be ≥ 1"))
    d, dep, asy = _expand_fullset_asymmetric_component(α, weights; singleton_parameter=1.0)
    return TawnTail(d, dep, asy)
end

function _tawn_dimension_from_subset_weights(asy::AbstractVector)
    return _dimension_from_asymmetric_subset_weights(asy, "Tawn")
end

function _tawn_convenience_copula(CT, α::Real, weights::AbstractVector)
    tail = TawnTail(α, weights)
    d = _ev_resolve_dimension(CT, tail.d, "weight-vector")
    return ExtremeValueCopula{d}(tail)
end

function (CT::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(α::Real, weights::AbstractVector,)
    return _tawn_convenience_copula(CT, α, weights)
end

function (CT::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(α::Int, weights::AbstractVector,)
    return _tawn_convenience_copula(CT, α, weights)
end

function (::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(d::Int, α::Real, weights::AbstractVector,)
    tail = TawnTail(α, weights)
    d == tail.d || throw(DimensionMismatch(
        "d=$d does not match weight-vector dimension $(tail.d)",
    ))
    return ExtremeValueCopula{d}(tail)
end

function (CT::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(dep::AbstractVector, asy::AbstractVector,)
    inferred = _tawn_dimension_from_subset_weights(asy)
    d = _ev_resolve_dimension(CT, inferred, "subset")
    return ExtremeValueCopula{d}(TawnTail(d, dep, asy))
end

function (::Type{<:ExtremeValueCopula{D,<:TawnTail} where D})(d::Int, dep::AbstractVector, asy::AbstractVector,)
    return ExtremeValueCopula{d}(TawnTail(d, dep, asy))
end

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

    subsets = _nonempty_subsets(tail.d)
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
    logS = LogExpFunctions.logsumexp(logterms)

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

    subsets = _nonempty_subsets(tail.d)
    expected_sign = isodd(length(I)) ? 1 : -1
    return _sum_component_partials(length(subsets), expected_sign) do j
        _tawn_component_partial_signlog(
            tail.α[j], @view(tail.β[:, j]), subsets[j], x, I,
        )
    end
end

function _tawn_rand_multivariate!(rng::Distributions.AbstractRNG, tail::TawnTail, X::AbstractMatrix{T},) where {T<:Real}
    d, n = size(X)
    d == tail.d || throw(DimensionMismatch("output dimension does not match Tawn tail dimension",))

    return _rand_subset_components!(
        rng, X, tail.α, tail.β, isone,
        (dimension, α) -> ExtremeValueCopula(dimension, LogTail(α));
        family="Tawn",
    )
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:TawnTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch("output dimension does not match copula dimension",))
    return _tawn_rand_multivariate!(rng, C.tail, X)
end
