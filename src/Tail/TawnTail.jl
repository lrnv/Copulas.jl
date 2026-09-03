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
    function TawnTail(d::Int, dep::AbstractVector, asy::AbstractVector)
        α, β = _normalize_asymmetric_subset_components(
            d, dep, asy;
            singleton_parameter=1.0,
            valid_parameter=parameter -> parameter >= one(parameter),
            family="Tawn",
        )

        component_is_active(j) = !isone(α[j]) && count(!iszero, @view β[:, j]) > 1
        non_singletons = (d + 1):length(α)
        return new{eltype(α)}(d, α, β)
    end
end

@inline _tawn_component_is_active(tail::TawnTail, j) =
    !isone(tail.α[j]) && count(!iszero, @view tail.β[:, j]) > 1

function _tawn_is_fullset_logistic(tail::TawnTail)
    fullset = lastindex(tail.α)
    preceding = (tail.d + 1):(fullset - 1)
    any(j -> _tawn_component_is_active(tail, j), preceding) && return false
    return all(isone, @view tail.β[:, fullset])
end

@inline function limit_kind(tail::TawnTail, ::Val{d}) where {d}
    d == tail.d || return NO_LIMIT
    non_singletons = (tail.d + 1):lastindex(tail.α)
    any(j -> _tawn_component_is_active(tail, j), non_singletons) || return Π_LIMIT

    fullset = lastindex(tail.α)
    return _tawn_is_fullset_logistic(tail) && isinf(tail.α[fullset]) ? M_LIMIT : NO_LIMIT
end

function tail_measure_style(tail::TawnTail)
    for j in (tail.d + 1):lastindex(tail.α)
        _tawn_component_is_active(tail, j) && isinf(tail.α[j]) &&
            return NonAbsolutelyContinuousMeasure()
    end
    return AbsolutelyContinuousMeasure()
end

"""
    TawnCopula{d}(α, weights)
    TawnCopula(d, α, weights)
    TawnCopula{d}(dep, asy)
    TawnCopula(d, dep, asy)

Construct a Tawn asymmetric-logistic extreme-value copula.

`TawnCopula{d}(α, weights)` is a convenience submodel with one full-set
logistic component plus singleton remainders.

`TawnCopula(d, dep, asy)` exposes the full subset model. It requires one
dependence parameter for each non-singleton subset,
`length(dep) = 2^d-d-1`, and one asymmetry-weight vector for each nonempty
subset, `length(asy) = 2^d-1`. The weights involving each margin must sum to
one.
"""
const TawnCopula{d,T} = ExtremeValueCopula{d,TawnTail{T}}

# Convenience submodel: one full-set logistic component plus singleton remainders.
function TawnTail(α::TA, weights::AbstractVector{TW}) where {TA<:Real,TW<:Real}
    T = promote_type(Float64, TA, TW)
    tail = TawnTail(_expand_fullset_asymmetric_component(α, weights; singleton_parameter=1.0)...)
    return tail::TawnTail{T}
end

TawnTail(α::Real, weights::AbstractVector) =
    TawnTail(_expand_fullset_asymmetric_component(α, weights; singleton_parameter=1.0)...)

TawnTail(dep::AbstractVector, asy::AbstractVector) =
    TawnTail(trailing_zeros(length(asy) + 1), dep, asy)

Distributions.params(tail::TawnTail) = (α = tail.α, β = tail.β)
_is_valid_in_dim(tail::TawnTail, d::Int) = d == tail.d

# The full subset parameterization does not yet expose an unconstrained fitting
# map. Do not advertise the generic MLE fallback until that map is implemented.
_available_fitting_methods(::Type{<:ExtremeValueCopula{D,<:TawnTail}}, d) where {D} = ()

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

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:TawnTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    kind = limit_kind(C.tail, Val(d))
    kind === Π_LIMIT && return Random.rand!(rng, X)
    kind === M_LIMIT && return _rand_M!(rng, X)
    
    tail = C.tail
    return _rand_subset_components!(
        rng, X, tail.α, tail.β, isone,
        (dimension, α) -> ExtremeValueCopula(dimension, LogTail(α));
        family="Tawn",
    )
end
