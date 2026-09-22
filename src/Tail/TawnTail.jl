"""
    TawnTail(d, dep, asy)
    TawnTail(dep, weights₁, ..., weights_d)
    TawnTail(α, weights)
    TawnCopula{d}(dep, weights₁, ..., weights_d)
    TawnCopula{d}(α, weights)
    TawnCopula(d, dep, asy)
    TawnCopula(d, α, weights)

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

The canonical parameterization stores one dependence parameter for every
non-singleton subset and, for every margin `i`, the vector
`(β_{i,C})_{C∋i}` as a probability simplex. `TawnTail(d, dep, asy)` retains
the historical subset-oriented input and converts it to this canonical form.
`TawnTail(α, weights)` is the convenience model containing only the full-set
logistic component plus singleton remainders.

The full subset representation grows exponentially with dimension, and zero
weights can put the model on reduced or partially independent boundaries where
some parameters are weakly identified.

See also: [`AsymLogTail`](@ref), [`ExtremeValueCopula`](@ref), [`ℓ`](@ref),
[`Distributions.fit`](@ref).

References:

* [tawn1988bivariate](@cite) for the bivariate precursor.
* [tawn1990multivariate](@cite) for the multivariate model.
"""
TawnTail
Paramorph.@paramorph T struct TawnTail{T<:Real} <: Tail
    d::Int
    dep::Vector{T} ~ Paramorph.TransformVariables.as(
        Vector, Paramorph.closed_lower(one(T)), 2^d - d - 1,
    )
    weights::Vector{Vector{T}} ~ Paramorph.repeat_transform(
        Paramorph.TransformVariables.UnitSimplex(2^(d - 1)), d,
    )
end

function TawnTail(dep::AbstractVector, weights::Vararg{AbstractVector,N}) where {N}
    normalized_dep, normalized_weights = _normalize_asymmetric_margin_components(
        N, dep, weights;
        singleton_parameter=1.0,
        valid_parameter=parameter -> parameter >= one(parameter),
        family="Tawn",
    )
    T = eltype(normalized_dep)
    return TawnTail{T}(N, normalized_dep, normalized_weights)
end


@inline _tawn_components(tail::TawnTail) = _asymmetric_subset_components(
    tail.d, tail.dep, tail.weights; singleton_parameter=1.0,
)

@inline function _tawn_bivariate_asym_log(tail::TawnTail)
    tail.d == 2 || throw(ArgumentError("the AsymLog reduction requires a bivariate TawnTail"))
    α, β = _tawn_components(tail)
    k = lastindex(α)
    return AsymLogTail(α[k], β[2, k], β[1, k])
end

function distortion(
    C::ExtremeValueCopula{2,<:TawnTail},
    js::NTuple{1,Int},
    uⱼₛ::NTuple{1,Float64},
    ::Int,
)
    kind = limit_kind(C.tail, Val(2))
    kind === Π_LIMIT && return NoDistortion()

    j = Int8(js[1])
    uⱼ = float(uⱼₛ[1])
    kind === M_LIMIT && return MDistortion(uⱼ, j)
    return BivEVDistortion(_tawn_bivariate_asym_log(C.tail), j, uⱼ)
end

@inline _tawn_component_is_active(α, β, j) =
    !isone(α[j]) && count(!iszero, @view β[:, j]) > 1

function _tawn_is_fullset_logistic(tail::TawnTail, α, β)
    fullset = lastindex(α)
    preceding = (tail.d + 1):(fullset - 1)
    any(j -> _tawn_component_is_active(α, β, j), preceding) && return false
    return all(isone, @view β[:, fullset])
end

@inline function limit_kind(tail::TawnTail, ::Val{d}) where {d}
    d == tail.d || return NO_LIMIT
    α, β = _tawn_components(tail)
    non_singletons = (tail.d + 1):lastindex(α)
    any(j -> _tawn_component_is_active(α, β, j), non_singletons) || return Π_LIMIT

    fullset = lastindex(α)
    return _tawn_is_fullset_logistic(tail, α, β) && isinf(α[fullset]) ? M_LIMIT : NO_LIMIT
end

function tail_measure_style(tail::TawnTail)
    α, β = _tawn_components(tail)
    for j in (tail.d + 1):lastindex(α)
        _tawn_component_is_active(α, β, j) && isinf(α[j]) &&
            return NonAbsolutelyContinuousMeasure()
    end
    return AbsolutelyContinuousMeasure()
end

"""
    TawnCopula{d}(α, weights)
    TawnCopula(d, α, weights)
    TawnCopula{d}(dep, weights₁, ..., weights_d)
    TawnCopula(d, dep, weights₁, ..., weights_d)
    TawnCopula{d}(dep, asy)
    TawnCopula(d, dep, asy)

Construct a Tawn asymmetric-logistic extreme-value copula.

The canonical full model uses one `dep` vector of length `2^d-d-1` and one
probability-simplex vector of length `2^(d-1)` for every margin. The historical
`(dep, asy)` form remains accepted, with one local weight vector for every
nonempty subset. `TawnCopula{d}(α, weights)` is the convenience model with one
full-set logistic component plus singleton remainders.
"""
const TawnCopula{d,T} = ExtremeValueCopula{d,TawnTail{T}}
function (::Type{TawnCopula{d}})(args...; kwargs...) where {d}
    return _wrap_extreme_value(Val(d), TawnTail(args...; kwargs...))
end
(::Type{TawnCopula})(d::Int, args...; kwargs...) = _wrap_extreme_value(Val(d), TawnTail(args...; kwargs...))

# Canonical runtime-dimension constructor used by generic Paramorph fitting of
# an explicitly dimensioned family type.
function TawnTail(d::Int, dep::AbstractVector, weights::Vararg{AbstractVector,N}) where {N}
    d == N || throw(DimensionMismatch(
        "expected one weight simplex for each of $d margins; got $N",
    ))
    return TawnTail(dep, weights...)
end

# Historical subset-oriented constructor.
function TawnTail(d::Int, dep::AbstractVector, asy::AbstractVector)
    weights = _subset_asymmetry_to_margin_weights(d, asy)
    return TawnTail(dep, weights...)
end
TawnTail(d::Int, dep::Vector{T}, asy::Vector{Vector{T}}) where {T<:Real} =
    invoke(TawnTail, Tuple{Int,AbstractVector,AbstractVector}, d, dep, asy)

# Convenience submodel: one full-set logistic component plus singleton remainders.
function TawnTail(α::TA, weights::AbstractVector{TW}) where {TA<:Real,TW<:Real}
    T = promote_type(Float64, TA, TW)
    tail = TawnTail(_expand_fullset_asymmetric_component(
        α, weights; singleton_parameter=1.0,
    )...)
    return tail::TawnTail{T}
end

TawnTail(dep::AbstractVector, asy::AbstractVector) =
    TawnTail(trailing_zeros(length(asy) + 1), dep, asy)

Distributions.params(C::ExtremeValueCopula{d,<:TawnTail}) where {d} =
    (copy(C.tail.dep), (copy(weight) for weight in C.tail.weights)...)

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
    subsets = _nonempty_subsets(tail.d)
    α, β = _tawn_components(tail)
    T = promote_type(eltype(x), eltype(α), eltype(β))
    out = zero(T)

    @inbounds for j in eachindex(subsets)
        out += _tawn_component_stdf(
            α[j],
            @view(β[:, j]),
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

    T = promote_type(typeof(float(α)), eltype(βcol), eltype(x))
    logterms = T[]
    @inbounds for i in C
        yi = float(βcol[i]) * float(x[i])
        yi > 0 && push!(logterms, float(α) * log(yi))
    end
    isempty(logterms) && return 0, -Inf
    logS = LogExpFunctions.logsumexp(logterms)

    logcoef = zero(T)
    @inbounds for j in 1:(k - 1)
        c = one(T) - j * float(α)
        iszero(c) && return 0, -Inf
        logcoef += log(abs(c))
    end

    logprod = zero(T)
    @inbounds for i in I
        logprod += float(α) * log(float(βcol[i]))
        logprod += (float(α) - one(T)) * log(float(x[i]))
    end

    logabs = logcoef + (inv(float(α)) - k) * logS + logprod
    sign = isodd(k) ? 1 : -1
    return sign, logabs
end

function _ellpartial_signlog(tail::TawnTail, x, I::Tuple{Vararg{Int}},)
    isempty(I) && return 1, log(float(ℓ(tail, x)))

    subsets = _nonempty_subsets(tail.d)
    α, β = _tawn_components(tail)
    expected_sign = isodd(length(I)) ? 1 : -1
    return _sum_component_partials(length(subsets), expected_sign) do j
        _tawn_component_partial_signlog(
            α[j], @view(β[:, j]), subsets[j], x, I,
        )
    end
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:TawnTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    return _rand_with_ev_limits!(rng, C, X) do
        α, β = _tawn_components(C.tail)
        return _rand_subset_components!(
            rng, X, α, β, isone,
            (dimension, parameter) -> ExtremeValueCopula(dimension, LogTail(parameter));
            family="Tawn",
        )
    end
end
