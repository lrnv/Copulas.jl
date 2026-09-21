"""
    AsymGalambosTail(α, θ₁, θ₂)
    AsymGalambosTail(α, weights)
    AsymGalambosTail(dep, weights₁, ..., weights_d)
    AsymGalambosTail(d, dep, asy)

    AsymGalambosCopula{2}(α, θ₁, θ₂)
    AsymGalambosCopula(2, α, θ₁, θ₂)
    AsymGalambosCopula{d}(α, weights)
    AsymGalambosCopula(d, α, weights)
    AsymGalambosCopula{d}(dep, weights₁, ..., weights_d)
    AsymGalambosCopula(d, dep, asy)

Asymmetric Galambos (negative-logistic) extreme-value family.

The family uses the subset-based negative-logistic/min-stable construction of
Joe [Joe1990](@cite). For nonempty subsets `C`,

```math
\\ell(x)
=
\\sum_C
\\ell_{\\mathrm{Galambos},\\alpha_C}
\\bigl((\\beta_{i,C}x_i)_{i\\in C}\\bigr),
```

with nonnegative asymmetry weights satisfying the marginal normalization
constraints.

The canonical parameterization stores one dependence parameter for every
non-singleton subset and, for every margin `i`, the vector
`(β_{i,C})_{C∋i}` as a probability simplex. `AsymGalambosTail(d, dep, asy)`
retains the historical subset-oriented input and converts it to this canonical
form. `AsymGalambosTail(α, weights)` contains only the full-set
negative-logistic component plus singleton remainders. In `d=2`, this is the
whole family and is equivalent to the historical `(α, θ₁, θ₂)` form.

The full subset form can be high-dimensional in both parameter count and
evaluation cost. Zero weights remove subset contributions and can place the
model on a lower-dimensional or partially independent boundary.

See also: [`Tail`](@ref), [`ExtremeValueCopula`](@ref), [`ℓ`](@ref),
[`Distributions.fit`](@ref).

References:

* [galambos1975order](@cite) Order statistics of samples from multivariate distributions. JASA, 1975.
* [Joe1990](@cite) Families of min-stable multivariate exponential and multivariate extreme value distributions. Statist. Probab, 1990.
"""
AsymGalambosTail, AsymGalambosCopula

Paramorph.@paramorph T struct AsymGalambosTail{T<:Real} <: BivariatePickandsTail
    d::Int
    dep::Vector{T}
    weights::Vector{Vector{T}}
end

function AsymGalambosTail(dep::AbstractVector, weights::Vararg{AbstractVector,N}) where {N}
    normalized_dep, normalized_weights = _normalize_asymmetric_margin_components(
        N, dep, weights;
        singleton_parameter=0.0,
        valid_parameter=parameter -> parameter >= zero(parameter),
        family="Galambos",
    )
    T = eltype(normalized_dep)
    return AsymGalambosTail{T}(N, normalized_dep, normalized_weights)
end

Paramorph.parameter_fields_override(::Type{<:AsymGalambosTail}) = (:dep, :weights)
function _asymgalambos_schema(d::Integer)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    return Paramorph.TransformVariables.as((
        dep=Paramorph.TransformVariables.as(
            Vector, Paramorph.nonnegative(), 2^d - d - 1,
        ),
        weights=Paramorph.repeat_transform(
            Paramorph.TransformVariables.UnitSimplex(2^(d - 1)), d,
        ),
    ))
end
Paramorph.schema_override(::Type{<:AsymGalambosTail}, ::NamedTuple) = throw(ArgumentError(
    "AsymGalambosTail requires a prototype because its parameter geometry depends on dimension",
))
Paramorph.schema_override(
    ::Type{<:AsymGalambosTail}, ::NamedTuple, values::NamedTuple,
) = _asymgalambos_schema(values.d)
Paramorph.schema_override(tail::AsymGalambosTail, ::NamedTuple) =
    _asymgalambos_schema(tail.d)

@inline _asymgal_components(tail::AsymGalambosTail) = _asymmetric_subset_components(
    tail.d, tail.dep, tail.weights; singleton_parameter=0.0,
)

@inline _asymgal_component_is_active(α, β, j) =
    !iszero(α[j]) && count(!iszero, @view β[:, j]) > 1

function _asymgal_is_fullset_galambos(tail::AsymGalambosTail, α, β)
    fullset = lastindex(α)
    preceding = (tail.d + 1):(fullset - 1)
    any(j -> _asymgal_component_is_active(α, β, j), preceding) && return false
    return all(isone, @view β[:, fullset])
end

@inline function limit_kind(tail::AsymGalambosTail, ::Val{d}) where {d}
    d == tail.d || return NO_LIMIT
    α, β = _asymgal_components(tail)
    non_singletons = (tail.d + 1):lastindex(α)
    any(j -> _asymgal_component_is_active(α, β, j), non_singletons) || return Π_LIMIT

    fullset = lastindex(α)
    return _asymgal_is_fullset_galambos(tail, α, β) && isinf(α[fullset]) ?
           M_LIMIT : NO_LIMIT
end

function tail_measure_style(tail::AsymGalambosTail)
    α, β = _asymgal_components(tail)
    for j in (tail.d + 1):lastindex(α)
        _asymgal_component_is_active(α, β, j) && isinf(α[j]) &&
            return NonAbsolutelyContinuousMeasure()
    end
    return AbsolutelyContinuousMeasure()
end

const AsymGalambosCopula{d,T} = ExtremeValueCopula{d,AsymGalambosTail{T}}
function (::Type{AsymGalambosCopula{d}})(args...; kwargs...) where {d}
    return _wrap_extreme_value(Val(d), AsymGalambosTail(args...; kwargs...))
end
(::Type{AsymGalambosCopula})(d::Int, args...; kwargs...) = _wrap_extreme_value(Val(d), AsymGalambosTail(args...; kwargs...))

# Canonical runtime-dimension constructor used by generic Paramorph fitting of
# an explicitly dimensioned family type.
function AsymGalambosTail(d::Int, dep::AbstractVector, weights::Vararg{AbstractVector,N}) where {N}
    d == N || throw(DimensionMismatch(
        "expected one weight simplex for each of $d margins; got $N",
    ))
    return AsymGalambosTail(dep, weights...)
end

# Historical subset-oriented constructor.
function AsymGalambosTail(d::Int, dep::AbstractVector, asy::AbstractVector)
    weights = _subset_asymmetry_to_margin_weights(d, asy)
    return AsymGalambosTail(dep, weights...)
end
AsymGalambosTail(d::Int, dep::Vector{T}, asy::Vector{Vector{T}}) where {T<:Real} =
    invoke(AsymGalambosTail, Tuple{Int,AbstractVector,AbstractVector}, d, dep, asy)

# Convenience submodel: one full-set Galambos component plus singleton
# remainders.
function AsymGalambosTail(α::TA, weights::AbstractVector{TW}) where {TA<:Real,TW<:Real}
    T = promote_type(Float64, TA, TW)
    tail = AsymGalambosTail(_expand_fullset_asymmetric_component(
        α, weights; singleton_parameter=0.0,
    )...)
    return tail::AsymGalambosTail{T}
end

function AsymGalambosTail(α::TA, θ₁::T1, θ₂::T2) where {TA<:Real,T1<:Real,T2<:Real}
    T = promote_type(Float64, TA, T1, T2)
    return AsymGalambosTail(T(α), T[θ₁, θ₂])::AsymGalambosTail{T}
end

AsymGalambosTail(dep::AbstractVector, asy::AbstractVector) =
    AsymGalambosTail(trailing_zeros(length(asy) + 1), dep, asy)

Distributions.params(C::ExtremeValueCopula{d,<:AsymGalambosTail}) where {d} =
    (copy(C.tail.dep), (copy(weight) for weight in C.tail.weights)...)

_is_valid_in_dim(tail::AsymGalambosTail, d::Int) = d == tail.d

@inline function _asymgal_bivariate_parameters(tail::AsymGalambosTail)
    tail.d == 2 || throw(DimensionMismatch("bivariate parameters require d = 2"))
    return tail.dep[1], tail.weights[1][end], tail.weights[2][end]
end

_tail_constructor_parameter_names(::Type{<:AsymGalambosTail}, _) = (:α, :θ₁, :θ₂)

function A(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = _asymgal_bivariate_parameters(tail)

    x = θ₁ * tt
    y = θ₂ * (1 - tt)
    # For the bivariate Galambos STDF,
    #   x + y - ℓ_Galambos(x,y) = (x^-α + y^-α)^(-1/α).
    # Writing the asymmetric model through that primitive makes the α=0,
    # inactive-support, symmetric, and α=∞ boundaries emerge naturally.
    dependence = x + y - ℓ(GalambosTail(α), (x, y))
    return one(tt) - dependence
end

function dA(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = _asymgal_bivariate_parameters(tail)

    (iszero(α) || iszero(θ₁) || iszero(θ₂)) && return zero(tt)

    a = tt
    b = 1 - tt
    x1 = -α * log(θ₁ * a)
    x2 = -α * log(θ₂ * b)
    logsum = LogExpFunctions.logaddexp(x1, x2)
    w1 = exp(x1 - logsum)
    w2 = exp(x2 - logsum)
    B = exp(-logsum / α)

    return B * (w2 / b - w1 / a)
end

function d²A(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = _asymgal_bivariate_parameters(tail)

    (iszero(α) || iszero(θ₁) || iszero(θ₂)) && return zero(tt)

    a = tt
    b = 1 - tt
    x1 = -α * log(θ₁ * a)
    x2 = -α * log(θ₂ * b)
    logsum = LogExpFunctions.logaddexp(x1, x2)
    w1 = exp(x1 - logsum)
    w2 = exp(x2 - logsum)
    B = exp(-logsum / α)
    inva = inv(a)
    invb = inv(b)
    g = w2 * invb - w1 * inva
    term1 = w2 * invb^2 + w1 * inva^2

    return (1 + α) * B * (term1 - g^2)
end

function ℓ(tail::AsymGalambosTail, x)
    subsets = _nonempty_subsets(tail.d)
    α, β = _asymgal_components(tail)
    T = promote_type(eltype(x), eltype(α), eltype(β))
    out = zero(T)

    @inbounds for j in eachindex(subsets)
        subset = subsets[j]
        active = [i for i in subset if β[i, j] > 0]
        isempty(active) && continue

        parameter = α[j]
        if iszero(parameter) || length(active) == 1
            for i in active
                out += β[i, j] * x[i]
            end
        else
            y = [β[i, j] * x[i] for i in active]
            out += ℓ(GalambosTail(parameter), y)
        end
    end
    return out
end

function _ellpartial_signlog(tail::AsymGalambosTail, x, I::Tuple{Vararg{Int}})
    isempty(I) && return 1, log(float(ℓ(tail, x)))
    k = length(I)
    expected_sign = isodd(k) ? 1 : -1
    subsets = _nonempty_subsets(tail.d)
    α, β = _asymgal_components(tail)
    return _sum_component_partials(size(β, 2), expected_sign) do j
        active = [i for i in subsets[j] if β[i, j] > 0]
        all(i -> i in active, I) || return 0, -Inf

        parameter = α[j]
        if iszero(parameter) || length(active) == 1
            k == 1 && return 1, log(float(β[only(I), j]))
            return 0, -Inf
        end

        y = [β[i, j] * x[i] for i in active]
        positions = Dict(i => q for (q, i) in enumerate(active))
        localI = ntuple(q -> positions[I[q]], k)
        sign, logabs = _ellpartial_signlog(GalambosTail(parameter), y, localI)
        iszero(sign) && return 0, -Inf

        logchain = sum(log(float(β[i, j])) for i in I)
        return sign, logabs + logchain
    end
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:AsymGalambosTail}, X::AbstractMatrix{T}) where {d,T<:Real}
    return _rand_with_ev_limits!(rng, C, X) do
        α, β = _asymgal_components(C.tail)
        _rand_subset_components!(
            rng,
            X,
            α,
            β,
            iszero,
            (dimension, parameter) -> ExtremeValueCopula(dimension, GalambosTail(parameter));
            family="asymmetric Galambos",
        )
    end
end

# Retain the generic Pickands sampler in dimension two.
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2,<:AsymGalambosTail}, X::AbstractMatrix{T}) where {T<:Real}
    signature = Tuple{
        Distributions.AbstractRNG,
        ExtremeValueCopula{2,<:BivariatePickandsTail},
        AbstractMatrix{T},
    }
    return invoke(Distributions._rand!, signature, rng, C, X)
end
