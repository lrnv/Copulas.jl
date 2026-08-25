"""
    AsymGalambosTail{T}, AsymGalambosCopula{d,T}

    AsymGalambosCopula{2}(α, θ₁, θ₂)
    AsymGalambosCopula(2, α, θ₁, θ₂)
    AsymGalambosCopula{d}(α, weights)
    AsymGalambosCopula(d, α, weights)
    AsymGalambosCopula(α, weights)
    AsymGalambosCopula{d}(dep, asy)
    AsymGalambosCopula(d, dep, asy)
    AsymGalambosCopula(dep, asy)

Asymmetric Galambos (negative-logistic) extreme-value family.

The first constructor is the specialized bivariate kernel. The full
multivariate implementation uses a subset-based negative-logistic/min-stable
construction of the type developed by Joe [Joe1990](@cite). For nonempty
subsets `C`,

```math
\\ell(x)
=
\\sum_C
\\ell_{\\mathrm{Galambos},\\alpha_C}
\\bigl((\\beta_{i,C}x_i)_{i\\in C}\\bigr),
```

with nonnegative asymmetry weights satisfying the marginal normalization
constraints.

`AsymGalambosCopula(d, dep, asy)` exposes the full subset representation.

`AsymGalambosCopula(α, weights)` is a Copulas.jl convenience
parameterization: one full-set negative-logistic component is combined with
singleton remainders. In `d=2` it reproduces the specialized asymmetric
Galambos model.

!!! note "Literature model versus package parameterization"
    [Joe1990](@cite) supports the multivariate min-stable/negative-logistic
    construction. The one-full-set-plus-singletons `weights` constructor is a
    convenience parameterization introduced at the implementation level in
    Copulas.jl.

References:

* [galambos1975order](@cite)
* [Joe1990](@cite)
"""
AsymGalambosTail, AsymGalambosCopula

struct AsymGalambosTail{T} <: BivariatePickandsTail
    α::T                 # α ≥ 0
    θ₁::T
    θ₂::T
    function AsymGalambosTail(α, θ₁, θ₂)

        T = promote_type(Float64, typeof(α), typeof(θ₁), typeof(θ₂))
        θ₁, θ₂, αT = T(θ₁), T(θ₂), T(α)

        (αT ≥ 0) || throw(ArgumentError("α must be ≥ 0"))
        (0 ≤ θ₁ ≤ 1 && 0 ≤ θ₂ ≤ 1) || throw(ArgumentError("each θ[i] must be in [0,1]"))
        αT == 0 || (θ₁ == 0 && θ₂ == 0) && return NoTail()
        θ₁ == 1 && θ₂ == 1 && return GalambosTail(α)
        return new{T}(αT, θ₁, θ₂)
    end
end

const AsymGalambosCopula{d,T} = ExtremeValueCopula{d,AsymGalambosTail{T}}

function _asymgal_dimension_from_subset_weights(asy::AbstractVector)
    return _subset_dimension(asy, "Asymmetric Galambos")
end

function _asymgal_convenience_copula(CT, α::Real, weights::AbstractVector)
    tail = AsymGalambosMultiTail(α, weights)
    d = _ev_resolve_dimension(CT, tail.d, "weight-vector")
    return ExtremeValueCopula{d}(tail)
end

function (CT::Type{<:ExtremeValueCopula{D,<:AsymGalambosTail} where D})(α::Real, weights::AbstractVector,)
    return _asymgal_convenience_copula(CT, α, weights)
end

function (CT::Type{<:ExtremeValueCopula{D,<:AsymGalambosTail} where D})(α::Int, weights::AbstractVector,)
    return _asymgal_convenience_copula(CT, α, weights)
end

function (::Type{<:ExtremeValueCopula{D,<:AsymGalambosTail} where D})(d::Int, α::Real, weights::AbstractVector,)
    tail = AsymGalambosMultiTail(α, weights)
    d == tail.d || throw(DimensionMismatch(
        "d=$d does not match weight-vector dimension $(tail.d)",
    ))
    return ExtremeValueCopula{d}(tail)
end

function (CT::Type{<:ExtremeValueCopula{D,<:AsymGalambosTail} where D})(dep::AbstractVector, asy::AbstractVector,)
    inferred = _asymgal_dimension_from_subset_weights(asy)
    d = _ev_resolve_dimension(CT, inferred, "subset")
    return ExtremeValueCopula{d}(AsymGalambosMultiTail(d, dep, asy))
end

function (::Type{<:ExtremeValueCopula{D,<:AsymGalambosTail} where D})(d::Int, dep::AbstractVector, asy::AbstractVector,)
    return ExtremeValueCopula{d}(AsymGalambosMultiTail(d, dep, asy))
end

Distributions.params(tail::AsymGalambosTail) = (α = tail.α, θ₁ = tail.θ₁, θ₂ = tail.θ₂)
_unbound_params(::Type{<:AsymGalambosTail}, d, θ) = [log(θ.α), LogExpFunctions.logit(θ.θ₁), LogExpFunctions.logit(θ.θ₂)]
_rebound_params(::Type{<:AsymGalambosTail}, d, α) = begin 
    (; α = exp(α[1]), θ₁ = LogExpFunctions.logistic(α[2]), θ₂ = LogExpFunctions.logistic(α[3]))
end

function A(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂  = tail.α, tail.θ₁, tail.θ₂

    α == 0 || (θ₁ == 0 && θ₂ == 0) && return one(tt)
    x1 = -α * log(θ₁ * tt)
    x2 = -α * log(θ₂ * (1 - tt))
    s  = LogExpFunctions.logaddexp(x1, x2) / α
    return -LogExpFunctions.expm1(-s)
end

function dA(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = tail.α, tail.θ₁, tail.θ₂

    α == 0 || (θ₁ == 0 && θ₂ == 0) && return one(tt)

    a = tt
    b = 1 - tt

    x1 = -α * log(θ₁ * a)
    x2 = -α * log(θ₂ * b)

    ℓ = LogExpFunctions.logaddexp(x1, x2)

    w1 = exp(x1 - ℓ)
    w2 = exp(x2 - ℓ)

    B = exp(-ℓ / α)

    return B * (w2 / b - w1 / a)
end

function d²A(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = tail.α, tail.θ₁, tail.θ₂

    α == 0 || (θ₁ == 0 && θ₂ == 0) && return one(tt)

    a = tt
    b = 1 - tt

    x1 = -α * log(θ₁ * a)
    x2 = -α * log(θ₂ * b)

    ℓ = LogExpFunctions.logaddexp(x1, x2)

    w1 = exp(x1 - ℓ)
    w2 = exp(x2 - ℓ)

    B = exp(-ℓ / α)

    inva = inv(a)
    invb = inv(b)

    g = w2 * invb - w1 * inva

    term1 = w2 * invb^2 + w1 * inva^2
    term2 = g^2

    return (1 + α) * B * (term1 - term2)
end

"""
    AsymGalambosMultiTail(d, dep, asy)
    AsymGalambosMultiTail(α, weights)

Internal multivariate representation of the asymmetric Galambos
(negative-logistic) family. For every nonempty subset ``C`` it combines a
weighted Galambos component,

```math
\\ell(x)
=
\\sum_{\\varnothing\\ne C\\subseteq\\{1,\\ldots,d\\}}
\\ell_{\\mathrm{Gal},\\alpha_C}
\\bigl((\\beta_{i,C}x_i)_{i\\in C}\\bigr),
```

with nonnegative weights satisfying
``\\sum_{C\\ni i}\\beta_{i,C}=1`` for each margin. Singleton components and
components with `α_C = 0` contribute linearly.

Prefer `AsymGalambosCopula(α, weights)` or
`AsymGalambosCopula(d, dep, asy)` in user code.
"""
struct AsymGalambosMultiTail{T} <: Tail
    d::Int
    α::Vector{T}
    β::Matrix{T}
end

function AsymGalambosMultiTail(d::Int, dep::AbstractVector, asy::AbstractVector,)
    α, β = _subset_model_parameters(
        d, dep, asy;
        singleton_parameter=0.0,
        valid_parameter=parameter -> parameter >= zero(parameter),
        family="Galambos",
    )
    return AsymGalambosMultiTail{eltype(α)}(d, α, β)
end

# Convenience submodel: one full-set Galambos component plus singleton
# remainders.  In d=2 this matches the historical AsymGalambosTail using
# weights in the same coordinate order: [θ₁, θ₂].
function AsymGalambosMultiTail(α::Real, weights::AbstractVector)
    α >= zero(α) || throw(ArgumentError("α must be ≥ 0"))
    d, dep, asy = _fullset_subset_parameters(α, weights; singleton_parameter=0.0)
    return AsymGalambosMultiTail(d, dep, asy)
end

Distributions.params(tail::AsymGalambosMultiTail) = (α = tail.α, β = tail.β)

_is_valid_in_dim(tail::AsymGalambosMultiTail, d::Int) = d == tail.d

function _asymgal_component_data(tail::AsymGalambosMultiTail, j, x)
    C = _nonempty_subsets(tail.d)[j]
    active = [i for i in C if tail.β[i, j] > 0]
    y = [tail.β[i, j] * x[i] for i in active]
    return C, active, y
end

function ℓ(tail::AsymGalambosMultiTail, x)
    length(x) == tail.d || throw(DimensionMismatch("input dimension does not match asymmetric Galambos tail dimension",))

    subsets = _nonempty_subsets(tail.d)
    T = promote_type(eltype(x), eltype(tail.α), eltype(tail.β))
    out = zero(T)

    @inbounds for j in eachindex(subsets)
        C = subsets[j]
        active = [i for i in C if tail.β[i, j] > 0]
        isempty(active) && continue

        a = tail.α[j]
        if a == 0 || length(active) == 1
            for i in active
                out += tail.β[i, j] * x[i]
            end
            continue
        end

        y = [tail.β[i, j] * x[i] for i in active]
        out += ℓ(GalambosTail(a), y)
    end

    return out
end

function _asymgal_component_partial_signlog(tail::AsymGalambosMultiTail, j::Int, x, I::Tuple{Vararg{Int}},)
    k = length(I)
    k > 0 || throw(ArgumentError("partial block must be nonempty"))

    C = _nonempty_subsets(tail.d)[j]
    active = [i for i in C if tail.β[i, j] > 0]

    all(i -> i in active, I) || return 0, -Inf

    a = tail.α[j]
    if a == 0 || length(active) == 1
        if k == 1
            return 1, log(float(tail.β[only(I), j]))
        end
        return 0, -Inf
    end

    y = [tail.β[i, j] * x[i] for i in active]
    positions = Dict(i => q for (q, i) in enumerate(active))
    localI = ntuple(q -> positions[I[q]], k)

    sign, logabs = _ellpartial_signlog(GalambosTail(a), y, localI,)
    sign == 0 && return 0, -Inf

    logchain = sum(log(float(tail.β[i, j])) for i in I)
    return sign, logabs + logchain
end

function _ellpartial_signlog(tail::AsymGalambosMultiTail, x, I::Tuple{Vararg{Int}},)
    isempty(I) && return 1, log(float(ℓ(tail, x)))

    expected_sign = isodd(length(I)) ? 1 : -1
    return _sum_component_partials(size(tail.β, 2), expected_sign) do j
        _asymgal_component_partial_signlog(tail, j, x, I,)
    end
end




function _asymgal_rand_multivariate!(rng::Distributions.AbstractRNG, tail::AsymGalambosMultiTail, X::AbstractMatrix{T},) where {T<:Real}
    d, n = size(X)
    d == tail.d || throw(DimensionMismatch("output dimension does not match asymmetric Galambos tail dimension",))

    return _rand_subset_components!(
        rng, X, tail.α, tail.β, iszero,
        (dimension, α) -> ExtremeValueCopula(dimension, GalambosTail(α));
        family="asymmetric Galambos",
    )
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:AsymGalambosMultiTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch("output dimension does not match copula dimension",))
    return _asymgal_rand_multivariate!(rng, C.tail, X)
end
