"""
    AsymGalambosTail{T}, AsymGalambosCopula{d,T}

    AsymGalambosCopula{2}(α, θ₁, θ₂)
    AsymGalambosCopula(2, α, θ₁, θ₂)
    AsymGalambosCopula{d}(α, weights)
    AsymGalambosCopula(d, α, weights)
    AsymGalambosCopula{d}(dep, asy)
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

`AsymGalambosCopula(d, dep, asy)` exposes the full subset representation.
`AsymGalambosCopula{d}(α, weights)` is a convenience parameterization with
one full-set negative-logistic component and singleton remainders. In `d=2`,
it is equivalent to the historical `(α, θ₁, θ₂)` parameterization and
retains the specialized scalar Pickands formulas.

!!! note "Literature model versus package parameterization"
    [Joe1990](@cite) supports the multivariate min-stable/negative-logistic
    construction. The one-full-set-plus-singletons `weights` constructor is a
    convenience parameterization introduced at the implementation level in
    Copulas.jl.

References:

* [galambos1975order](@cite) Order statistics of samples from multivariate distributions. JASA, 1975.
* [Joe1990](@cite) Families of min-stable multivariate exponential and multivariate extreme value distributions. Statist. Probab, 1990.
"""
AsymGalambosTail, AsymGalambosCopula

struct AsymGalambosTail{T} <: BivariatePickandsTail
    d::Int
    α::Vector{T}
    β::Matrix{T}
end

const AsymGalambosCopula{d,T} = ExtremeValueCopula{d,AsymGalambosTail{T}}

function AsymGalambosTail(d::Int, dep::AbstractVector, asy::AbstractVector)
    α, β = _normalize_asymmetric_subset_components(
        d, dep, asy;
        singleton_parameter=0.0,
        valid_parameter=parameter -> parameter >= zero(parameter),
        family="Galambos",
    )
    return AsymGalambosTail{eltype(α)}(d, α, β)
end

# Convenience submodel: one full-set Galambos component plus singleton
# remainders.
function AsymGalambosTail(α::Real, weights::AbstractVector)
    α >= zero(α) || throw(ArgumentError("α must be ≥ 0"))
    d = length(weights)
    d >= 2 || throw(ArgumentError("weights must contain at least two entries"))
    all(weight -> zero(weight) <= weight <= one(weight), weights) ||
        throw(ArgumentError("all full-set weights must lie in [0,1]"))
    d == 2 && (iszero(α) || all(iszero, weights)) && return NoTail()
    d == 2 && all(isone, weights) && return GalambosTail(α)
    _, dep, asy = _expand_fullset_asymmetric_component(α, weights; singleton_parameter=0.0)
    return AsymGalambosTail(d, dep, asy)
end

AsymGalambosTail(α::Real, θ₁::Real, θ₂::Real) = AsymGalambosTail(α, [θ₁, θ₂])

function AsymGalambosTail(dep::AbstractVector, asy::AbstractVector)
    d = trailing_zeros(length(asy) + 1)
    return AsymGalambosTail(d, dep, asy)
end

_is_valid_in_dim(tail::AsymGalambosTail, d::Int) = d == tail.d

@inline function _asymgal_bivariate_parameters(tail::AsymGalambosTail)
    tail.d == 2 || throw(DimensionMismatch("the scalar Pickands representation requires a bivariate tail"))
    return tail.α[end], tail.β[1, end], tail.β[2, end]
end

function Distributions.params(tail::AsymGalambosTail)
    if tail.d == 2
        α, θ₁, θ₂ = _asymgal_bivariate_parameters(tail)
        return (; α, θ₁, θ₂)
    end
    subsets = _nonempty_subsets(tail.d)
    dep = tail.α[(tail.d + 1):end]
    asy = [collect(@view tail.β[subset, j]) for (j, subset) in enumerate(subsets)]
    return (; dep, asy)
end

_available_fitting_methods(::Type{<:ExtremeValueCopula{D,<:AsymGalambosTail} where D}, d) =
    d == 2 ? (:mle,) : ()

function _unbound_params(::Type{<:AsymGalambosTail}, d, θ)
    d == 2 || throw(ArgumentError("generic fitting is not implemented for multivariate asymmetric Galambos tails"))
    return [log(θ.α), LogExpFunctions.logit(θ.θ₁), LogExpFunctions.logit(θ.θ₂)]
end

function _rebound_params(::Type{<:AsymGalambosTail}, d, α)
    d == 2 || throw(ArgumentError("generic fitting is not implemented for multivariate asymmetric Galambos tails"))
    return (; α=exp(α[1]), θ₁=LogExpFunctions.logistic(α[2]), θ₂=LogExpFunctions.logistic(α[3]))
end

function A(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = _asymgal_bivariate_parameters(tail)

    (iszero(α) || (iszero(θ₁) && iszero(θ₂))) && return one(tt)
    x1 = -α * log(θ₁ * tt)
    x2 = -α * log(θ₂ * (1 - tt))
    s = LogExpFunctions.logaddexp(x1, x2) / α
    return -LogExpFunctions.expm1(-s)
end

function dA(tail::AsymGalambosTail, t::Real)
    tt = _safett(t)
    α, θ₁, θ₂ = _asymgal_bivariate_parameters(tail)

    (iszero(α) || (iszero(θ₁) && iszero(θ₂))) && return zero(tt)

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

    (iszero(α) || (iszero(θ₁) && iszero(θ₂))) && return zero(tt)

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
    length(x) == tail.d || throw(DimensionMismatch(
        "input dimension does not match asymmetric Galambos tail dimension",
    ))

    subsets = _nonempty_subsets(tail.d)
    T = promote_type(eltype(x), eltype(tail.α), eltype(tail.β))
    out = zero(T)

    @inbounds for j in eachindex(subsets)
        subset = subsets[j]
        active = [i for i in subset if tail.β[i, j] > 0]
        isempty(active) && continue

        α = tail.α[j]
        if iszero(α) || length(active) == 1
            for i in active
                out += tail.β[i, j] * x[i]
            end
        else
            y = [tail.β[i, j] * x[i] for i in active]
            out += ℓ(GalambosTail(α), y)
        end
    end
    return out
end

function _asymgal_component_partial_signlog(tail::AsymGalambosTail, j::Int, x, I::Tuple{Vararg{Int}})
    k = length(I)
    k > 0 || throw(ArgumentError("partial block must be nonempty"))

    subset = _nonempty_subsets(tail.d)[j]
    active = [i for i in subset if tail.β[i, j] > 0]
    all(i -> i in active, I) || return 0, -Inf

    α = tail.α[j]
    if iszero(α) || length(active) == 1
        k == 1 && return 1, log(float(tail.β[only(I), j]))
        return 0, -Inf
    end

    y = [tail.β[i, j] * x[i] for i in active]
    positions = Dict(i => q for (q, i) in enumerate(active))
    localI = ntuple(q -> positions[I[q]], k)
    sign, logabs = _ellpartial_signlog(GalambosTail(α), y, localI)
    iszero(sign) && return 0, -Inf

    logchain = sum(log(float(tail.β[i, j])) for i in I)
    return sign, logabs + logchain
end

function _ellpartial_signlog(tail::AsymGalambosTail, x, I::Tuple{Vararg{Int}})
    isempty(I) && return 1, log(float(ℓ(tail, x)))
    expected_sign = isodd(length(I)) ? 1 : -1
    return _sum_component_partials(size(tail.β, 2), expected_sign) do j
        _asymgal_component_partial_signlog(tail, j, x, I)
    end
end

function _asymgal_rand_multivariate!(rng::Distributions.AbstractRNG, tail::AsymGalambosTail, X::AbstractMatrix{T}) where {T<:Real}
    d = size(X, 1)
    d == tail.d || throw(DimensionMismatch("output dimension does not match asymmetric Galambos tail dimension"))
    return _rand_subset_components!(
        rng,
        X,
        tail.α,
        tail.β,
        iszero,
        (dimension, α) -> ExtremeValueCopula(dimension, GalambosTail(α));
        family="asymmetric Galambos",
    )
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:AsymGalambosTail}, X::AbstractMatrix{T}) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch("output dimension does not match copula dimension"))
    return _asymgal_rand_multivariate!(rng, C.tail, X)
end

# Resolve the intersection above with the generic bivariate Pickands sampler.
function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{2,<:AsymGalambosTail}, X::AbstractMatrix{T}) where {T<:Real}
    signature = Tuple{
        Distributions.AbstractRNG,
        ExtremeValueCopula{2,<:BivariatePickandsTail},
        AbstractMatrix{T},
    }
    return invoke(Distributions._rand!, signature, rng, C, X)
end
