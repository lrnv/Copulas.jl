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
    m = length(asy) + 1
    ispow2(m) || throw(DimensionMismatch(
        "asy must contain 2^d-1 subset-weight vectors",
    ))
    d = trailing_zeros(m)
    d >= 2 || throw(ArgumentError("Asymmetric Galambos dimension must be at least two"))
    return d
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
_unbound_params(::Type{<:AsymGalambosTail}, d, θ) = [log(θ.α), log(θ.θ₁) - log1p(-θ.θ₁), log(θ.θ₂) - log1p(-θ.θ₂)] 
_rebound_params(::Type{<:AsymGalambosTail}, d, α) = begin 
    σ(x) = 1 / (1 + exp(-x)) 
    (; α = exp(α[1]), θ₁ = σ(α[2]), θ₂ = σ(α[3])) 
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

function _asymgal_subsets(d::Int)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    subsets = Vector{Vector{Int}}()
    for k in 1:d
        for C in Combinatorics.combinations(1:d, k)
            push!(subsets, collect(C))
        end
    end
    return subsets
end

function AsymGalambosMultiTail(d::Int, dep::AbstractVector, asy::AbstractVector,)
    subsets = _asymgal_subsets(d)
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

    α = zeros(T, m)
    @inbounds for j in (d + 1):m
        a = T(dep[j - d])
        a >= zero(T) || throw(ArgumentError("each non-singleton Galambos parameter must be ≥ 0",))
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
            throw(ArgumentError("asymmetry weights for margin $i must sum to one; got $rowsum",))
    end

    return AsymGalambosMultiTail{T}(d, α, β)
end

# Convenience submodel: one full-set Galambos component plus singleton
# remainders.  In d=2 this matches the historical AsymGalambosTail using
# weights in the same coordinate order: [θ₁, θ₂].
function AsymGalambosMultiTail(α::Real, weights::AbstractVector)
d = length(weights)
    d >= 2 || throw(ArgumentError("weights must contain at least two entries",))

    subsets = _asymgal_subsets(d)
    m = length(subsets)
    T = promote_type(Float64, typeof(α), eltype(weights))
    a = T(α)
    a >= zero(T) || throw(ArgumentError("α must be ≥ 0"))

    w = T.(weights)
    all(v -> zero(T) <= v <= one(T), w) || throw(ArgumentError("all full-set weights must lie in [0,1]"))

    dep = zeros(T, m - d)
    dep[end] = a

    asy = Vector{Vector{T}}(undef, m)
    for (j, C) in enumerate(subsets)
        asy[j] = zeros(T, length(C))
    end

    for i in 1:d
        asy[i][1] = one(T) - w[i]
    end
    asy[end] .= w

    return AsymGalambosMultiTail(d, dep, asy)
end

Distributions.params(tail::AsymGalambosMultiTail) = (α = tail.α, β = tail.β)

_is_valid_in_dim(tail::AsymGalambosMultiTail, d::Int) = d == tail.d

function _asymgal_component_data(tail::AsymGalambosMultiTail, j, x)
    C = _asymgal_subsets(tail.d)[j]
    active = [i for i in C if tail.β[i, j] > 0]
    y = [tail.β[i, j] * x[i] for i in active]
    return C, active, y
end

function ℓ(tail::AsymGalambosMultiTail, x)
    length(x) == tail.d || throw(DimensionMismatch("input dimension does not match asymmetric Galambos tail dimension",))

    subsets = _asymgal_subsets(tail.d)
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

@inline function _asymgal_logsumexp(logs::AbstractVector)
    isempty(logs) && return -Inf
    m = maximum(logs)
    isinf(m) && return m
    return m + log(sum(exp(v - m) for v in logs))
end

function _asymgal_component_partial_signlog(tail::AsymGalambosMultiTail, j::Int, x, I::Tuple{Vararg{Int}},)
    k = length(I)
    k > 0 || throw(ArgumentError("partial block must be nonempty"))

    C = _asymgal_subsets(tail.d)[j]
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

    logs = Float64[]
    expected_sign = isodd(length(I)) ? 1 : -1

    for j in axes(tail.β, 2)
        sign, logabs = _asymgal_component_partial_signlog(tail, j, x, I,)
        sign == 0 && continue
        sign == expected_sign || throw(ArgumentError("unexpected asymmetric Galambos component partial sign",))
        push!(logs, logabs)
    end

    isempty(logs) && return 0, -Inf
    return expected_sign, _asymgal_logsumexp(logs)
end




function _asymgal_rand_multivariate!(rng::Distributions.AbstractRNG, tail::AsymGalambosMultiTail, X::AbstractMatrix{T},) where {T<:Real}
    d, n = size(X)
    d == tail.d || throw(DimensionMismatch("output dimension does not match asymmetric Galambos tail dimension",))

    subsets = _asymgal_subsets(d)

    # Work on unit-Fréchet margins. Independent max-stable components add
    # their exponent functions, so their componentwise maximum has the sum
    # of the component STDFs.
    Z = zeros(Float64, d, n)

    @inbounds for j in eachindex(subsets)
        C = subsets[j]
        active = [i for i in C if tail.β[i, j] > 0]
        isempty(active) && continue

        a = tail.α[j]

        # α = 0 is the independence limit. A one-coordinate component is
        # unit-Fréchet irrespective of the Galambos parameter.
        if a == 0 || length(active) == 1
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
        Cgal = ExtremeValueCopula(k, GalambosTail(a))
        U = Random.rand(rng, Cgal, n)

        for (q, i) in enumerate(active)
            βij = Float64(tail.β[i, j])
            for col in 1:n
                candidate = βij / (-log(Float64(U[q, col])))
                if candidate > Z[i, col]
                    Z[i, col] = candidate
                end
            end
        end
    end

    @inbounds for i in 1:d, col in 1:n
        zi = Z[i, col]
        zi > 0 || throw(ArgumentError("asymmetric Galambos weights leave margin $i without a positive component",))
        X[i, col] = T(exp(-inv(zi)))
    end

    return X
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::ExtremeValueCopula{d,<:AsymGalambosMultiTail}, X::AbstractMatrix{T},) where {d,T<:Real}
    size(X, 1) == d || throw(DimensionMismatch("output dimension does not match copula dimension",))
    return _asymgal_rand_multivariate!(rng, C.tail, X)
end
