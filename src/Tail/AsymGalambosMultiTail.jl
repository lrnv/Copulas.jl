
# General multivariate asymmetric Galambos / negative-logistic tail.
#
# Each nonempty subset C carries a weighted Galambos component.  With
# β[i,C] ≥ 0 and Σ_{C∋i} β[i,C] = 1,
#
#     ℓ(x) = Σ_C ℓ_Galambos,α[C]((β[i,C] x[i])_{i∈C}).
#
# Singleton components and α[C] == 0 components are interpreted as linear
# (independent) exponent contributions.
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

function AsymGalambosMultiTail(
    d::Int,
    dep::AbstractVector,
    asy::AbstractVector,
)
    subsets = _asymgal_subsets(d)
    m = length(subsets)

    length(dep) == m - d || throw(DimensionMismatch(
        "dep must contain one parameter for each non-singleton subset: expected $(m-d)",
    ))
    length(asy) == m || throw(DimensionMismatch(
        "asy must contain one weight vector for each nonempty subset: expected $m",
    ))

    vals = Any[1.0]
    append!(vals, dep)
    for w in asy
        w isa AbstractVector || throw(ArgumentError(
            "each asymmetry component must be an AbstractVector",
        ))
        append!(vals, w)
    end
    T = promote_type(Float64, map(typeof, vals)...)

    α = zeros(T, m)
    @inbounds for j in (d + 1):m
        a = T(dep[j - d])
        a >= zero(T) || throw(ArgumentError(
            "each non-singleton Galambos parameter must be ≥ 0",
        ))
        α[j] = a
    end

    β = zeros(T, d, m)
    @inbounds for (j, C) in enumerate(subsets)
        w = asy[j]
        length(w) == length(C) || throw(DimensionMismatch(
            "asy[$j] must have length $(length(C)) for subset $(Tuple(C))",
        ))
        for (a, i) in enumerate(C)
            wij = T(w[a])
            zero(T) <= wij <= one(T) || throw(ArgumentError(
                "all asymmetry weights must lie in [0,1]",
            ))
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

    return AsymGalambosMultiTail{T}(d, α, β)
end

# Convenience submodel: one full-set Galambos component plus singleton
# remainders.  In d=2 this matches the historical AsymGalambosTail using
# weights in the same coordinate order: [θ₁, θ₂].
function AsymGalambosMultiTail(α::Real, weights::AbstractVector)
    d = length(weights)
    d >= 2 || throw(ArgumentError(
        "weights must contain at least two entries",
    ))

    subsets = _asymgal_subsets(d)
    m = length(subsets)
    T = promote_type(Float64, typeof(α), eltype(weights))
    a = T(α)
    a >= zero(T) || throw(ArgumentError("α must be ≥ 0"))

    w = T.(weights)
    all(v -> zero(T) <= v <= one(T), w) ||
        throw(ArgumentError("all full-set weights must lie in [0,1]"))

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

Distributions.params(tail::AsymGalambosMultiTail) =
    (α = tail.α, β = tail.β)

_is_valid_in_dim(tail::AsymGalambosMultiTail, d::Int) = d == tail.d

function _asymgal_component_data(tail::AsymGalambosMultiTail, j, x)
    C = _asymgal_subsets(tail.d)[j]
    active = [i for i in C if tail.β[i, j] > 0]
    y = [tail.β[i, j] * x[i] for i in active]
    return C, active, y
end

function ℓ(tail::AsymGalambosMultiTail, x)
    length(x) == tail.d || throw(DimensionMismatch(
        "input dimension does not match asymmetric Galambos tail dimension",
    ))

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

function _asymgal_component_partial_signlog(
    tail::AsymGalambosMultiTail,
    j::Int,
    x,
    I::Tuple{Vararg{Int}},
)
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

    sign, logabs = _ellpartial_signlog(
        GalambosTail(a),
        y,
        localI,
    )
    sign == 0 && return 0, -Inf

    logchain = sum(log(float(tail.β[i, j])) for i in I)
    return sign, logabs + logchain
end

function _ellpartial_signlog(
    tail::AsymGalambosMultiTail,
    x,
    I::Tuple{Vararg{Int}},
)
    isempty(I) && return 1, log(float(ℓ(tail, x)))

    logs = Float64[]
    expected_sign = isodd(length(I)) ? 1 : -1

    for j in axes(tail.β, 2)
        sign, logabs = _asymgal_component_partial_signlog(
            tail,
            j,
            x,
            I,
        )
        sign == 0 && continue
        sign == expected_sign || throw(ArgumentError(
            "unexpected asymmetric Galambos component partial sign",
        ))
        push!(logs, logabs)
    end

    isempty(logs) && return 0, -Inf
    return expected_sign, _asymgal_logsumexp(logs)
end

_ellpartial_signlog(
    tail::AsymGalambosMultiTail,
    x,
    I::AbstractVector{<:Integer},
) = _ellpartial_signlog(tail, x, Tuple(I))

function ellpartial(
    tail::AsymGalambosMultiTail,
    x,
    I::Tuple{Vararg{Int}},
)
    isempty(I) && return ℓ(tail, x)
    sign, logabs = _ellpartial_signlog(tail, x, I)
    sign == 0 && return zero(float(first(x)))
    return sign * exp(logabs)
end

ellpartial(
    tail::AsymGalambosMultiTail,
    x,
    I::AbstractVector{<:Integer},
) = ellpartial(tail, x, Tuple(I))

Distributions._logpdf(
    C::ExtremeValueCopula{d,<:AsymGalambosMultiTail},
    u,
) where {d} = _ev_logpdf_from_partials(C, u)

# Resolve intersection with the generic bivariate EV density.
Distributions._logpdf(
    C::ExtremeValueCopula{2,<:AsymGalambosMultiTail},
    u,
) = _ev_logpdf_from_partials(C, u)


function _asymgal_rand_multivariate!(
    rng::Distributions.AbstractRNG,
    tail::AsymGalambosMultiTail,
    X::AbstractMatrix{T},
) where {T<:Real}
    d, n = size(X)
    d == tail.d || throw(DimensionMismatch(
        "output dimension does not match asymmetric Galambos tail dimension",
    ))

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
        zi > 0 || throw(ArgumentError(
            "asymmetric Galambos weights leave margin $i without a positive component",
        ))
        X[i, col] = T(exp(-inv(zi)))
    end

    return X
end

_rand_ev_multivariate!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{d,<:AsymGalambosMultiTail},
    X::AbstractMatrix{T},
) where {d,T<:Real} =
    _asymgal_rand_multivariate!(rng, C.tail, X)

# The generic d=2 EV sampler uses Pickands A/dA. This tail is represented
# directly through its multivariate STDF, so force the exact max-mixture
# sampler in dimension two as well.
function Distributions._rand!(
    rng::Distributions.AbstractRNG,
    C::ExtremeValueCopula{2,<:AsymGalambosMultiTail},
    X::AbstractMatrix{T},
) where {T<:Real}
    size(X, 1) == 2 || throw(DimensionMismatch(
        "output must have two rows for a bivariate asymmetric Galambos copula",
    ))
    return _asymgal_rand_multivariate!(rng, C.tail, X)
end
