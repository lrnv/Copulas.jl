# Infrastructure shared by asymmetric models represented as mixtures over all
# nonempty subsets of the margins (currently Tawn and Galambos). Subsets follow
# `_nonempty_subsets(d)`: singletons first, then increasing cardinality.

_component_eltype(::Type{<:AbstractVector{T}}) where {T} = T
_component_eltype(::Type) = Any

@inline _asymmetric_dependence_count(d::Int) = 2^d - d - 1
@inline _asymmetric_margin_weight_count(d::Int) = 2^(d - 1)

function _normalize_asymmetric_margin_components(
    d::Int,
    dep::AbstractVector,
    weights;
    singleton_parameter,
    valid_parameter,
    family::AbstractString,
)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    q = _asymmetric_dependence_count(d)
    nweights = _asymmetric_margin_weight_count(d)
    length(dep) == q || throw(DimensionMismatch(
        "dep must contain one parameter for each non-singleton subset: expected $q",
    ))
    length(weights) == d || throw(DimensionMismatch(
        "weights must contain one simplex for each margin: expected $d",
    ))
    all(weight -> weight isa AbstractVector, weights) || throw(ArgumentError(
        "each margin weight block must be an AbstractVector",
    ))
    all(weight -> length(weight) == nweights, weights) || throw(DimensionMismatch(
        "each margin weight block must contain $nweights entries",
    ))

    T = float(promote_type(
        typeof(singleton_parameter),
        eltype(dep),
        (eltype(weight) for weight in weights)...,
    ))

    normalized_dep = Vector{T}(undef, q)
    @inbounds for j in eachindex(dep)
        parameter = T(dep[j])
        valid_parameter(parameter) || throw(ArgumentError(
            "invalid non-singleton $family parameter: $parameter",
        ))
        normalized_dep[j] = parameter
    end

    normalized_weights = Vector{Vector{T}}(undef, d)
    tolerance = 64 * eps(T)
    @inbounds for i in 1:d
        source = weights[i]
        target = Vector{T}(undef, nweights)
        for k in eachindex(source)
            weight = T(source[k])
            zero(T) <= weight <= one(T) || throw(ArgumentError(
                "all asymmetry weights must lie in [0,1]",
            ))
            target[k] = weight
        end
        total = sum(target)
        abs(total - one(T)) <= tolerance * max(one(T), abs(total)) ||
            throw(ArgumentError(
                "asymmetry weights for margin $i must sum to one; got $total",
            ))
        normalized_weights[i] = target
    end
    return normalized_dep, normalized_weights
end

# Convert the historical subset-oriented representation into the canonical
# margin-oriented representation. For each margin i, the resulting vector lists
# β_{i,C} in the same global subset order, restricted to subsets C containing i.
function _subset_asymmetry_to_margin_weights(d::Int, asy::AbstractVector)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    subsets = _nonempty_subsets(d)
    length(asy) == length(subsets) || throw(DimensionMismatch(
        "asy must contain one weight vector for each nonempty subset: expected $(length(subsets))",
    ))
    all(weight -> weight isa AbstractVector, asy) || throw(ArgumentError(
        "each asymmetry component must be an AbstractVector",
    ))

    T = float(promote_type((eltype(weight) for weight in asy)...))
    nweights = _asymmetric_margin_weight_count(d)
    weights = [Vector{T}() for _ in 1:d]
    foreach(weight -> sizehint!(weight, nweights), weights)

    @inbounds for (j, subset) in enumerate(subsets)
        source = asy[j]
        length(source) == length(subset) || throw(DimensionMismatch(
            "asy[$j] must have length $(length(subset)) for subset $(Tuple(subset))",
        ))
        for (position, i) in enumerate(subset)
            push!(weights[i], T(source[position]))
        end
    end
    return weights
end

# Materialize the subset-oriented arrays consumed by the numerical kernels from
# the canonical algebraic parameters. The singleton dependence parameter is
# structural and is therefore not stored in `dep`.
function _asymmetric_subset_components(
    d::Int,
    dep::AbstractVector,
    weights;
    singleton_parameter,
)
    subsets = _nonempty_subsets(d)
    m = length(subsets)
    T = float(promote_type(
        typeof(singleton_parameter),
        eltype(dep),
        (eltype(weight) for weight in weights)...,
    ))

    parameters = fill(T(singleton_parameter), m)
    @inbounds for j in eachindex(dep)
        parameters[d + j] = T(dep[j])
    end

    β = zeros(T, d, m)
    positions = zeros(Int, d)
    @inbounds for (j, subset) in enumerate(subsets)
        for i in subset
            positions[i] += 1
            β[i, j] = T(weights[i][positions[i]])
        end
    end
    return parameters, β
end

# Expand the convenience representation containing only the full-set component
# and singleton remainders into the canonical margin-oriented representation.
# The first entry of every margin simplex corresponds to its singleton and the
# final entry to the full set.
function _expand_fullset_asymmetric_component(
    parameter::Real,
    weights::AbstractVector;
    singleton_parameter,
)
    d = length(weights)
    T = float(promote_type(typeof(parameter), typeof(singleton_parameter), eltype(weights)))
    q = _asymmetric_dependence_count(d)
    nweights = _asymmetric_margin_weight_count(d)

    dep = fill(T(singleton_parameter), q)
    isempty(dep) || (dep[end] = T(parameter))
    margin_weights = [zeros(T, nweights) for _ in 1:d]
    @inbounds for i in 1:d
        w = T(weights[i])
        margin_weights[i][1] = one(T) - w
        margin_weights[i][end] = w
    end
    return (dep, margin_weights...)
end

function _sum_component_partials(component, count::Int, expected_sign::Int)
    logs = Float64[]
    @inbounds for j in 1:count
        sign, logabs = component(j)
        iszero(sign) && continue
        sign == expected_sign || throw(ArgumentError("unexpected component partial sign"))
        push!(logs, logabs)
    end
    isempty(logs) && return 0, -Inf
    return expected_sign, LogExpFunctions.logsumexp(logs)
end

function _rand_subset_components!(
    rng::Distributions.AbstractRNG,
    X::AbstractMatrix{T},
    parameters,
    β,
    is_independent,
    component_copula;
    family::AbstractString,
) where {T<:Real}
    d, n = size(X)
    subsets = _nonempty_subsets(d)
    S = promote_type(T, eltype(parameters), eltype(β))
    Z = zeros(S, d, n)

    @inbounds for j in eachindex(subsets)
        active = [i for i in subsets[j] if β[i, j] > 0]
        isempty(active) && continue
        parameter = parameters[j]
        if is_independent(parameter) || length(active) == 1
            for i in active, col in 1:n
                Z[i, col] = max(Z[i, col], S(β[i, j]) / Random.randexp(rng, S))
            end
            continue
        end

        U = rand(rng, component_copula(length(active), parameter), n)
        for (position, i) in enumerate(active), col in 1:n
            candidate = S(β[i, j]) / (-log(S(U[position, col])))
            Z[i, col] = max(Z[i, col], candidate)
        end
    end

    @inbounds for i in 1:d, col in 1:n
        Z[i, col] > 0 || throw(ArgumentError(
            "$family weights leave margin $i without a positive component",
        ))
        X[i, col] = T(exp(-inv(Z[i, col])))
    end
    return X
end
