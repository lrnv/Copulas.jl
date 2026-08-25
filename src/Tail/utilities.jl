# Infrastructure shared by asymmetric models represented as mixtures over all
# nonempty subsets of the margins (currently Tawn and Galambos). Subsets follow
# `_nonempty_subsets(d)`: singletons first, then increasing cardinality.

function _normalize_asymmetric_subset_components(
    d::Int,
    dep::AbstractVector,
    asy::AbstractVector;
    singleton_parameter,
    valid_parameter,
    family::AbstractString,
)
    d >= 2 || throw(ArgumentError("dimension must be at least 2"))
    subsets = _nonempty_subsets(d)
    m = length(subsets)
    length(dep) == m - d || throw(DimensionMismatch(
        "dep must contain one parameter for each non-singleton subset: expected $(m-d)",
    ))
    length(asy) == m || throw(DimensionMismatch(
        "asy must contain one weight vector for each nonempty subset: expected $m",
    ))

    vals = Any[singleton_parameter]
    append!(vals, dep)
    for weights in asy
        weights isa AbstractVector || throw(ArgumentError(
            "each asymmetry component must be an AbstractVector",
        ))
        append!(vals, weights)
    end
    T = promote_type(Float64, map(typeof, vals)...)

    parameters = fill(T(singleton_parameter), m)
    @inbounds for j in (d + 1):m
        parameter = T(dep[j - d])
        valid_parameter(parameter) || throw(ArgumentError(
            "invalid non-singleton $family parameter: $parameter",
        ))
        parameters[j] = parameter
    end

    β = zeros(T, d, m)
    @inbounds for (j, subset) in enumerate(subsets)
        weights = asy[j]
        length(weights) == length(subset) || throw(DimensionMismatch(
            "asy[$j] must have length $(length(subset)) for subset $(Tuple(subset))",
        ))
        for (position, i) in enumerate(subset)
            weight = T(weights[position])
            zero(T) <= weight <= one(T) || throw(ArgumentError(
                "all asymmetry weights must lie in [0,1]",
            ))
            β[i, j] = weight
        end
    end

    tolerance = 64 * eps(T)
    @inbounds for i in 1:d
        rowsum = sum(@view β[i, :])
        abs(rowsum - one(T)) <= tolerance * max(one(T), abs(rowsum)) ||
            throw(ArgumentError(
                "asymmetry weights for margin $i must sum to one; got $rowsum",
            ))
    end
    return parameters, β
end

# Expand the convenience representation containing only the full-set component
# and its weights into the general all-subsets constructor representation.
function _expand_fullset_asymmetric_component(parameter::Real, weights::AbstractVector; singleton_parameter)
    d = length(weights)
    d >= 2 || throw(ArgumentError("weights must contain at least two entries"))
    subsets = _nonempty_subsets(d)
    T = promote_type(Float64, typeof(parameter), eltype(weights))
    w = T.(weights)
    all(value -> zero(T) <= value <= one(T), w) || throw(ArgumentError(
        "all full-set weights must lie in [0,1]",
    ))

    dep = fill(T(singleton_parameter), length(subsets) - d)
    dep[end] = T(parameter)
    asy = [zeros(T, length(subset)) for subset in subsets]
    @inbounds for i in 1:d
        asy[i][1] = one(T) - w[i]
    end
    asy[end] .= w
    return d, dep, asy
end

# Infer d from the required number 2^d - 1 of nonempty-subset weight vectors.
function _dimension_from_asymmetric_subset_weights(asy::AbstractVector, family::AbstractString)
    m = length(asy) + 1
    ispow2(m) || throw(DimensionMismatch(
        "$family asy must contain 2^d-1 subset-weight vectors",
    ))
    d = trailing_zeros(m)
    d >= 2 || throw(ArgumentError("$family dimension must be at least two"))
    return d
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
    Z = zeros(Float64, d, n)

    @inbounds for j in eachindex(subsets)
        active = [i for i in subsets[j] if β[i, j] > 0]
        isempty(active) && continue
        parameter = parameters[j]
        if is_independent(parameter) || length(active) == 1
            for i in active, col in 1:n
                Z[i, col] = max(Z[i, col], Float64(β[i, j]) / Random.randexp(rng))
            end
            continue
        end

        U = rand(rng, component_copula(length(active), parameter), n)
        for (position, i) in enumerate(active), col in 1:n
            candidate = Float64(β[i, j]) / (-log(Float64(U[position, col])))
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
