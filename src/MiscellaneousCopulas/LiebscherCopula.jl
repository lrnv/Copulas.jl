"""
    LiebscherCopula{d}(copulas, weights)
    LiebscherCopula(d, copulas, weights)

The power Liebscher copula in dimension ``d`` is defined as

```math
C(\\boldsymbol u) = \\prod_{k=1}^K C_k\\!\\left(u_1^{a_{k1}},\\ldots,u_d^{a_{kd}}\\right),
```

where `C_1,\\ldots,C_K` are `d`-dimensional copulas and the weights satisfy

```math
a_{kj} \\ge 0, \\qquad \\sum_{k=1}^K a_{kj}=1, \\qquad j=1,\\ldots,d.
```

Notes:

* `copulas` is a tuple of component copulas and `weights` is a `K × d` matrix,
  with one row per component and one column per coordinate.
* All component copulas must have dimension `d`.
* The constructor stores a floating-point copy of `weights`. Small numerical
  deviations from unit column sums are normalized; invalid columns are rejected.
* Zero weights deactivate the corresponding coordinate of a component.
* Supports `cdf`, `logpdf` when the resulting copula is absolutely continuous,
  random sampling, subsetting, conditioning and Rosenblatt transforms.

See also: [`KhoudrajiCopula`](@ref), [`Copula`](@ref), [`subsetdims`](@ref), [`condition`](@ref).

References:

* [liebscher2008](@cite) Liebscher, E. (2008). Construction of asymmetric multivariate copulas. Journal of Multivariate Analysis, 99(10), 2234-2250.
"""
struct LiebscherCopula{d,CT,WT} <: Copula{d}
    copulas::CT
    weights::WT

    function LiebscherCopula{d}(copulas::Tuple, weights::AbstractMatrix{<:Real}) where {d}
        d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))

        K = length(copulas)
        K >= 1 || throw(ArgumentError("LiebscherCopula requires at least one component copula"))

        all(C -> C isa Copula{d}, copulas) || throw(DimensionMismatch("all Liebscher component copulas must have dimension $d"))
        size(weights) == (K, d) || throw(DimensionMismatch("weights must have size ($K, $d); got $(size(weights))"))

        W = Matrix(float.(weights))

        all(isfinite, W) || throw(ArgumentError("Liebscher weights must be finite"))
        all(a -> a >= zero(a), W) || throw(DomainError(weights, "Liebscher weights must be non-negative"))

        @inbounds for j in 1:d
            s = sum(@view W[:, j])
            isapprox(s, one(s)) || throw(ArgumentError("Liebscher weights must sum to one in every dimension; column $j sums to $s"))
            @views W[:, j] ./= s
        end

        return new{d,typeof(copulas),typeof(W)}(copulas, W)
    end
end

LiebscherCopula(d::Integer, copulas::Tuple, weights::AbstractMatrix{<:Real}) = LiebscherCopula{d}(copulas, weights)

(::Type{LiebscherCopula{d,CT,WT}})(copulas::Tuple, weights::AbstractMatrix{<:Real}) where {d,CT,WT} = LiebscherCopula{d}(copulas, weights)

Distributions.params(C::LiebscherCopula) = (; copulas=C.copulas, weights=copy(C.weights))

function Base.eltype(C::LiebscherCopula)
    T = eltype(C.weights)
    for component in C.copulas
        T = promote_type(T, eltype(component))
    end
    return float(T)
end

function _cdf(C::LiebscherCopula{d}, u) where {d}
    T = promote_type(eltype(u), eltype(C))
    v = Vector{T}(undef, d)
    value = one(T)

    @inbounds for k in eachindex(C.copulas)
        active = false

        for j in 1:d
            a = C.weights[k, j]
            if iszero(a)
                v[j] = one(T)
            else
                active = true
                v[j] = u[j]^a
            end
        end

        active || continue
        value *= Distributions.cdf(C.copulas[k], v)
    end

    return value
end

function _liebscher_component_partial(C::Copula{d}, weights, u, observed, differentiated) where {d}
    T = promote_type(eltype(u), eltype(weights))

    for j in differentiated
        iszero(weights[j]) && return zero(T)
    end

    active_observed = Tuple(j for j in observed if !iszero(weights[j]))
    isempty(active_observed) && return one(T)

    z = ntuple(k -> u[active_observed[k]]^weights[active_observed[k]], length(active_observed))
    differentiated_local = Tuple(k for k in eachindex(active_observed) if active_observed[k] in differentiated)

    p = length(differentiated_local)
    q = length(active_observed)

    base_partial = if p == 0
        if q == 1
            z[1]
        else
            Csub = subsetdims(C, active_observed)
            Distributions.cdf(Csub, collect(z))
        end
    elseif p == q
        if q == 1
            one(T)
        else
            Csub = subsetdims(C, active_observed)
            Distributions.pdf(Csub, collect(z))
        end
    else
        Csub = subsetdims(C, active_observed)

        remaining_local = Tuple(k for k in 1:q if k ∉ differentiated_local)
        zremaining = ntuple(k -> z[remaining_local[k]], length(remaining_local))
        zdifferentiated = ntuple(k -> z[differentiated_local[k]], p)

        _partial_cdf(
            Csub,
            remaining_local,
            differentiated_local,
            zremaining,
            zdifferentiated,
        )
    end

    chain = one(T)
    for j in differentiated
        a = weights[j]
        chain *= a * u[j]^(a - one(a))
    end

    return chain * base_partial
end

function _partial_cdf(C::LiebscherCopula{d}, is, js, uᵢₛ, uⱼₛ) where {d}
    u = _assemble(d, is, js, uᵢₛ, uⱼₛ)
    p = length(js)
    nstates = 1 << p
    T = promote_type(eltype(u), eltype(C))
    observed = (is..., js...)

    current = zeros(T, nstates)
    current[1] = one(T)

    for k in eachindex(C.copulas)
        factor = zeros(T, nstates)
        weights = @view C.weights[k, :]

        for mask in 0:(nstates - 1)
            differentiated = Tuple(js[r] for r in 1:p if ((mask >> (r - 1)) & 1) == 1)
            factor[mask + 1] = _liebscher_component_partial(C.copulas[k], weights, u, observed, differentiated)
        end

        next = zeros(T, nstates)

        for mask in 0:(nstates - 1)
            submask = mask

            while true
                complement = mask ⊻ submask
                next[mask + 1] += current[complement + 1] * factor[submask + 1]

                submask == 0 && break
                submask = (submask - 1) & mask
            end
        end

        current = next
    end

    return current[end]
end

function Distributions._logpdf(C::LiebscherCopula{d}, u::AbstractVector{<:Real}) where {d}
    copula_measure_style(C) isa AbsolutelyContinuousMeasure || throw(ArgumentError("a global Lebesgue density is not defined for this LiebscherCopula"))

    value = _partial_cdf(C, (), ntuple(identity, d), (), Tuple(u))

    iszero(value) && return oftype(value, -Inf)
    value < zero(value) && return oftype(value, NaN)

    return log(value)
end

function Distributions._rand!(rng::Distributions.AbstractRNG, C::LiebscherCopula{d}, U::AbstractMatrix{T}) where {d,T<:Real}
    fill!(U, zero(T))
    V = similar(U)

    @inbounds for k in eachindex(C.copulas)
        active = any(j -> !iszero(C.weights[k, j]), 1:d)
        active || continue

        Distributions._rand!(rng, C.copulas[k], V)

        for j in 1:d
            a = C.weights[k, j]
            iszero(a) && continue

            inva = inv(a)

            for n in axes(U, 2)
                U[j, n] = max(U[j, n], V[j, n]^inva)
            end
        end
    end

    return U
end

function copula_measure_style(C::LiebscherCopula{d}) where {d}
    @inbounds for k in eachindex(C.copulas)
        active_dims = Tuple(j for j in 1:d if !iszero(C.weights[k, j]))
        length(active_dims) <= 1 && continue

        component = subsetdims(C.copulas[k], active_dims)
        style = copula_measure_style(component)

        style isa NonAbsolutelyContinuousMeasure && return style
    end

    return AbsolutelyContinuousMeasure()
end

function SubsetCopula(C::LiebscherCopula{d}, dims::NTuple{p,Int}) where {d,p}
    dims == Tuple(1:d) && return C
    p == 1 && return Distributions.Uniform()

    copulas = ntuple(k -> subsetdims(C.copulas[k], dims), length(C.copulas))
    weights = C.weights[:, collect(dims)]

    return LiebscherCopula{p}(copulas, weights)
end

function _khoudraji_weights(::Val{d}, shapes) where {d}
    length(shapes) == d || throw(DimensionMismatch("Khoudraji shapes must have length $d; got $(length(shapes))"))

    α = collect(float.(shapes))

    all(isfinite, α) || throw(ArgumentError("Khoudraji shapes must be finite"))
    all(a -> zero(a) <= a <= one(a), α) || throw(DomainError(shapes, "Khoudraji shapes must lie in [0, 1]"))

    T = eltype(α)
    W = Matrix{T}(undef, 2, d)

    @inbounds for j in 1:d
        W[1, j] = one(T) - α[j]
        W[2, j] = α[j]
    end

    return W
end

"""
    KhoudrajiCopula(d, C, shapes)
    KhoudrajiCopula(d, (C1, C2), shapes)

Construct a Khoudraji copula in dimension ``d`` as a power [`LiebscherCopula`](@ref).

For two component copulas ``C_1`` and ``C_2``,

```math
C_K(\\boldsymbol u) = C_1\\!\\left(u_1^{1-\\alpha_1},\\ldots,u_d^{1-\\alpha_d}\\right)C_2\\!\\left(u_1^{\\alpha_1},\\ldots,u_d^{\\alpha_d}\\right),
````

where `0 \\le \\alpha_j \\le 1` for `j=1,\\ldots,d`.

Notes:

* `shapes` contains the `d` parameters `\\alpha_1,\\ldots,\\alpha_d`.
* `KhoudrajiCopula(d, C, shapes)` uses [`IndependentCopula`](@ref) as the first component.
* `KhoudrajiCopula` is a convenience constructor and returns a [`LiebscherCopula`](@ref); it does not define a separate copula type.
* All probability, sampling, subsetting, conditioning and Rosenblatt operations are therefore inherited from `LiebscherCopula`.

See also: [`LiebscherCopula`](@ref), [`IndependentCopula`](@ref).

References:

* [khoudraji1995](@cite) Khoudraji, A. (1995). Contributions à l'étude des copules et à la modélisation des valeurs extrêmes bivariées. PhD thesis, Université Laval, Québec, Canada.
* [liebscher2008](@cite) Liebscher, E. (2008). Construction of asymmetric multivariate copulas. Journal of Multivariate Analysis, 99(10), 2234-2250.
"""
function KhoudrajiCopula(d::Integer, C::Copula, shapes::Union{Tuple,AbstractVector})
    C isa Copula{d} || throw(DimensionMismatch("the Khoudraji component copula must have dimension $d"))

    weights = _khoudraji_weights(Val(d), shapes)
    return LiebscherCopula(d, (IndependentCopula{d}(), C), weights)
end

function KhoudrajiCopula(d::Integer, copulas::Tuple{<:Copula,<:Copula}, shapes::Union{Tuple,AbstractVector})
    all(C -> C isa Copula{d}, copulas) || throw(DimensionMismatch("all Khoudraji component copulas must have dimension $d"))

    weights = _khoudraji_weights(Val(d), shapes)
    return LiebscherCopula(d, copulas, weights)
end

#########################################
# Fitting template                      #
#########################################

_available_fitting_methods(::Type{<:LiebscherCopula}, d) = ()

function _liebscher_component_is_fittable(C::Copula{d}) where {d}
    θ = Distributions.params(C)
    return !isempty(θ) && (:mle in _available_fitting_methods(typeof(C), d))
end

function _liebscher_component_unbound(C::Copula{d}) where {d}
    _liebscher_component_is_fittable(C) || return Float64[]
    return _unbound_params(typeof(C), d, Distributions.params(C))
end

function _liebscher_weights_unbound(W::AbstractMatrix{<:Real})
    T = float(eltype(W))
    α = T[]
    K, d = size(W)

    @inbounds for j in 1:d
        active = [k for k in 1:K if !iszero(W[k, j])]
        length(active) <= 1 && continue
        reference = active[1]

        for k in active[2:end]
            push!(α, log(W[k, j] / W[reference, j]))
        end
    end

    return α
end

function _liebscher_unbound(C::LiebscherCopula)
    T = eltype(C)
    α = T[]

    for component in C.copulas
        append!(α, _liebscher_component_unbound(component))
    end

    append!(α, _liebscher_weights_unbound(C.weights))
    return α
end

function _liebscher_rebound_component(C::Copula{d}, α, i::Ref{Int}) where {d}
    _liebscher_component_is_fittable(C) || return C

    n = length(_liebscher_component_unbound(C))
    iszero(n) && return C

    β = @view α[i[]:(i[] + n - 1)]
    θ = _rebound_params(typeof(C), d, β)
    fitted = _construct_fitted_copula(typeof(C), Val(d), θ, C)

    i[] += n
    return fitted
end

function _liebscher_weights_rebound(W0::AbstractMatrix{<:Real}, α, i::Ref{Int})
    K, d = size(W0)
    T = promote_type(eltype(W0), eltype(α))
    W = zeros(T, K, d)

    @inbounds for j in 1:d
        active = [k for k in 1:K if !iszero(W0[k, j])]
        m = length(active)

        if m == 1
            W[only(active), j] = one(T)
            continue
        end

        η = Vector{T}(undef, m)
        η[1] = zero(T)

        for r in 2:m
            η[r] = α[i[]]
            i[] += 1
        end

        shift = maximum(η)
        e = exp.(η .- shift)
        den = sum(e)

        for r in 1:m
            W[active[r], j] = e[r] / den
        end
    end

    return W
end

function _liebscher_rebound(C0::LiebscherCopula{d}, α) where {d}
    i = Ref(1)
    copulas = ntuple(k -> _liebscher_rebound_component(C0.copulas[k], α, i), length(C0.copulas))
    weights = _liebscher_weights_rebound(C0.weights, α, i)

    i[] == length(α) + 1 || throw(ArgumentError("invalid Liebscher fitting parameter vector"))

    return LiebscherCopula(d, copulas, weights)
end

function _validate_liebscher_fit_data(U, d::Int)
    ndims(U) == 2 || throw(ArgumentError("U must be a d×n matrix of pseudo-observations"))
    size(U, 1) == d || throw(DimensionMismatch("data dimension $(size(U, 1)) does not match copula dimension $d"))
    size(U, 2) > 0 || throw(ArgumentError("U must contain at least one observation"))
    all(isfinite, U) || throw(ArgumentError("pseudo-observations must be finite"))
    all(0 .< U .< 1) || throw(ArgumentError("pseudo-observations must lie strictly inside (0, 1)"))
    return nothing
end

function _fit_liebscher(C0::LiebscherCopula, U)
    copula_measure_style(C0) isa AbsolutelyContinuousMeasure || throw(ArgumentError("LiebscherCopula template fitting requires an absolutely continuous starting model"))

    α0 = _liebscher_unbound(C0)
    isempty(α0) && return C0

    reconstruct(α) = _liebscher_rebound(C0, α)

    loss(α) = begin
        C = reconstruct(α)
        copula_measure_style(C) isa AbsolutelyContinuousMeasure || return oftype(first(α), Inf)

        ℓ = Distributions.loglikelihood(C, U)
        isfinite(ℓ) || return oftype(first(α), Inf)

        return -ℓ
    end

    result = Optim.optimize(loss, α0, Optim.LBFGS(); autodiff=ADTypes.AutoForwardDiff())

    return reconstruct(Optim.minimizer(result))
end

function Distributions.fit(::Type{CopulaModel}, C0::LiebscherCopula{d}, U; method=:mle, kwargs...) where {d}
    _reject_inference_fit_keywords((; kwargs...))
    isempty(kwargs) || throw(ArgumentError("unsupported LiebscherCopula fitting keyword(s): $(join(keys(kwargs), ", "))"))
    method === :mle || throw(ArgumentError("LiebscherCopula template fitting supports only method=:mle (got $method)"))

    _validate_liebscher_fit_data(U, d)

    fitted = _fit_liebscher(C0, U)
    fit_spec = _CopulaFitSpec(C0, :mle, (;))
    ll = Distributions.loglikelihood(fitted, U)

    return CopulaModel(fitted, U, ll, fit_spec)
end

function Distributions.fit(C0::LiebscherCopula{d}, U; method=:mle, kwargs...) where {d}
    return fitted_distribution(Distributions.fit(CopulaModel, C0, U; method, kwargs...))
end

function _natural_parameters(C::LiebscherCopula{d}) where {d}
    names = String[]
    values = Any[]

    for (k, component) in pairs(C.copulas)
        _liebscher_component_is_fittable(component) || continue
        component_names, component_values = _natural_parameters(component)

        for i in eachindex(component_values)
            push!(names, "C$(k)_$(component_names[i])")
            push!(values, component_values[i])
        end
    end

    K = length(C.copulas)

    @inbounds for j in 1:d
        active = [k for k in 1:K if !iszero(C.weights[k, j])]
        length(active) <= 1 && continue

        for k in active[2:end]
            push!(names, "a$(k)_$(j)")
            push!(values, C.weights[k, j])
        end
    end

    θ = isempty(values) ? Float64[] : collect(promote(float.(values)...))
    return names, θ
end