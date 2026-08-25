"""
    Tail

Abstract type. Implements the API for stable tail dependence functions (STDFs) of extreme-value copulas in dimension `d`.

A STDF is a function
``\\ell : \\mathbb{R}_{+}^d → [0,\\infty)`` that is 1-homogeneous (``\\ell(t·x)=t·\\ell(x)`` for all ``t≥0``), convex, 
and satisfies the bounds
``\\max(x_1,\\ldots,x_d) ≤ \\ell(x) ≤ x_1+ \\cdots +x_d`` (in particular ``\\ell(e_i)=1``).

Pickands representation. By homogeneity, for ``x\\neq 0`` let ``\\left\\| x\\right\\|_1=x_1+\\cdots+x_d`` and
``\\omega=x/\\left\\| x \\right\\|_1 \\in \\Delta_{d-1}``. There exists a Pickands dependence function
``A:\\Delta_{d-1}\\to [0,1]`` (convex, ``\\max(\\omega_i)≤A(\\omega)≤1``) such that
``\\ell(x)=\\left\\| x\\right\\|_1·A(\\omega)``. For ``d=2``, ``A`` reduces to a convex function on ``[0,1]`` with
``\\max(t,1-t)≤A(t)≤1`` and ``A(0)=A(1)=1``.

Interface.
- `A(tail::Tail, ω::NTuple{d,Real})` — Pickands function on the simplex `\\Delta_{d-1}`.
  (For `d=2`, a convenience `A(tail::Tail{2}, t::Real)` may be provided.)
- `ℓ(tail::Tail, x::NTuple{d,Real})` — STDF. By default the package defines
  `ℓ(tail, x) = ‖x‖₁ * A(tail, x/‖x‖₁)` when `A` is available.

We do not algorithmically verify convexity/bounds; implementers are responsible for validity.

Additional helpers (with defaults).
- For `d=2`: `dA`, `d²A` via AD; stable `logpdf`/`rand` (Ghoudi sampler).
- In any `d`: `cdf(u) = exp(-ℓ(-log.(u)))`.

References:
* Pickands (1981); Gudendorf & Segers (2010); Ghoudi, Khoudraji & Rivest (1998); de Haan & Ferreira (2006).
* Rasell
"""
abstract type Tail end
function (TT::Type{<:Tail})(args...;kwargs...)
    S = hasproperty(TT, :body) ? TT.body : TT
    T = S.name.wrapper 
    return T(args..., values(kwargs)...)
end
Base.broadcastable(tail::Tail) = Ref(tail)

####### Functions you need to overload: 
_is_valid_in_dim(::Tail, d::Int) = d >= 2
A(::Tail, ω::NTuple{d,<:Real}) where {d} = throw(ArgumentError("Implement A(Tail{$d}, ω) en el simplex Δ_{d-1}"))

####### Rest of the interface you can overload if more efficient:
needs_binary_search(::Tail) = false
# \ell function
function ℓ(tail::Tail, x)
    s = sum(x)
    return s == 0 ? zero(eltype(x)) : s * A(tail, ntuple(i->x[i]/s, length(x)))
end

# Mixed STDF partials. A new Tail only needs to implement ℓ.
# Generic mixed partials come from the shared AD helper.
function _ellpartial_signlog(tail::Tail, x, I::Tuple{Vararg{Int}})
    v = _mixed_partial(z -> ℓ(tail, z), x, I)
    iszero(v) && return 0, oftype(v, -Inf)
    return v < zero(v) ? -1 : 1, log(abs(v))
end

_ellpartial_signlog(tail::Tail, x, I::AbstractVector{<:Integer}) = _ellpartial_signlog(tail, x, Tuple(I))

function ellpartial(tail::Tail, x, I::Tuple{Vararg{Int}})
    isempty(I) && return ℓ(tail, x)
    sign, logabs = _ellpartial_signlog(tail, x, I)
    return iszero(sign) ? zero(logabs) : sign * exp(logabs)
end

ellpartial(tail::Tail, x, I::AbstractVector{<:Integer}) = ellpartial(tail, x, Tuple(I))

function _subset_model_parameters(
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

function _fullset_subset_parameters(parameter::Real, weights::AbstractVector; singleton_parameter)
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

function _subset_dimension(asy::AbstractVector, family::AbstractString)
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

# Native scalar Pickands interface in d=2.
#
# `BivariatePickandsTail` is a computational capability: the tail provides the
# scalar Pickands representation A(t) and therefore has access to the mature
# bivariate derivative, density, conditioning, and sampling machinery.
#
# The capability is bivariate by default. Mathematical families that also have
# a valid multivariate STDF override `_is_valid_in_dim`.
abstract type BivariatePickandsTail <: Tail end

# Marker used by fitting routines for one-parameter Pickands families.
abstract type OneParameterPickandsTail <: BivariatePickandsTail end

_is_valid_in_dim(::BivariatePickandsTail, d::Int) = d == 2
A(tail::BivariatePickandsTail, t::NTuple{2, <:Real}) = A(tail, t[1])
dA(tail::BivariatePickandsTail, t::Real) = ForwardDiff.derivative(z -> A(tail, z), t)
d²A(tail::BivariatePickandsTail, t::Real) = ForwardDiff.derivative(z -> dA(tail, z), t)

# One-sided Pickands slopes for conditional endpoint extensions.
_pickands_left_slope(tail::BivariatePickandsTail, x::Real) = dA(tail, _safett(zero(x)))
_pickands_right_slope(tail::BivariatePickandsTail, x::Real) = dA(tail, _safett(one(x)))

_A_dA_d²A(tail::BivariatePickandsTail, t::Real) = let tt = _safett(t); (A(tail, tt), dA(tail, tt), d²A(tail, tt)) end
function _biv_der_ℓ(tail::BivariatePickandsTail, uv)
    u, v = uv
    s  = u + v
    x  = u / s
    y  = v / s
    a, da, d2a = _A_dA_d²A(tail, x)
    val  = s * a
    du   = a + da * y
    dv   = a - x * da
    dudv = - x * y * d2a / s
    return val, du, dv, dudv
end
function _probability_z(tail::BivariatePickandsTail, z::Real)
    # p(z) = z(1-z) A''(z) / [ A(z) g_Z(z) ] 
    num = z * (1 - z) * d²A(tail, z) 
    dem = A(tail, z) * _pdf(ExtremeDist(tail), z) # usa pdf, no _pdf 
    p = num / dem 
    return clamp(p, 0, 1) 
end
