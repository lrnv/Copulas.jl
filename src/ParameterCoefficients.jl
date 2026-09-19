###############################################################################
##### Natural coefficient metadata from Paramorph spaces
###############################################################################

# Paramorph owns the logical parameter structure. A Sklar model is exactly the
# Cartesian product of the copula space and one marginal space per coordinate;
# prefixes are metadata only and therefore do not alter any transformation.
function Paramorph.param_space(S::SklarDist)
    copula_space = Paramorph.Prefixed(:copula, Paramorph.param_space(S.C))
    margin_spaces = ntuple(length(S.m)) do i
        Paramorph.Prefixed(Symbol("margin_", string(i)), Paramorph.param_space(S.m[i]))
    end
    return (copula_space, margin_spaces...)
end

# Empirical observations are structural state, not finite-dimensional model
# parameters. Reflections likewise add no geometry of their own.
Paramorph.param_space(::EmpiricalCopula) = ()
Paramorph.param_space(S::AbstractReflectedCopula) =
    Paramorph.param_space(basecopula(S))

# Return the constrained logical values corresponding to `p` as one flat tuple.
# `Distributions.params` is the canonical source whenever it already contains
# exactly those logical values. The property fallback covers distributions with
# structural constructor arguments (for example Binomial's trial count), which
# Paramorph deliberately omits from the free parameter space.
function _coefficient_parameter_values(D, p)
    p isa Paramorph.Prefixed &&
        return _coefficient_parameter_values(D, p.space)

    n = length(Paramorph.names(p))
    iszero(n) && return ()

    raw = Distributions.params(D)
    params = raw isa NamedTuple ? Tuple(values(raw)) : Tuple(raw)
    length(params) == n && return params

    names = Paramorph.names(p)
    if all(name -> hasproperty(D, name), names)
        return ntuple(i -> getproperty(D, names[i]), n)
    end

    # Generic distribution wrappers in Paramorph (truncation, censoring,
    # order-statistics, ...) prefix the space of an underlying distribution and
    # keep wrapper configuration structural. If the direct parameters do not
    # match, find the child distribution whose logical space does.
    if D isa Distributions.Distribution
        for field in propertynames(D)
            child = getproperty(D, field)
            child isa Distributions.Distribution || continue
            child_space = try
                Paramorph.param_space(child)
            catch
                continue
            end
            Paramorph.names(child_space) == names || continue
            return _coefficient_parameter_values(child, child_space)
        end
    end

    throw(DimensionMismatch(
        "could not match $(typeof(D)) parameters to Paramorph names $(names)"))
end

function _coefficient_parameter_values(S::SklarDist, p::Tuple)
    components = (S.C, S.m...)
    length(components) == length(p) || throw(DimensionMismatch(
        "Sklar parameter space must contain one copula block and one block per margin"))
    values = Any[]
    for (component, space) in zip(components, p)
        append!(values, _coefficient_parameter_values(component, space))
    end
    return Tuple(values)
end

_coefficient_parameter_values(S::AbstractReflectedCopula, p) =
    _coefficient_parameter_values(basecopula(S), p)

@inline function _parameter_prefix(prefix::String, part)
    text = string(part)
    isempty(prefix) && return text
    isempty(text) && return prefix
    return string(prefix, "_", text)
end

@inline function _indexed_parameter_name(name::String, i::Int, n::Int)
    return n > 9 ? "$(name)_$(i)" : "$(name)$(Char(0x2080 + i))"
end

@inline function _indexed_parameter_name(name::String, i::Int, j::Int, n::Int)
    return n > 9 ? "$(name)_$(i)_$(j)" :
           "$(name)$(Char(0x2080 + i))$(Char(0x2080 + j))"
end

function _append_logical_coefficient!(names, values, value, name::String)
    if value isa Number
        push!(names, name)
        push!(values, value)
    elseif value isa AbstractVector
        n = length(value)
        for i in eachindex(value)
            push!(names, _indexed_parameter_name(name, Int(i), n))
            push!(values, value[i])
        end
    elseif value isa AbstractMatrix
        n = maximum(size(value))
        for j in axes(value, 2), i in axes(value, 1)
            push!(names, _indexed_parameter_name(name, Int(i), Int(j), n))
            push!(values, value[i, j])
        end
    elseif value isa NamedTuple
        for (key, child) in pairs(value)
            _append_logical_coefficient!(
                names, values, child, _parameter_prefix(name, key))
        end
    elseif value isa Tuple
        for (i, child) in pairs(value)
            _append_logical_coefficient!(
                names, values, child, _parameter_prefix(name, i))
        end
    else
        throw(ArgumentError(
            "cannot expose logical parameter $(name) with value type $(typeof(value))"))
    end
    return nothing
end

function _append_space_coefficients!(names, values, p::Tuple,
        logical::Tuple, prefix::String)
    offset = 0
    for q in p
        n = length(Paramorph.names(q))
        qvalues = ntuple(i -> logical[offset + i], n)
        _append_space_coefficients!(names, values, q, qvalues, prefix)
        offset += n
    end
    offset == length(logical) || throw(DimensionMismatch(
        "parameter-space product and logical values have different lengths"))
    return nothing
end

function _append_space_coefficients!(names, values,
        p::Paramorph.Prefixed{P}, logical::Tuple, prefix::String) where {P}
    return _append_space_coefficients!(
        names, values, p.space, logical, _parameter_prefix(prefix, P))
end

# Correlation matrices expose exactly their free off-diagonal natural entries.
function _append_space_coefficients!(names, values,
        p::Paramorph.Correlation, logical::Tuple, prefix::String)
    length(logical) == 1 || throw(DimensionMismatch(
        "a correlation space has one logical matrix parameter"))
    R = only(logical)
    name = _parameter_prefix(prefix, only(Paramorph.names(p)))
    n = size(R, 1)
    size(R, 2) == n || throw(DimensionMismatch("correlation parameter must be square"))
    @inbounds for j in 2:n, i in 1:j-1
        push!(names, _indexed_parameter_name(name, i, j, n))
        push!(values, R[i, j])
    end
    return nothing
end

# Positive-definite matrices have a free triangular parameterization including
# the diagonal. These remain natural matrix entries, not Cholesky/chart values.
function _append_space_coefficients!(names, values,
        p::Paramorph.SPD, logical::Tuple, prefix::String)
    length(logical) == 1 || throw(DimensionMismatch(
        "an SPD space has one logical matrix parameter"))
    Σ = only(logical)
    name = _parameter_prefix(prefix, only(Paramorph.names(p)))
    n = size(Σ, 1)
    size(Σ, 2) == n || throw(DimensionMismatch("SPD parameter must be square"))
    @inbounds for j in 1:n, i in 1:j
        push!(names, _indexed_parameter_name(name, i, j, n))
        push!(values, Σ[i, j])
    end
    return nothing
end

# A simplex has one redundant natural entry. StatsBase coefficients use a fixed
# natural convention (omit the first entry), independent of Paramorph's chart
# anchor. The anchor is an optimization detail and may legitimately differ
# between instances; exposing it here would make coefficient identities change
# across bootstrap refits.
function _append_space_coefficients!(names, values,
        p::Paramorph.Simplex, logical::Tuple, prefix::String)
    length(logical) == 1 || throw(DimensionMismatch(
        "a simplex space has one logical vector parameter"))
    x = only(logical)
    name = _parameter_prefix(prefix, only(Paramorph.names(p)))
    n = length(x)
    n == p.n || throw(DimensionMismatch(
        "simplex parameter has length $n; expected $(p.n)"))
    for i in eachindex(x)
        i == firstindex(x) && continue
        push!(names, _indexed_parameter_name(name, Int(i), n))
        push!(values, x[i])
    end
    return nothing
end

# All remaining Paramorph primitives expose one scalar per logical scalar, or
# every entry of an elementwise vector/matrix. Coupled primitives with several
# logical parameters (Ordered, Between, NIG, ...) arrive as several tuple values.
function _append_space_coefficients!(names, values,
        p::Paramorph.AbstractParameterSpace, logical::Tuple, prefix::String)
    pnames = Paramorph.names(p)
    length(pnames) == length(logical) || throw(DimensionMismatch(
        "parameter-space names and constrained logical values have different lengths"))
    for (name, value) in zip(pnames, logical)
        _append_logical_coefficient!(
            names, values, value, _parameter_prefix(prefix, name))
    end
    return nothing
end

function _space_coefficients(p, logical::Tuple)
    length(logical) == length(Paramorph.names(p)) || throw(DimensionMismatch(
        "parameter-space names and constrained logical values have different lengths"))
    names = String[]
    values = Any[]
    _append_space_coefficients!(names, values, p, logical, "")
    expected = Paramorph.dimension(p)
    length(values) == expected || throw(DimensionMismatch(
        "natural coefficient representation has $(length(values)) entries; " *
        "Paramorph space has dimension $expected"))
    scalars = isempty(values) ? Float64[] : collect(promote(float.(values)...))
    return names, scalars
end

function _distribution_coefficients(D)
    p = Paramorph.param_space(D)
    logical = _coefficient_parameter_values(D, p)
    return _space_coefficients(p, logical)
end

# These two families still have model structure that is not honestly described
# by a single Paramorph space: nested Archimedean models are recursive trees,
# while multivariate FGM has coupled hypercube-corner inequalities. Keep their
# coefficient presentation local instead of pretending that geometry is a
# Cartesian product.
_distribution_coefficients(C::NestedArchimedeanCopula) = _nested_coef(C)
function _distribution_coefficients(C::FGMCopula)
    names = String[]
    values = Any[]
    _append_logical_coefficient!(names, values, C.θ, "θ")
    scalars = isempty(values) ? Float64[] : collect(promote(float.(values)...))
    return names, scalars
end
_distribution_coefficients(S::AbstractReflectedCopula) =
    _distribution_coefficients(basecopula(S))

function _structured_coefficient_data(M::CopulaModel)
    spec = M.recipe
    if spec isa _CopulaFitSpec && spec.target isa NamedTuple &&
            haskey(spec.target, :coordinates)
        α = spec.target.coordinates
        return ["α$(i)" for i in eachindex(α)], collect(float.(α))
    end
    return _distribution_coefficients(fitted_distribution(M))
end

# More-specific methods replace the legacy shape-driven fallback in Fitting.jl
# for every fitted result handled by Copulas.jl, without exposing optimizer
# coordinates as statistical coefficients.
_coefficient_data(M::CopulaModel{<:Copula}) = _structured_coefficient_data(M)
_coefficient_data(M::CopulaModel{<:SklarDist}) = _structured_coefficient_data(M)

_parameter_blocks(M::CopulaModel{<:Copula}) =
    (; copula=eachindex(StatsBase.coef(M)), margins=())

function _parameter_blocks(M::CopulaModel{<:SklarDist})
    D = fitted_distribution(M)
    p = Paramorph.param_space(D)
    logical = _coefficient_parameter_values(D, p)

    blocks = Vector{Vector{Int}}(undef, length(p))
    value_offset = 0
    coefficient_offset = 0
    for (k, q) in pairs(p)
        nlogical = length(Paramorph.names(q))
        qvalues = ntuple(i -> logical[value_offset + i], nlogical)
        ncoefficients = length(last(_space_coefficients(q, qvalues)))
        blocks[k] = collect((coefficient_offset + 1):(coefficient_offset + ncoefficients))
        value_offset += nlogical
        coefficient_offset += ncoefficients
    end

    value_offset == length(logical) || throw(DimensionMismatch(
        "Sklar component spaces do not cover all logical parameter values"))
    coefficient_offset == length(StatsBase.coef(M)) || throw(DimensionMismatch(
        "Sklar component spaces do not cover all model coefficients"))

    return (; copula=first(blocks), margins=Tuple(blocks[2:end]))
end

# Analytical inference is performed in the Euclidean Paramorph chart, then
# pushed forward to the same natural scalar coefficients exposed by StatsBase.
# Specializing on Paramorph spaces keeps the old shape-driven flattening path out
# of every current copula inference route and, in particular, keeps simplex
# covariance dimensions equal to the model's free-parameter count.
function _vcov_finalize(CT::Type{<:Copula}, U::AbstractMatrix, θ::Tuple,
        d::Int, pspace::Paramorph.AbstractParameterSpace, α, Vα)
    J = ForwardDiff.jacobian(
        αv -> _space_coefficients(
            pspace,
            Distributions.params(_parameter_space_copula(CT, d, pspace, αv)),
        )[2],
        α,
    )
    return _validate_inference_covariance(J * Vα * J')
end

function _vcov_finalize(CT::Type{<:Copula}, U::AbstractMatrix, θ::Tuple,
        d::Int, pspace::Tuple, α, Vα)
    J = ForwardDiff.jacobian(
        αv -> _space_coefficients(
            pspace,
            Distributions.params(_parameter_space_copula(CT, d, pspace, αv)),
        )[2],
        α,
    )
    return _validate_inference_covariance(J * Vα * J')
end
