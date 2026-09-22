###############################################################################
##### Natural model coefficients
###############################################################################

# Parameter geometry owns statistical dimension and optimizer coordinates.
# StatsBase coefficients deliberately expose the fitted distribution's natural `params`
# representation instead, flattened mechanically without trying to remove
# constraints or redundancies such as matrix symmetry or simplex sums.
@inline function _parameter_prefix(prefix::String, part)
    text = string(part)
    isempty(prefix) && return text
    isempty(text) && return prefix
    return string(prefix, "_", text)
end

@inline function _indexed_parameter_name(name::String, I::CartesianIndex, shape)
    idx = Tuple(I)
    small = all(<=(9), shape)
    suffix = small ? join((Char(0x2080 + i) for i in idx)) :
                     "_" * join(idx, "_")
    return isempty(name) ? "θ" * suffix : name * suffix
end

function _tuple_parameter_names(D, raw::Tuple)
    props = try
        propertynames(D)
    catch
        ()
    end
    if length(props) == length(raw)
        matches = true
        for i in eachindex(raw)
            matches &= try
                isequal(getproperty(D, props[i]), raw[i])
            catch
                false
            end
        end
        matches && return string.(props)
    end

    values = _declared_parameter_values(D)
    values isa NamedTuple && length(values) == length(raw) &&
        return string.(keys(values))

    names = _declared_parameter_names(typeof(D))
    names !== nothing && length(names) == length(raw) && return string.(names)

    return ["θ$(i)" for i in eachindex(raw)]
end

# These wrappers deliberately hide their storage fields from the natural
# statistical representation. Delegate labels to the component that owns the
# displayed parameters instead of exposing implementation names such as `G`
# or synthesizing positional labels.
_tuple_parameter_names(C::ArchimedeanCopula, raw::Tuple) =
    _tuple_parameter_names(C.G, raw)
_tuple_parameter_names(C::AbstractReflectedCopula, raw::Tuple) =
    _tuple_parameter_names(basecopula(C), raw)
function _tuple_parameter_names(
    C::ExtremeValueCopula{d,<:Union{TawnTail,AsymGalambosTail}}, raw::Tuple,
) where {d}
    length(raw) == d + 1 || return ["θ$(i)" for i in eachindex(raw)]
    return ["dep"; ["weights$(i)" for i in 1:d]]
end

function _append_natural_coefficient!(names, values, value, name::String)
    if value isa Number
        push!(names, isempty(name) ? "θ" : name)
        push!(values, value)
    elseif value isa AbstractArray
        for I in CartesianIndices(value)
            _append_natural_coefficient!(
                names, values, value[I], _indexed_parameter_name(name, I, size(value)))
        end
    elseif value isa NamedTuple
        for (key, child) in pairs(value)
            _append_natural_coefficient!(
                names, values, child, _parameter_prefix(name, key))
        end
    elseif value isa Tuple
        for (i, child) in pairs(value)
            _append_natural_coefficient!(
                names, values, child, _parameter_prefix(name, i))
        end
    elseif applicable(Distributions.params, value)
        _append_distribution_parameters!(names, values, value, name)
    else
        throw(ArgumentError(
            "cannot expose parameter $(name) with value type $(typeof(value))"))
    end
    return nothing
end

function _append_distribution_parameters!(names, values, D, prefix::String)
    raw = Distributions.params(D)
    if raw isa NamedTuple
        for (key, value) in pairs(raw)
            _append_natural_coefficient!(
                names, values, value, _parameter_prefix(prefix, key))
        end
    elseif raw isa Tuple
        labels = _tuple_parameter_names(D, raw)
        for (label, value) in zip(labels, raw)
            _append_natural_coefficient!(
                names, values, value, _parameter_prefix(prefix, label))
        end
    else
        _append_natural_coefficient!(names, values, raw, prefix)
    end
    return nothing
end

_promoted_parameter_values(values) =
    isempty(values) ? Float64[] : collect(promote(float.(values)...))

# A component with a known known zero-dimensional parameter geometry has no statistical
# coefficients. This keeps empirical or purely structural state out of `coef`.
function _has_natural_coefficients(D)
    dimension = _parameter_dimension_or_nothing(D)
    return dimension === nothing || !iszero(dimension)
end

function _distribution_coefficients(D; prefix::String="")
    _has_natural_coefficients(D) || return String[], Float64[]
    names = String[]
    values = Any[]
    _append_distribution_parameters!(names, values, D, prefix)
    return names, _promoted_parameter_values(values)
end

function _distribution_coefficients(S::SklarDist; prefix::String="")
    names = String[]
    values = Float64[]
    component_names, component_values =
        _distribution_coefficients(S.C; prefix=_parameter_prefix(prefix, :copula))
    append!(names, component_names)
    append!(values, component_values)
    for (i, margin) in pairs(S.m)
        component_names, component_values = _distribution_coefficients(
            margin; prefix=_parameter_prefix(prefix, Symbol("margin_", i)))
        append!(names, component_names)
        append!(values, component_values)
    end
    return names, values
end

# Nested Archimedean topology is structural rather than a flat distribution
# parameter. Preserve its fitted-generator natural coefficient representation.
_distribution_coefficients(C::NestedArchimedeanCopula; prefix::String="") =
    _nested_coef(C)

# Liebscher topology is also structural. Expose natural parameters only for
# component charts that participate in template fitting and for the active
# simplex weights. The displayed coefficient vector may therefore be longer
# than the intrinsic optimizer dimension because each simplex keeps all of its
# natural weights while contributing one fewer fitting degree of freedom.
function _distribution_coefficients(C::LiebscherCopula{d}; prefix::String="") where {d}
    names = String[]
    values = Any[]

    for (k, component) in pairs(C.copulas)
        _liebscher_component_space(component) === nothing && continue
        component_names, component_values = _distribution_coefficients(
            component; prefix=_parameter_prefix(prefix, Symbol("C", k)))
        append!(names, component_names)
        append!(values, component_values)
    end

    for j in 1:d
        active, p = _liebscher_weight_geometry(C.weights, j)
        p === nothing && continue
        for k in active
            push!(names, _parameter_prefix(prefix, "a$(k)_$(j)"))
            push!(values, C.weights[k, j])
        end
    end

    return names, _promoted_parameter_values(values)
end

_distribution_coefficient_values(D) = last(_distribution_coefficients(D))

_coefficient_data(M::CopulaModel{<:Copula}) =
    _distribution_coefficients(fitted_distribution(M))
_coefficient_data(M::CopulaModel{<:SklarDist}) =
    _distribution_coefficients(fitted_distribution(M))

_parameter_blocks(M::CopulaModel{<:Copula}) =
    (; copula=eachindex(StatsBase.coef(M)), margins=())

function _parameter_blocks(M::CopulaModel{<:SklarDist})
    D = fitted_distribution(M)
    lengths = (length(_distribution_coefficient_values(D.C)),
               map(length ∘ _distribution_coefficient_values, D.m)...)
    blocks = Vector{UnitRange{Int}}(undef, length(lengths))
    offset = 0
    for i in eachindex(lengths)
        blocks[i] = (offset + 1):(offset + lengths[i])
        offset += lengths[i]
    end
    offset == length(StatsBase.coef(M)) || throw(DimensionMismatch(
        "Sklar component parameters do not cover all model coefficients"))
    return (; copula=first(blocks), margins=Tuple(blocks[2:end]))
end

function _distribution_dof(D)
    dimension = _parameter_dimension_or_nothing(D)
    return dimension === nothing ?
           length(_distribution_coefficient_values(D)) : dimension
end

# Distributions.jl/StatsBase already know which natural parameters of a
# univariate margin are structural. Respect that contract when Paramorph does
# not declare a geometry (for example Binomial `n` or a simplex probability).
function _distribution_dof(D::Distributions.UnivariateDistribution)
    dimension = _parameter_dimension_or_nothing(D)
    dimension === nothing || return dimension
    return hasmethod(StatsBase.dof, Tuple{typeof(D)}) ?
           StatsBase.dof(D) : length(_distribution_coefficient_values(D))
end

# Liebscher exposes all active simplex weights as natural coefficients, while
# each simplex contributes one fewer optimizer coordinate.
_distribution_dof(C::LiebscherCopula) = length(_liebscher_initial_coordinates(C))

_distribution_dof(S::SklarDist) =
    _distribution_dof(S.C) + sum(_distribution_dof, S.m; init=0)

# Declared parameter geometry is the source of statistical dimension for ordinary
# parametric copulas and margins. The natural coefficient vector may be longer
# because it deliberately retains constraints and redundant entries.
StatsBase.dof(C::Copula) = _distribution_dof(C)
StatsBase.dof(S::SklarDist) = _distribution_dof(S)

function _model_dof(M::CopulaModel)
    spec = M.recipe
    if spec isa _CopulaFitSpec && spec.target isa NamedTuple &&
            haskey(spec.target, :coordinates)
        return length(spec.target.coordinates)
    end
    return _distribution_dof(fitted_distribution(M))
end
