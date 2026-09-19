###############################################################################
##### Natural model coefficients
###############################################################################

# `Paramorph.param_space` is fitting/statistical geometry, not a prerequisite
# for the public distribution interface. Named Archimedean families use it to
# identify their canonical mathematical parameters, while a downstream
# generator that implements only the documented `Generator` contract falls
# back to its stored constructor fields. This keeps `params` usable without
# silently turning Paramorph into an additional generator extension hook.
function Distributions.params(C::ArchimedeanCopula{d,G}) where {d,G<:Generator}
    if applicable(Paramorph.param_space, G, d)
        p = Paramorph.param_space(G, d)
        return map(Paramorph.names(p)) do name
            value = getproperty(C.G, name)
            return value isa AbstractArray ? copy(value) : value
        end
    end

    return ntuple(fieldcount(G)) do i
        value = getfield(C.G, i)
        return value isa AbstractArray ? copy(value) : value
    end
end

# Paramorph owns statistical dimension and optimization geometry. StatsBase
# coefficients deliberately expose the fitted distribution's natural `params`
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

    if applicable(Paramorph.param_space, D)
        names = try
            Paramorph.names(Paramorph.param_space(D))
        catch
            ()
        end
        length(names) == length(raw) && return string.(names)
    end

    return ["θ$(i)" for i in eachindex(raw)]
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

# A component with a known zero-dimensional Paramorph space has no statistical
# coefficients. This keeps empirical or purely structural state out of `coef`.
function _has_natural_coefficients(D)
    p = try
        Paramorph.param_space(D)
    catch
        nothing
    end
    return p === nothing || !iszero(Paramorph.dimension(p))
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
# parameter. Preserve its existing fitted-generator coefficient representation.
_distribution_coefficients(C::NestedArchimedeanCopula; prefix::String="") =
    _nested_coef(C)

_distribution_coefficient_values(D) = last(_distribution_coefficients(D))

function _structured_coefficient_data(M::CopulaModel)
    spec = M.recipe
    if spec isa _CopulaFitSpec && spec.target isa NamedTuple &&
            haskey(spec.target, :coordinates)
        α = spec.target.coordinates
        return ["α$(i)" for i in eachindex(α)], collect(float.(α))
    end
    return _distribution_coefficients(fitted_distribution(M))
end

_coefficient_data(M::CopulaModel{<:Copula}) = _structured_coefficient_data(M)
_coefficient_data(M::CopulaModel{<:SklarDist}) = _structured_coefficient_data(M)

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
    p = try
        Paramorph.param_space(D)
    catch
        nothing
    end
    return p === nothing ? length(_distribution_coefficient_values(D)) :
                           Paramorph.dimension(p)
end

_distribution_dof(S::SklarDist) =
    _distribution_dof(S.C) + sum(_distribution_dof, S.m; init=0)

# Paramorph is the single source of statistical dimension for ordinary
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
