###############################################################################
##### Natural model coefficients
###############################################################################

# Paramorph owns statistical dimension and optimization geometry. StatsBase
# coefficients deliberately expose the fitted distribution's natural `params`
# representation instead, flattened mechanically without trying to remove
# constraints or redundancies such as matrix symmetry or simplex sums.
function Paramorph.param_space(S::SklarDist)
    copula_space = Paramorph.Prefixed(:copula, Paramorph.param_space(S.C))
    margin_spaces = ntuple(length(S.m)) do i
        Paramorph.Prefixed(Symbol("margin_", i), Paramorph.param_space(S.m[i]))
    end
    return (copula_space, margin_spaces...)
end

Paramorph.param_space(::EmpiricalCopula) = ()
Paramorph.param_space(S::AbstractReflectedCopula) =
    Paramorph.param_space(basecopula(S))

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

function _distribution_coefficients(D; prefix::String="")
    names = String[]
    values = Any[]
    _append_distribution_parameters!(names, values, D, prefix)
    return names, _promoted_parameter_values(values)
end

function _distribution_coefficients(S::SklarDist; prefix::String="")
    names = String[]
    values = Any[]
    _append_distribution_parameters!(
        names, values, S.C, _parameter_prefix(prefix, :copula))
    for (i, margin) in pairs(S.m)
        _append_distribution_parameters!(
            names, values, margin, _parameter_prefix(prefix, Symbol("margin_", i)))
    end
    return names, _promoted_parameter_values(values)
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
    D isa NestedArchimedeanCopula && return length(last(_nested_coef(D)))
    p = try
        Paramorph.param_space(D)
    catch
        nothing
    end
    return p === nothing ? length(_distribution_coefficient_values(D)) :
                           Paramorph.dimension(p)
end

function _model_dof(M::CopulaModel)
    spec = M.recipe
    if spec isa _CopulaFitSpec && spec.target isa NamedTuple &&
            haskey(spec.target, :coordinates)
        return length(spec.target.coordinates)
    end
    return _distribution_dof(fitted_distribution(M))
end
