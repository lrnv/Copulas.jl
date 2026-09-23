###############################################################################
##### Natural model coefficient reporting
###############################################################################

# `params` is the complete natural constructor representation. StatsBase
# coefficients flatten the natural statistical values used for reporting;
# unlike optimizer coordinates, they need not form an irreducible chart.
# Consequently `length(coef(model))` may exceed `dof(model)` when a natural
# representation contains constrained or redundant values such as a simplex.

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
    declared = _declared_parameter_values(D)
    if declared !== nothing && length(declared) == length(raw)
        return string.(keys(declared))
    end

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
    return ["θ$(i)" for i in eachindex(raw)]
end

_tuple_parameter_names(::Distributions.Categorical, raw::Tuple) =
    length(raw) == 1 ? ["p"] : ["θ$(i)" for i in eachindex(raw)]
_tuple_parameter_names(C::ArchimedeanCopula, raw::Tuple) =
    _tuple_parameter_names(C.G, raw)
_tuple_parameter_names(C::ExtremeValueCopula, raw::Tuple) =
    _tuple_parameter_names(C.tail, raw)
_tuple_parameter_names(C::AbstractReflectedCopula, raw::Tuple) =
    _tuple_parameter_names(basecopula(C), raw)

function _append_natural_coefficient!(names, values, value, name::String)
    if value isa Number
        push!(names, isempty(name) ? "θ" : name)
        push!(values, value)
    elseif value isa AbstractArray
        if length(value) == 1
            _append_natural_coefficient!(names, values, only(value), name)
        else
            for I in CartesianIndices(value)
                _append_natural_coefficient!(
                    names, values, value[I], _indexed_parameter_name(name, I, size(value)))
            end
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
    dimension = _parameter_dimension_or_nothing(D)
    iszero(something(dimension, 1)) && return String[], Float64[]
    names = String[]
    values = Any[]
    _append_distribution_parameters!(names, values, D, prefix)
    return names, _promoted_parameter_values(values)
end

# Keep the conventional concise representation for Gaussian correlations. This
# is a direct natural parameterization, not a generic requirement imposed on all
# structured constrained parameters.
function _distribution_coefficients(C::GaussianCopula; prefix::String="")
    names = String[]
    values = Any[]
    full_name = _parameter_prefix(prefix, "Σ")
    for j in 2:size(C.Σ, 2), i in 1:(j - 1)
        I = CartesianIndex(i, j)
        push!(names, _indexed_parameter_name(full_name, I, size(C.Σ)))
        push!(values, C.Σ[I])
    end
    return names, _promoted_parameter_values(values)
end

_distribution_coefficients(C::AbstractReflectedCopula; prefix::String="") =
    _distribution_coefficients(basecopula(C); prefix)

function _distribution_coefficients(C::ArchimaxCopula{d}; prefix::String="") where {d}
    names = String[]
    values = Float64[]
    generator_names, generator_values = _distribution_coefficients(
        ArchimedeanCopula{d}(C.gen);
        prefix=_parameter_prefix(prefix, :generator),
    )
    tail_names, tail_values = _distribution_coefficients(
        ExtremeValueCopula{d}(C.tail);
        prefix=_parameter_prefix(prefix, :tail),
    )
    append!(names, generator_names)
    append!(values, generator_values)
    append!(names, tail_names)
    append!(values, tail_values)
    return names, values
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

# Discrete constructors contain structural arguments that are not estimated
# coefficients.
_distribution_coefficients(D::Distributions.Binomial; prefix::String="") =
    ([_parameter_prefix(prefix, :p)], [float(D.p)])
function _distribution_coefficients(D::Distributions.BetaBinomial; prefix::String="")
    return [
        _parameter_prefix(prefix, :α),
        _parameter_prefix(prefix, :β),
    ], [float(D.α), float(D.β)]
end

# Nested topology is structural. Template fits expose local generator parameters;
# custom runtime maps may intentionally have a smaller fitting dimension, which
# is recorded in their fit recipe.
_distribution_coefficients(C::NestedArchimedeanCopula; prefix::String="") =
    _nested_coef(C)

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

_distribution_dof(::Distributions.Binomial) = 1
_distribution_dof(::Distributions.BetaBinomial) = 2
_distribution_dof(D::Distributions.Categorical) =
    max(length(Distributions.probs(D)) - 1, 0)
_distribution_dof(C::LiebscherCopula) = length(_liebscher_initial_coordinates(C))
_distribution_dof(S::SklarDist) =
    _distribution_dof(S.C) + sum(_distribution_dof, S.m; init=0)

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