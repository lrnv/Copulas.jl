from pathlib import Path

root = Path("src")

# Central bridge: expose semantic metadata helpers so the rest of Copulas never
# needs to know Paramorph's protocol names.
p = root / "ParamorphFitting.jl"
text = p.read_text()
needle = '''_parameter_dimension(object) = Paramorph.intrinsic_dimension(object)
_parameter_coordinates(object) = Paramorph.unconstrain(object)
_from_parameter_coordinates(object, α) = Paramorph.constraint(object, α)
'''
replacement = needle + '''
_declares_parameter_geometry(::Type{T}) where {T} = Paramorph.is_paramorph_type(T)
_declared_parameter_values(object) =
    _declares_parameter_geometry(typeof(object)) ? Paramorph.parameter_values(object) : nothing
_declared_parameter_names(::Type{T}) where {T} =
    _declares_parameter_geometry(T) ? Paramorph.parameter_fields(T) : nothing

function _parameter_dimension_or_nothing(object)
    try
        return _parameter_dimension(object)
    catch err
        (err isa ArgumentError || err isa MethodError) || rethrow()
        return nothing
    end
end
'''
assert needle in text
p.write_text(text.replace(needle, replacement, 1))

# Natural parameters consume the bridge, not Paramorph directly.
p = root / "NaturalParameters.jl"
text = p.read_text()
text = text.replace(
'''# Paramorph names are the canonical logical names for ordinary parametric
# components. Keep array-valued natural parameters independent of internal
# storage so `params` behaves like the generic Copula implementation.
''',
'''# Declared geometry supplies canonical logical names for ordinary parametric
# components. Keep array-valued natural parameters independent of internal
# storage so `params` behaves like the generic Copula implementation.
''')
text = text.replace(
'''function _named_parameter_values(component)
    return map(values(Paramorph.parameter_values(component))) do value
        _copy_parameter_value(value)
    end
end
''',
'''function _named_parameter_values(component)
    declared = _declared_parameter_values(component)
    declared === nothing && throw(ArgumentError(
        "$(typeof(component)) does not declare parameter geometry",
    ))
    return map(values(declared)) do value
        _copy_parameter_value(value)
    end
end
''')
text = text.replace('In-package univariate generator families use Paramorph as their canonical',
                    'In-package univariate generator families use their declared geometry as the canonical')
text = text.replace('Dirichlet vector described by its Paramorph schema.',
                    'Dirichlet vector described by its parameter geometry.')
text = text.replace('Paramorph.is_paramorph_type(TG) || return (C.G, C.α)',
                    '_declares_parameter_geometry(TG) || return (C.G, C.α)')
text = text.replace('logical Paramorph names', 'logical geometry names')
text = text.replace('zero-dimensional Paramorph schema', 'zero-dimensional fitted geometry')
assert 'Paramorph.' not in text
p.write_text(text)

# Statistical coefficient metadata is a consumer of the bridge as well.
p = root / "ParameterCoefficients.jl"
text = p.read_text()
text = text.replace(
'''# Paramorph owns statistical dimension and optimization geometry. StatsBase
# coefficients deliberately expose the fitted distribution's natural `params`
''',
'''# Parameter geometry owns statistical dimension and optimizer coordinates.
# StatsBase coefficients deliberately expose the fitted distribution's natural `params`
''')
old = '''    values = try
        Paramorph.parameter_values(D)
    catch
        nothing
    end
    values isa NamedTuple && length(values) == length(raw) &&
        return string.(keys(values))

    if Paramorph.is_paramorph_type(typeof(D))
        names = Paramorph.parameter_fields(typeof(D))
        length(names) == length(raw) && return string.(names)
    end
'''
new = '''    values = _declared_parameter_values(D)
    values isa NamedTuple && length(values) == length(raw) &&
        return string.(keys(values))

    names = _declared_parameter_names(typeof(D))
    names !== nothing && length(names) == length(raw) && return string.(names)
'''
assert old in text
text = text.replace(old, new, 1)
text = text.replace('zero-dimensional Paramorph schema', 'known zero-dimensional parameter geometry')
old = '''function _has_natural_coefficients(D)
    return !Paramorph.is_paramorph_type(typeof(D)) ||
           !iszero(Paramorph.intrinsic_dimension(D))
end
'''
new = '''function _has_natural_coefficients(D)
    dimension = _parameter_dimension_or_nothing(D)
    return dimension === nothing || !iszero(dimension)
end
'''
assert old in text
text = text.replace(old, new, 1)
text = text.replace('intrinsic Paramorph dimension', 'intrinsic optimizer dimension')
old = '''function _distribution_dof(D)
    return _has_specific_paramorph_schema(D) ?
           Paramorph.intrinsic_dimension(D) :
           length(_distribution_coefficient_values(D))
end


# Paramorph's Distributions extension supplies prototype-dependent schemas for
# distributions with structural constructor arguments (for example Binomial's
# fixed `n`) without claiming ownership of those external types through
# `is_paramorph_type`. Detect such a specialization separately from
# Paramorph's universal scalar fallback.
function _has_specific_paramorph_schema(D)
    Paramorph.is_paramorph_type(typeof(D)) && return true
    method = which(Paramorph.transformation_schema, (typeof(D), NamedTuple))
    return method.module !== Paramorph
end
'''
new = '''function _distribution_dof(D)
    dimension = _parameter_dimension_or_nothing(D)
    return dimension === nothing ?
           length(_distribution_coefficient_values(D)) : dimension
end
'''
assert old in text
text = text.replace(old, new, 1)
text = text.replace('Paramorph is the single source of statistical dimension for ordinary',
                    'Declared parameter geometry is the source of statistical dimension for ordinary')
assert 'Paramorph.' not in text
p.write_text(text)

print("parameter metadata now consumes only the Copulas geometry bridge")
