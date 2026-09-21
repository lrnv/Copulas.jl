# Family-specific starting objects for the generic Paramorph fitting driver.
# This file is included only after every component type has been defined.

Paramorph.schema_context(
    ::Type{<:ArchimedeanCopula{d,<:ClaytonGenerator}}, ::Val{:G},
) where {d} = (; lower=-inv(d - 1), dimension=d)
Paramorph.schema_context(
    ::Type{<:ArchimedeanCopula{d,<:AMHGenerator}}, ::Val{:G},
) where {d} = (; lower=clamp(_find_critical_value_amh(d), -1, 1), dimension=d)
Paramorph.schema_context(
    ::Type{<:ArchimedeanCopula{d,<:GumbelBarnettGenerator}}, ::Val{:G},
) where {d} = (; upper=clamp(_find_critical_value_gumbelbarnett(d), 0, 1), dimension=d)

function _fit_prototype(CT::Type{<:ArchimedeanCopula}, ::Val{d}) where {d}
    GT = _concrete_paramorph_type(generatorof(CT))
    context = Paramorph.schema_context(ArchimedeanCopula{d,GT}, Val(:G))
    return ArchimedeanCopula{d}(_component_prototype(GT, context))
end

function _fit_prototype(CT::Type{<:ExtremeValueCopula}, ::Val{d}) where {d}
    TT = _concrete_paramorph_type(tailof(CT))
    context = Paramorph.schema_context(ExtremeValueCopula{d,TT}, Val(:tail))
    return ExtremeValueCopula{d}(_component_prototype(TT, context))
end

function _fit_prototype(CT::Type{<:ArchimaxCopula}, vd::Val{d}) where {d}
    GT, TT = genandtailof(CT)
    AT = Core.apply_type(ArchimedeanCopula, d, GT)
    ET = Core.apply_type(ExtremeValueCopula, d, TT)
    return ArchimaxCopula{d}(
        _fit_prototype(AT, vd).G,
        _fit_prototype(ET, vd).tail,
    )
end

function _fit_prototype(
    ::Type{<:TawnCopula}, ::Val{d},
) where {d}
    schema = _tawn_schema(d, Float64)
    values = Paramorph.TransformVariables.transform(
        schema, zeros(Paramorph.TransformVariables.dimension(schema)),
    )
    return ExtremeValueCopula{d}(TawnTail(values.dep, values.weights...))
end

function _fit_prototype(
    ::Type{<:AsymGalambosCopula}, ::Val{d},
) where {d}
    schema = _asymgalambos_schema(d)
    values = Paramorph.TransformVariables.transform(
        schema, zeros(Paramorph.TransformVariables.dimension(schema)),
    )
    return ExtremeValueCopula{d}(AsymGalambosTail(values.dep, values.weights...))
end

function _fit_prototype(
    ::Type{<:MOCopula}, ::Val{d},
) where {d}
    schema = _mo_schema(d)
    values = Paramorph.TransformVariables.transform(
        schema, zeros(Paramorph.TransformVariables.dimension(schema)),
    )
    return ExtremeValueCopula{d}(MOTail(d, values.λ))
end

function _fit_prototype(
    ::Type{<:BC2Copula}, ::Val{d},
) where {d}
    d == 2 || throw(DimensionMismatch("BC2Copula is only defined in dimension two"))
    return ExtremeValueCopula{2}(BC2Tail(fill(0.5, 2)))
end

_fit_prototype(::Type{<:PlackettCopula}, ::Val{d}) where {d} =
    Paramorph.constraint(PlackettCopula{d,Float64}, zeros(1))
_fit_prototype(::Type{<:RafteryCopula}, ::Val{d}) where {d} =
    Paramorph.constraint(RafteryCopula{d,Float64}, zeros(1))
_fit_prototype(::Type{<:GaussianCopula}, ::Val{d}) where {d} =
    Paramorph.constraint(GaussianCopula{d,Float64}, zeros(d * (d - 1) ÷ 2))
_fit_prototype(::Type{<:TCopula}, ::Val{d}) where {d} =
    Paramorph.constraint(TCopula{d,Float64}, zeros(1 + d * (d - 1) ÷ 2))
_fit_prototype(::Type{<:FGMCopula}, ::Val{d}) where {d} =
    Paramorph.constraint(
        FGMCopula{d,Float64},
        zeros(Paramorph.intrinsic_dimension(FGMCopula{d,Float64})),
    )
