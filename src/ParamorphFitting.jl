# Internal bridge between Copulas' notion of parameter geometry and Paramorph.
#
# Model files declare constraints with `@paramorph`; fitting and inference use
# only the `_parameter_*` functions below.  Paramorph-specific runtime calls are
# intentionally centralized here.

function _concrete_paramorph_type(T::Type, ::Type{N}=Float64) where {N<:Real}
    concrete = Paramorph.rebind_numeric_type(T, N)
    if concrete isa UnionAll
        candidate = try
            Core.apply_type(concrete, N)
        catch err
            err isa TypeError || rethrow()
            concrete
        end
        Paramorph.is_paramorph_type(candidate) && return candidate
    end
    return concrete
end

_parameter_dimension(object) = Paramorph.intrinsic_dimension(object)
_parameter_coordinates(object) = Paramorph.unconstrain(object)
_from_parameter_coordinates(object, α) = Paramorph.constraint(object, α)

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

function _component_prototype(
    T::Type, context::NamedTuple=NamedTuple(); auxiliary::NamedTuple=NamedTuple(),
)
    concrete = _concrete_paramorph_type(T)
    Paramorph.is_paramorph_type(concrete) || throw(ArgumentError(
        "$T does not declare parameter geometry with @paramorph",
    ))
    n = Paramorph.intrinsic_dimension(concrete; context, auxiliary)
    return Paramorph.constraint(concrete, zeros(n); context, auxiliary)
end

function _parameter_prototype(CT::Type{<:Copula}, ::Val{d}) where {d}
    unwrapped = Base.unwrap_unionall(CT)
    encoded_dimension = unwrapped.parameters[1]
    dimensioned = encoded_dimension isa TypeVar ?
                  Core.apply_type(Base.typename(unwrapped).wrapper, d) : CT
    concrete = _concrete_paramorph_type(dimensioned)
    return _component_prototype(concrete, (; dimension=d))
end

# Structural wrappers are deliberately not Paramorph types. Their components
# own the geometry; this bridge supplies only composition/reconstruction.
function _parameter_prototype(CT::Type{<:ArchimedeanCopula}, ::Val{d}) where {d}
    G = _component_prototype(generatorof(CT), (; dimension=d))
    return ArchimedeanCopula{d}(G)
end
function _parameter_dimension(C::ArchimedeanCopula{d}) where {d}
    return Paramorph.intrinsic_dimension(C.G; context=(; dimension=d))
end
function _parameter_coordinates(C::ArchimedeanCopula{d}) where {d}
    return Paramorph.unconstrain(C.G; context=(; dimension=d))
end
function _from_parameter_coordinates(C::ArchimedeanCopula{d}, α) where {d}
    G = Paramorph.constraint(C.G, α; context=(; dimension=d))
    return ArchimedeanCopula{d}(G)
end

function _tail_prototype(TT::Type, ::Val{d}) where {d}
    concrete = _concrete_paramorph_type(TT)
    return _component_prototype(
        concrete, (; dimension=d); auxiliary=(; d=d),
    )
end

function _tail_prototype(::Type{<:HuslerReissTail}, ::Val{d}) where {d}
    geometry = _component_prototype(_HuslerReissScalarGeometry{Float64})
    return HuslerReissTail(geometry.θ)
end

function _tail_prototype(::Type{<:tEVTail}, ::Val{d}) where {d}
    geometry = _component_prototype(
        _tEVScalarGeometry{Float64}, (; dimension=d),
    )
    return tEVTail(geometry.ν, geometry.ρ)
end

function _parameter_prototype(CT::Type{<:ExtremeValueCopula}, vd::Val{d}) where {d}
    return ExtremeValueCopula{d}(_tail_prototype(tailof(CT), vd))
end
function _parameter_dimension(C::ExtremeValueCopula{d}) where {d}
    return _parameter_dimension(C.tail, Val(d))
end
function _parameter_coordinates(C::ExtremeValueCopula{d}) where {d}
    return _parameter_coordinates(C.tail, Val(d))
end
function _from_parameter_coordinates(C::ExtremeValueCopula{d}, α) where {d}
    return ExtremeValueCopula{d}(_from_parameter_coordinates(C.tail, α, Val(d)))
end

# Ordinary @paramorph tails need only a dimension context. Runtime auxiliary
# fields (for example a stored dimension) are already available from the object.
_parameter_dimension(tail::Tail, ::Val{d}) where {d} =
    Paramorph.intrinsic_dimension(tail; context=(; dimension=d))
_parameter_coordinates(tail::Tail, ::Val{d}) where {d} =
    Paramorph.unconstrain(tail; context=(; dimension=d))
_from_parameter_coordinates(tail::Tail, α, ::Val{d}) where {d} =
    Paramorph.constraint(tail, α; context=(; dimension=d))

function _parameter_prototype(CT::Type{<:ArchimaxCopula}, vd::Val{d}) where {d}
    GT, TT = genandtailof(CT)
    G = _component_prototype(GT, (; dimension=d))
    tail = _tail_prototype(TT, vd)
    return ArchimaxCopula{d}(G, tail)
end
function _parameter_dimension(C::ArchimaxCopula{d}) where {d}
    return Paramorph.intrinsic_dimension(C.gen; context=(; dimension=d)) +
           _parameter_dimension(C.tail, Val(d))
end
function _parameter_coordinates(C::ArchimaxCopula{d}) where {d}
    return vcat(
        Paramorph.unconstrain(C.gen; context=(; dimension=d)),
        _parameter_coordinates(C.tail, Val(d)),
    )
end
function _from_parameter_coordinates(C::ArchimaxCopula{d}, α) where {d}
    ng = Paramorph.intrinsic_dimension(C.gen; context=(; dimension=d))
    G = Paramorph.constraint(C.gen, view(α, 1:ng); context=(; dimension=d))
    tail = _from_parameter_coordinates(C.tail, view(α, (ng + 1):length(α)), Val(d))
    return ArchimaxCopula{d}(G, tail)
end

# Reflection wrappers preserve their reflection metadata while delegating the
# actual chart to the underlying copula.
_parameter_dimension(C::AbstractReflectedCopula) = _parameter_dimension(basecopula(C))
_parameter_coordinates(C::AbstractReflectedCopula) = _parameter_coordinates(basecopula(C))
_from_parameter_coordinates(C::SurvivalCopula{d}, α) where {d} =
    SurvivalCopula{d}(_from_parameter_coordinates(basecopula(C), α), flipmask(C))
_from_parameter_coordinates(C::Rotated90Copula, α) =
    Rotated90Copula(_from_parameter_coordinates(basecopula(C), α))
_from_parameter_coordinates(C::Rotated180Copula, α) =
    Rotated180Copula(_from_parameter_coordinates(basecopula(C), α))
_from_parameter_coordinates(C::Rotated270Copula, α) =
    Rotated270Copula(_from_parameter_coordinates(basecopula(C), α))

# Liouville has a structural product chart: generator parameters followed by
# positive Dirichlet parameters. The chart composition belongs here, not in the
# model definition.
function _liouville_schema(C::LiouvilleCopula{d}) where {d}
    return Paramorph.TransformVariables.as((
        G=Paramorph.recursive_schema(C.G, (; dimension=2)),
        α=Paramorph.TransformVariables.as(
            Vector, Paramorph.TransformVariables.asℝ₊, d,
        ),
    ))
end
function _parameter_dimension(C::LiouvilleCopula)
    return Paramorph.TransformVariables.dimension(_liouville_schema(C))
end
function _parameter_coordinates(C::LiouvilleCopula)
    values = (; G=Paramorph.parameter_values(C.G), α=collect(C.α))
    return Paramorph.TransformVariables.inverse(_liouville_schema(C), values)
end
function _from_parameter_coordinates(C::LiouvilleCopula{d}, α) where {d}
    values = Paramorph.TransformVariables.transform(_liouville_schema(C), α)
    G = Paramorph.constraint(
        C.G,
        Paramorph.TransformVariables.inverse(
            Paramorph.recursive_schema(C.G, (; dimension=2)), values.G,
        );
        context=(; dimension=2),
    )
    return LiouvilleCopula{d}(G, Tuple(values.α))
end

# Representation variants use private @paramorph geometry objects. The domain
# tails keep their constructor/storage invariants while every optimizer
# constraint remains declarative.
_hr_geometry(tail::HuslerReissTail{<:Real}) =
    _HuslerReissScalarGeometry(something(tail.θ))
function _hr_geometry(tail::HuslerReissTail{<:AbstractMatrix})
    Γ = something(tail.Γ)
    return _HuslerReissMatrixGeometry(size(Γ, 1), Matrix(Γ))
end
_hr_tail(g::_HuslerReissScalarGeometry) = HuslerReissTail(g.θ)
_hr_tail(g::_HuslerReissMatrixGeometry) = HuslerReissTail(g.Γ)

_parameter_dimension(tail::HuslerReissTail, ::Val) =
    Paramorph.intrinsic_dimension(_hr_geometry(tail))
_parameter_coordinates(tail::HuslerReissTail, ::Val) =
    Paramorph.unconstrain(_hr_geometry(tail))
_from_parameter_coordinates(tail::HuslerReissTail, α, ::Val) =
    _hr_tail(Paramorph.constraint(_hr_geometry(tail), α))

_tev_geometry(tail::tEVTail{<:Any,<:Real}, d) =
    _tEVScalarGeometry(tail.ν, something(tail.ρ))
function _tev_geometry(tail::tEVTail{<:Any,<:AbstractMatrix}, d)
    R = something(tail.R)
    return _tEVMatrixGeometry(size(R, 1), tail.ν, Matrix(R))
end
_tev_tail(g::_tEVScalarGeometry) = tEVTail(g.ν, g.ρ)
_tev_tail(g::_tEVMatrixGeometry) = tEVTail(g.ν, g.R)

_parameter_dimension(tail::tEVTail, ::Val{d}) where {d} =
    Paramorph.intrinsic_dimension(_tev_geometry(tail, d); context=(; dimension=d))
_parameter_coordinates(tail::tEVTail, ::Val{d}) where {d} =
    Paramorph.unconstrain(_tev_geometry(tail, d); context=(; dimension=d))
_from_parameter_coordinates(tail::tEVTail, α, ::Val{d}) where {d} =
    _tev_tail(Paramorph.constraint(_tev_geometry(tail, d), α; context=(; dimension=d)))
