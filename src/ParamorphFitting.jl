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

function _tail_prototype(TT::Type{<:HuslerReissTail}, ::Val{d}) where {d}
    U = Base.unwrap_unionall(TT)
    encoded_d = U.parameters[1]
    encoded_d isa TypeVar || encoded_d == d || throw(DimensionMismatch(
        "Hüsler-Reiss tail dimension $encoded_d does not match d=$d",
    ))
    return _component_prototype(HuslerReissTail{d,Float64}, (; dimension=d))
end

function _tail_prototype(TT::Type{<:tEVTail}, ::Val{d}) where {d}
    U = Base.unwrap_unionall(TT)
    encoded_d = U.parameters[1]
    encoded_d isa TypeVar || encoded_d == d || throw(DimensionMismatch(
        "extremal-t tail dimension $encoded_d does not match d=$d",
    ))
    return _component_prototype(tEVTail{d,Float64}, (; dimension=d))
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

# Inference reconstructs a fitted family from its natural parameters. A target
# may already encode the copula dimension (`GumbelCopula{2}`), in which case
# passing `d` again selects the wrong constructor. Keep that distinction in the
# same bridge that owns dimension-aware prototype reconstruction.
function _analytical_parameter_coordinates(target::Type{<:Copula}, d, parameters)
    try
        unwrapped = Base.unwrap_unionall(target)
        encoded_dimension = unwrapped.parameters[1]
        fitted = encoded_dimension isa TypeVar ?
                 target(d, parameters...) : target(parameters...)
        α = _parameter_coordinates(fitted)
        return all(isfinite, α) ? (fitted, α) : nothing
    catch err
        err isa InterruptException && rethrow()
        return nothing
    end
end

# Rank inversions are pairwise, but some one-parameter Archimedean families have
# a narrower admissible domain when the fitted copula dimension is larger than
# two. Derive the target interval from the Paramorph chart itself instead of
# maintaining a second family-specific bounds table.
const _DimensionDependentRankGenerator = Union{
    AMHGenerator,
    ClaytonGenerator,
    FrankGenerator,
    GumbelBarnettGenerator,
}

function _scalar_parameter_endpoints(GT::Type{<:Generator}, d::Int)
    concrete = _concrete_paramorph_type(GT)
    context = (; dimension=d)
    Paramorph.intrinsic_dimension(concrete; context) == 1 || throw(ArgumentError(
        "$GT does not have a scalar parameter geometry in dimension $d",
    ))
    endpoint(z) = only(values(Paramorph.parameter_values(
        Paramorph.constraint(concrete, [z]; context),
    )))
    return endpoint(-Inf), endpoint(Inf)
end

function _project_scalar_parameter(GT::Type{<:Generator}, d::Int, θ)
    lower, upper = _scalar_parameter_endpoints(GT, d)
    return clamp(θ, lower, upper)
end

function _fit(
    CT::Type{<:ArchimedeanCopula{D,GT} where {D,GT<:_DimensionDependentRankGenerator}},
    U,
    vd::Val{d},
    m::Union{Val{:itau},Val{:irho}};
    weights=nothing,
) where {d}
    GT = _generator_family(generatorof(CT))
    invf = m isa Val{:itau} ? τ⁻¹ : ρ⁻¹
    measure = _rank_measure(m, U, weights)
    upper_triangle_flat = [
        measure[idx] for idx in CartesianIndices(measure) if idx[1] < idx[2]
    ]
    θs = map(v -> invf(GT, clamp(v, -1, 1)), upper_triangle_flat)
    θ = _project_scalar_parameter(GT, d, Statistics.mean(θs))
    return _dynamic_archimedean(CT, vd, θ)
end

# Unsupported nested generator families should fail with the documented public
# error instead of leaking a MethodError from the family-specific bounds table.
function _nested_scalar_bounds(G::Generator, ::Int, ::Bool)
    throw(ArgumentError(
        "template fitting currently provides nesting geometries only for standard " *
        "one-parameter generators; $(nameof(typeof(G))) is open for contributions",
    ))
end

# Independence has no scalar parent parameter. Its fitting rule is genuinely
# free, so delegate directly to the child's local geometry instead of trying to
# extract a nonexistent parent value in `_nested_edge_transform`.
function _nested_edge_transform(
    ::IndependentGenerator, child::Generator, dloc::Int; parent_role::Bool,
)
    return _nested_standard_transform(child, dloc; parent_role)
end
