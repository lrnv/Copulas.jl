# Small adapter between Copulas fitting/inference and Paramorph's public API.

# Paramorph 0.1.2's public `parameter_prototype` accepts concrete parameter
# types, but Julia family aliases commonly leave their final numeric parameter
# free (for example `ClaytonGenerator{T} where T`).  Dimension-dependent
# parameters are fixed by the callers below, so any remaining UnionAll variable
# is the numeric storage type selected for fitting.
_fitting_parameter_type(T::UnionAll, ::Type{N}=Float64) where {N<:Real} =
    Core.apply_type(T, N)
_fitting_parameter_type(T::Type, ::Type{<:Real}=Float64) = T

_parameter_dimension(object) = Paramorph.intrinsic_dimension(object)
_parameter_coordinates(object) = Paramorph.unconstrain(object)
_from_parameter_coordinates(object, α) = Paramorph.constraint(object, α)

_declares_parameter_geometry(::Type{T}) where {T} = Paramorph.has_parameter_geometry(T)
_declared_parameter_values(object) =
    _declares_parameter_geometry(typeof(object)) ? Paramorph.parameter_values(object) : nothing

# Capability detection is explicit. Once an object declares a geometry, errors
# while constructing or evaluating that geometry are implementation errors and
# must propagate instead of being reclassified as "no geometry".
_parameter_dimension_or_nothing(object) =
    _declares_parameter_geometry(typeof(object)) ? _parameter_dimension(object) : nothing
function _parameter_dimension_or_nothing(C::LiouvilleCopula)
    return _declares_parameter_geometry(typeof(C.G)) ? _parameter_dimension(C) : nothing
end

function _parameter_prototype(CT::Type{<:Copula}, ::Val{d}) where {d}
    unwrapped = Base.unwrap_unionall(CT)
    encoded_dimension = unwrapped.parameters[1]
    dimensioned = encoded_dimension isa TypeVar ?
                  Core.apply_type(Base.typename(unwrapped).wrapper, d) : CT
    return Paramorph.parameter_prototype(
        _fitting_parameter_type(dimensioned); context=(; dimension=d),
    )
end

function _generator_parameter_prototype(GT::Type{<:Generator}, d::Int)
    return Paramorph.parameter_prototype(
        _fitting_parameter_type(GT); context=(; dimension=d),
    )
end

# Family aliases such as `ClaytonCopula` encode their generator restriction in
# the second type parameter. Dimensioning the outer wrapper generically would
# erase that restriction, so construct the nested child prototype explicitly
# through Paramorph's public API. Object-level fitting then uses the wrapper's
# declared nested geometry directly.
function _parameter_prototype(CT::Type{<:ArchimedeanCopula}, ::Val{d}) where {d}
    G = _generator_parameter_prototype(generatorof(CT), d)
    return ArchimedeanCopula{d}(G)
end

# Some tail families store their dimension as an auxiliary runtime field. The
# outer copula knows that value structurally, so provide it through the public
# prototype API when reconstructing a family from its type.
function _tail_prototype(TT::Type, ::Val{d}) where {d}
    return Paramorph.parameter_prototype(
        _fitting_parameter_type(TT);
        context=(; dimension=d),
        auxiliary=(; d=d),
    )
end

function _tail_prototype(TT::Type{<:HuslerReissTail}, ::Val{d}) where {d}
    U = Base.unwrap_unionall(TT)
    encoded_d = U.parameters[1]
    encoded_d isa TypeVar || encoded_d == d || throw(DimensionMismatch(
        "Hüsler-Reiss tail dimension $encoded_d does not match d=$d",
    ))
    return Paramorph.parameter_prototype(
        HuslerReissTail{d,Float64}; context=(; dimension=d),
    )
end

function _tail_prototype(TT::Type{<:tEVTail}, ::Val{d}) where {d}
    U = Base.unwrap_unionall(TT)
    encoded_d = U.parameters[1]
    encoded_d isa TypeVar || encoded_d == d || throw(DimensionMismatch(
        "extremal-t tail dimension $encoded_d does not match d=$d",
    ))
    return Paramorph.parameter_prototype(
        tEVTail{d,Float64}; context=(; dimension=d),
    )
end

function _parameter_prototype(CT::Type{<:ExtremeValueCopula}, vd::Val{d}) where {d}
    return ExtremeValueCopula{d}(_tail_prototype(tailof(CT), vd))
end

function _parameter_prototype(CT::Type{<:ArchimaxCopula}, vd::Val{d}) where {d}
    GT, TT = genandtailof(CT)
    G = _generator_parameter_prototype(GT, d)
    tail = _tail_prototype(TT, vd)
    return ArchimaxCopula{d}(G, tail)
end

# Liouville has a genuinely coupled domain constraint between the generator and
# the Dirichlet weights, so it remains a Copulas-specific chart. Keep its
# implementation on Paramorph's public object operations.
function _parameter_dimension(C::LiouvilleCopula{d}) where {d}
    return Paramorph.intrinsic_dimension(C.G; context=(; dimension=2)) + d
end
function _parameter_coordinates(C::LiouvilleCopula)
    generator = Paramorph.unconstrain(C.G; context=(; dimension=2))
    return vcat(generator, log.(collect(C.α)))
end
function _from_parameter_coordinates(C::LiouvilleCopula{d}, α) where {d}
    ng = Paramorph.intrinsic_dimension(C.G; context=(; dimension=2))
    G = Paramorph.constraint(C.G, view(α, 1:ng); context=(; dimension=2))
    weights = ntuple(i -> exp(α[ng + i]), d)
    return LiouvilleCopula{d}(G, weights)
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
    context = (; dimension=d)
    prototype = _generator_parameter_prototype(GT, d)
    Paramorph.intrinsic_dimension(prototype; context) == 1 || throw(ArgumentError(
        "$GT does not have a scalar parameter geometry in dimension $d",
    ))
    endpoint(z) = only(values(Paramorph.parameter_values(
        Paramorph.constraint(prototype, [z]; context),
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
    return _nested_interval_transform(_nested_scalar_bounds(child, dloc, parent_role)...)
end
