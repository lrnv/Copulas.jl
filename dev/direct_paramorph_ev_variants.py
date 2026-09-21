from pathlib import Path

# -----------------------------------------------------------------------------
# Huesler-Reiss: one Paramorph type, representation encoded in the type.
# -----------------------------------------------------------------------------
p = Path('src/Tail/HuslerReissTail.jl')
text = p.read_text()
start = text.index('struct HuslerReissTail{P}')
end = text.index('function A(tail::HuslerReissTail', start)
new = r'''_hr_parameter_geometry(::Val{:exchangeable}, ::Int, ::Type{T}) where {T<:Real} =
    Paramorph.nonnegative()
_hr_parameter_geometry(::Val{:general}, d::Int, ::Type{T}) where {T<:Real} =
    Paramorph.variogram_matrix(d)

Paramorph.@paramorph T struct HuslerReissTail{d,R,T<:Real} <: OneParameterPickandsTail
    parameter::Union{T,Matrix{T}} ~ _hr_parameter_geometry(Val(R), d, T)
end

function _hr_exchangeable_variogram(d::Int, θ::Real)
    d >= 2 || throw(ArgumentError("Hüsler-Reiss dimension must be at least two"))
    γ = abs2(2 / θ)
    Γ = fill(float(γ), d, d)
    @inbounds for i in 1:d
        Γ[i, i] = zero(eltype(Γ))
    end
    return Γ
end

function _hr_general_tail(::Val{d}, Γ::AbstractMatrix) where {d}
    size(Γ) == (d, d) || throw(DimensionMismatch(
        "variogram dimension $(size(Γ)) does not match d=$d",
    ))
    T = float(eltype(Γ))
    G = Matrix{T}(Γ)
    all(isfinite, G) || throw(ArgumentError("Γ must contain only finite entries"))

    # The zero variogram is the complete-dependence boundary.  Keep exact
    # boundary points in the exchangeable chart; the general matrix chart is
    # deliberately the strict variogram interior.
    all(iszero, G) && return HuslerReissTail{d,:exchangeable,T}(T(Inf))

    try
        return HuslerReissTail{d,:general,T}(G)
    catch err
        (err isa DomainError || err isa LinearAlgebra.PosDefException) || rethrow()
        throw(ArgumentError("Γ must be a strict Hüsler-Reiss variogram"))
    end
end

function (::Type{HuslerReissTail{d}})(θ::Real) where {d}
    d >= 2 || throw(ArgumentError("Hüsler-Reiss dimension must be at least two"))
    θ < 0 && throw(ArgumentError("θ must be ≥ 0"))
    θf = float(θ)

    # In d=2 the scalar and general representations are the same one-dimensional
    # model. Canonicalize interior scalar input to the general matrix geometry;
    # retain the scalar chart only for exact limit points, where the strict
    # variogram chart has no finite matrix representative.
    if d == 2 && isfinite(θf) && !iszero(θf)
        return _hr_general_tail(Val(d), _hr_exchangeable_variogram(d, θf))
    end
    return HuslerReissTail{d,:exchangeable,typeof(θf)}(θf)
end
HuslerReissTail(θ::Real) = HuslerReissTail{2}(θ)

(::Type{HuslerReissTail{d}})(Γ::AbstractMatrix) where {d} =
    _hr_general_tail(Val(d), Γ)
HuslerReissTail(Γ::AbstractMatrix) =
    HuslerReissTail{size(Γ, 1)}(Γ)

@inline _hr_representation(::HuslerReissTail{d,R}) where {d,R} = R
@inline _hr_is_independent(tail::HuslerReissTail) =
    _hr_representation(tail) === :exchangeable && iszero(tail.parameter)
@inline function limit_kind(tail::HuslerReissTail, ::Val)
    _hr_representation(tail) === :general && return NO_LIMIT
    iszero(tail.parameter) && return Π_LIMIT
    isinf(tail.parameter) && return M_LIMIT
    return NO_LIMIT
end

const HuslerReissCopula{d,R,T} = ExtremeValueCopula{d,HuslerReissTail{d,R,T}}

function (::Type{HuslerReissCopula{d}})(θ::Real) where {d}
    return _wrap_extreme_value(Val(d), HuslerReissTail{d}(θ))
end
(::Type{HuslerReissCopula})(d::Int, θ::Real) =
    _wrap_extreme_value(Val(d), HuslerReissTail{d}(θ))

function (::Type{HuslerReissCopula{d}})(Γ::AbstractMatrix) where {d}
    return _wrap_extreme_value(Val(d), HuslerReissTail{d}(Γ))
end
HuslerReissCopula(Γ::AbstractMatrix) =
    _wrap_extreme_value(Val(size(Γ, 1)), HuslerReissTail(Γ))
function (::Type{HuslerReissCopula})(d::Int, Γ::AbstractMatrix)
    # Infer the tail's own dimension first so the runtime-dimension form keeps
    # ExtremeValueCopula's ArgumentError contract on a mismatch.
    return _wrap_extreme_value(Val(d), HuslerReissTail(Γ))
end

_is_valid_in_dim(::HuslerReissTail{D}, d::Int) where {D} = D == d

_hr_theta(tail::HuslerReissTail{D,:exchangeable}) where {D} = tail.parameter
_hr_theta(tail::HuslerReissTail{D,:general}) where {D} =
    2 / sqrt(tail.parameter[1, 2])
function _hr_variogram(tail::HuslerReissTail{D,:exchangeable}, d::Int) where {D}
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    return _hr_exchangeable_variogram(d, tail.parameter)
end
_hr_variogram(tail::HuslerReissTail{D,:general}, d::Int) where {D} = begin
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    tail.parameter
end

# Keep the public natural parameter convention stable: bivariate Hüsler-Reiss
# is exposed by its scalar θ even though the interior is stored in the general
# 2×2 variogram representation.
Distributions.params(C::ExtremeValueCopula{2,<:HuslerReissTail}) =
    (_hr_theta(C.tail),)
Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{D,:exchangeable}}) where {D} =
    (C.tail.parameter,)
Distributions.params(C::ExtremeValueCopula{D,<:HuslerReissTail{D,:general}}) where {D} =
    (copy(C.tail.parameter),)

_tail_constructor_parameter_names(::Type{<:HuslerReissTail{2}}, _) = (:θ,)
_tail_constructor_parameter_names(::Type{<:HuslerReissTail{D,:exchangeable}}, _) where {D} = (:θ,)
_tail_constructor_parameter_names(::Type{<:HuslerReissTail{D,:general}}, _) where {D} = (:Γ,)

# General d>2 variograms remain construction/evaluation objects rather than a
# default fitted family. In d=2 the general chart is one-dimensional and is the
# canonical representation of the ordinary Hüsler-Reiss family.
_available_fitting_methods(
    ::Type{<:ExtremeValueCopula{D,<:HuslerReissTail{D,:general}} where D}, d,
) = d == 2 ? (:mle, :itau, :irho, :ibeta, :iupper) : ()

'''
text = text[:start] + new + text[end:]

# Scalar/equicorrelated evaluation now dispatches on the representation tag.
text = text.replace(
    'function ℓ(tail::HuslerReissTail{<:Real}, x)\n    θ = something(tail.θ)',
    'function ℓ(tail::HuslerReissTail{D,:exchangeable}, x) where {D}\n    θ = tail.parameter',
    1,
)
text = text.replace(
    'ℓ(tail::HuslerReissTail{<:AbstractMatrix}, x) =\n    all(iszero, something(tail.Γ)) ? maximum(x) : _hr_stdf(something(tail.Γ), x)',
    'ℓ(tail::HuslerReissTail{D,:general}, x) where {D} =\n    _hr_stdf(tail.parameter, x)',
    1,
)
text = text.replace(
    'if tail isa HuslerReissTail{<:Real} && _hr_is_independent(tail)',
    'if _hr_is_independent(tail)',
    1,
)
text = text.replace(
    '''        reduced_tail = tail isa HuslerReissTail{<:Real} ? tail :
                       HuslerReissTail(Γ[active, active])''',
    '''        reduced_tail = _hr_representation(tail) === :exchangeable ?
                       HuslerReissTail{length(active)}(_hr_theta(tail)) :
                       HuslerReissTail(Γ[active, active])''',
    1,
)
p.write_text(text)

# -----------------------------------------------------------------------------
# Extremal-t: same representation-tagged pattern.
# -----------------------------------------------------------------------------
p = Path('src/Tail/tEVTail.jl')
text = p.read_text()
start = text.index('struct tEVTail{T,P}')
end = text.index('function _tev_stdf', start)
new = r'''_tev_parameter_geometry(::Val{:exchangeable}, d::Int, ::Type{T}) where {T<:Real} =
    Paramorph.bounded_interval(-inv(T(d - 1)), one(T); left_closed=false)
_tev_parameter_geometry(::Val{:general}, d::Int, ::Type{T}) where {T<:Real} =
    Paramorph.correlation_matrix(d)

Paramorph.@paramorph T struct tEVTail{d,R,T<:Real} <: BivariatePickandsTail
    ν::T ~ Paramorph.open_lower(zero(T))
    parameter::Union{T,Matrix{T}} ~ _tev_parameter_geometry(Val(R), d, T)
end

function _tev_general_tail(::Val{d}, ν::Real, R::AbstractMatrix) where {d}
    size(R) == (d, d) || throw(ArgumentError(
        "correlation matrix dimension $(size(R)) does not match d=$d",
    ))
    νf = float(ν)
    νf > 0 || throw(ArgumentError("ν must be > 0"))
    T = promote_type(typeof(νf), float(eltype(R)))
    RF = Matrix{T}(R)
    all(isfinite, RF) || throw(ArgumentError("R must contain only finite entries"))

    # The all-ones matrix is the complete-dependence boundary and is represented
    # exactly by the exchangeable scalar chart. The general correlation chart is
    # the strict positive-definite interior.
    if all(isone, RF)
        return tEVTail{d,:exchangeable,T}(T(νf), one(T))
    end

    try
        return tEVTail{d,:general,T}(T(νf), RF)
    catch err
        (err isa DomainError || err isa LinearAlgebra.PosDefException) || rethrow()
        throw(ArgumentError("R must be a strict correlation matrix"))
    end
end

function (::Type{tEVTail{d}})(ν::Real, ρ::Real) where {d}
    d >= 2 || throw(ArgumentError("extremal-t dimension must be at least two"))
    νf, ρf = promote(float(ν), float(ρ))
    νf > 0 || throw(ArgumentError("ν must be > 0"))
    lower = -inv(d - 1)
    ρf > lower || throw(ArgumentError("equicorrelation ρ must satisfy ρ > -1/(d-1) in dimension d=$d"))
    ρf <= 1 || throw(ArgumentError("ρ must be ≤ 1"))

    # In d=2 every correlation matrix is exchangeable, so interior scalar input
    # is canonicalized to the general matrix representation. Keep ρ=1 in the
    # scalar chart because the general chart is intentionally strict SPD.
    if d == 2 && ρf < 1
        R = T = typeof(ρf)
        M = fill(T(ρf), 2, 2)
        M[1, 1] = M[2, 2] = one(T)
        return _tev_general_tail(Val(2), νf, M)
    end
    T = typeof(νf + ρf)
    return tEVTail{d,:exchangeable,T}(T(νf), T(ρf))
end
tEVTail(ν::Real, ρ::Real) = tEVTail{2}(ν, ρ)

(::Type{tEVTail{d}})(ν::Real, R::AbstractMatrix) where {d} =
    _tev_general_tail(Val(d), ν, R)
tEVTail(ν::Real, R::AbstractMatrix) =
    tEVTail{size(R, 1)}(ν, R)

@inline _tev_representation(::tEVTail{d,R}) where {d,R} = R
@inline function limit_kind(tail::tEVTail, ::Val)
    _tev_representation(tail) === :exchangeable && isone(tail.parameter) ? M_LIMIT : NO_LIMIT
end

const tEVCopula{d,R,T} = ExtremeValueCopula{d,tEVTail{d,R,T}}
function (::Type{tEVCopula{d}})(ν::Real, ρ::Real) where {d}
    return _wrap_extreme_value(Val(d), tEVTail{d}(ν, ρ))
end
(::Type{tEVCopula})(d::Int, ν::Real, ρ::Real) =
    _wrap_extreme_value(Val(d), tEVTail{d}(ν, ρ))
function (::Type{tEVCopula{d}})(ν::Real, R::AbstractMatrix) where {d}
    return _wrap_extreme_value(Val(d), tEVTail{d}(ν, R))
end
function (::Type{tEVCopula})(d::Int, ν::Real, R::AbstractMatrix)
    return _wrap_extreme_value(Val(d), tEVTail(ν, R))
end

_is_valid_in_dim(::tEVTail{D}, d::Int) where {D} = D == d

_tev_rho(tail::tEVTail{D,:exchangeable}) where {D} = tail.parameter
_tev_rho(tail::tEVTail{D,:general}) where {D} = tail.parameter[1, 2]
function _tev_correlation(tail::tEVTail{D,:exchangeable}, d::Int) where {D}
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    ρ = tail.parameter
    R = fill(float(ρ), d, d)
    @inbounds for i in 1:d
        R[i, i] = one(eltype(R))
    end
    return R
end
_tev_correlation(tail::tEVTail{D,:general}, d::Int) where {D} = begin
    D == d || throw(DimensionMismatch("tail dimension $D does not match d=$d"))
    tail.parameter
end

# Preserve the public natural representation while canonicalizing d=2 storage.
Distributions.params(C::ExtremeValueCopula{2,<:tEVTail}) =
    (C.tail.ν, _tev_rho(C.tail))
Distributions.params(C::ExtremeValueCopula{D,<:tEVTail{D,:exchangeable}}) where {D} =
    (C.tail.ν, C.tail.parameter)
Distributions.params(C::ExtremeValueCopula{D,<:tEVTail{D,:general}}) where {D} =
    (C.tail.ν, copy(C.tail.parameter))

_tail_constructor_parameter_names(::Type{<:tEVTail{2}}, _) = (:ν, :ρ)
_tail_constructor_parameter_names(::Type{<:tEVTail{D,:exchangeable}}, _) where {D} = (:ν, :ρ)
_tail_constructor_parameter_names(::Type{<:tEVTail{D,:general}}, _) where {D} = (:ν, :R)

_available_fitting_methods(
    ::Type{<:ExtremeValueCopula{D,<:tEVTail{D,:general}} where D}, d,
) = d == 2 ? (:mle,) : ()

'''
text = text[:start] + new + text[end:]
text = text.replace(
    'function ℓ(tail::tEVTail{<:Any,<:Real}, x)\n    isone(something(tail.ρ)) && return maximum(x)',
    'function ℓ(tail::tEVTail{D,:exchangeable}, x) where {D}\n    isone(tail.parameter) && return maximum(x)',
    1,
)
text = text.replace(
    'ℓ(tail::tEVTail{<:Any,<:AbstractMatrix}, x) =\n    all(isone, something(tail.R)) ? maximum(x) : _tev_stdf(tail.ν, something(tail.R), x)',
    'ℓ(tail::tEVTail{D,:general}, x) where {D} =\n    _tev_stdf(tail.ν, tail.parameter, x)',
    1,
)
p.write_text(text)

# -----------------------------------------------------------------------------
# Central bridge: tails now own their geometry directly.
# -----------------------------------------------------------------------------
p = Path('src/ParamorphFitting.jl')
text = p.read_text()
old = '''function _tail_prototype(::Type{<:HuslerReissTail}, ::Val{d}) where {d}
    geometry = _component_prototype(_HuslerReissScalarGeometry{Float64})
    return HuslerReissTail(geometry.θ)
end

function _tail_prototype(::Type{<:tEVTail}, ::Val{d}) where {d}
    geometry = _component_prototype(
        _tEVScalarGeometry{Float64}, (; dimension=d),
    )
    return tEVTail(geometry.ν, geometry.ρ)
end
'''
new = '''function _tail_prototype(TT::Type{<:HuslerReissTail}, ::Val{d}) where {d}
    U = Base.unwrap_unionall(TT)
    encoded_d, encoded_rep = U.parameters[1], U.parameters[2]
    rep = encoded_rep isa TypeVar ? (d == 2 ? :general : :exchangeable) : encoded_rep
    encoded_d isa TypeVar || encoded_d == d || throw(DimensionMismatch(
        "Hüsler-Reiss tail dimension $encoded_d does not match d=$d",
    ))
    return _component_prototype(HuslerReissTail{d,rep,Float64}, (; dimension=d))
end

function _tail_prototype(TT::Type{<:tEVTail}, ::Val{d}) where {d}
    U = Base.unwrap_unionall(TT)
    encoded_d, encoded_rep = U.parameters[1], U.parameters[2]
    rep = encoded_rep isa TypeVar ? (d == 2 ? :general : :exchangeable) : encoded_rep
    encoded_d isa TypeVar || encoded_d == d || throw(DimensionMismatch(
        "extremal-t tail dimension $encoded_d does not match d=$d",
    ))
    return _component_prototype(tEVTail{d,rep,Float64}, (; dimension=d))
end
'''
assert old in text, 'legacy HR/tEV prototype helpers not found'
text = text.replace(old, new, 1)
marker = '# Representation variants use private @paramorph geometry objects.'
if marker in text:
    text = text[:text.index(marker)].rstrip() + '\n'
p.write_text(text)

print('direct Paramorph representation-tagged EV tails prepared')
