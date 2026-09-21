"""
    AbstractReflectedCopula{d,CT} <: Copula{d}

Internal supertype shared by copulas obtained by reflecting coordinates of an
underlying copula of type `CT`. Implementations provide [`basecopula`](@ref)
and [`flipmask`](@ref); the common distribution and conditioning behaviour is
then inherited from this interface.
"""
abstract type AbstractReflectedCopula{d,CT} <: Copula{d} end

"""
    SurvivalCopula(C)
    SurvivalCopula(C, flips)
    SurvivalCopula{d}(C, flips)
    SurvivalCopula(d, C, flips)
Construct a reflected version of a copula. `SurvivalCopula(C)` reflects every
coordinate and therefore constructs the survival copula in the usual sense.
The existing `SurvivalCopula(C, flips)` form reflects only the given indices.

The ergonomic constructor `SurvivalCopula(C, flips)` accepts the indices to flip:

    SurvivalCopula(ClaytonCopula(4, θ), (2,3))

If `U ∼ C`, the resulting random vector has coordinates `1-U[i]` for
`i ∈ flips` and `U[i]` otherwise. Its CDF is the corresponding
inclusion--exclusion transform of `C`; it is not obtained merely by evaluating
`C` at reflected arguments. Indices are one-based, unique, and must lie in
`1:length(C)`. A Boolean tuple of length `d` is also accepted as a flip mask.

Notes:
- In the bivariate case, this includes the usual 90/180/270-degree "rotations" of a copula family.
- The resulting object is handled like the base copula: same API (cdf, pdf/logpdf, rand, fit) and uniform marginals in ``[0,1]^d``.
- Since the flip pattern belongs to an instance rather than its type, preserve a
  particular pattern during fitting with
  `fit(typeof(S), U; flips=(...))`. If `flips` is omitted, fitting flips every
  coordinate. The fitting methods available are those of the underlying copula.

Flipping all coordinates gives the usual survival copula. Partial flips can
exchange upper- and lower-tail behavior and can change the sign of pairwise
association; they do not change the uniform margins.

See also: [`Copula`](@ref), [`subsetdims`](@ref), [`condition`](@ref),
[`Rotated90Copula`](@ref), [`Rotated180Copula`](@ref),
[`Rotated270Copula`](@ref), [`Distributions.fit`](@ref).

References:
* [nelsen2006](@cite) Nelsen (2006), An introduction to copulas.
"""
struct SurvivalCopula{d,CT} <: AbstractReflectedCopula{d,CT}
    C::CT
    flipmask::NTuple{d,Bool}
    function SurvivalCopula{d}(C::Copula{d}, flips) where {d}
        mask = _survival_flipmask(Val(d), flips)
        return new{d,typeof(C)}(C, mask)
    end
end

"""
    Rotated90Copula(C)
    Rotated90Copula{2}(C)
    Rotated90Copula(2, C)

Construct the 90-degree rotation of the bivariate copula `C`. Its first
coordinate is reflected. The reflection pattern is encoded by the concrete
type and is therefore preserved by fitting.

See also: [`SurvivalCopula`](@ref), [`Copulas.flipmask`](@ref),
[`Copulas.flips`](@ref).
"""
struct Rotated90Copula{d,CT} <: AbstractReflectedCopula{d,CT}
    C::CT
    Rotated90Copula{2}(C::Copula{2}) = new{2,typeof(C)}(C)
end
Rotated90Copula(C::Copula{2}) = Rotated90Copula{2}(C)
function Rotated90Copula(d::Integer, C::Copula{2})
    d == 2 || throw(DimensionMismatch("Rotated90Copula is only defined in dimension 2"))
    return Rotated90Copula{2}(C)
end

"""
    Rotated180Copula(C)
    Rotated180Copula{2}(C)
    Rotated180Copula(2, C)

Construct the 180-degree rotation of the bivariate copula `C`. Both
coordinates are reflected, so this is the bivariate survival copula. The
reflection pattern is encoded by the concrete type and is preserved by fitting.

See also: [`SurvivalCopula`](@ref), [`Copulas.flipmask`](@ref),
[`Copulas.flips`](@ref).
"""
struct Rotated180Copula{d,CT} <: AbstractReflectedCopula{d,CT}
    C::CT
    Rotated180Copula{2}(C::Copula{2}) = new{2,typeof(C)}(C)
end
Rotated180Copula(C::Copula{2}) = Rotated180Copula{2}(C)
function Rotated180Copula(d::Integer, C::Copula{2})
    d == 2 || throw(DimensionMismatch("Rotated180Copula is only defined in dimension 2"))
    return Rotated180Copula{2}(C)
end

"""
    Rotated270Copula(C)
    Rotated270Copula{2}(C)
    Rotated270Copula(2, C)

Construct the 270-degree rotation of the bivariate copula `C`. Its second
coordinate is reflected. The reflection pattern is encoded by the concrete
type and is therefore preserved by fitting.

See also: [`SurvivalCopula`](@ref), [`Copulas.flipmask`](@ref),
[`Copulas.flips`](@ref).
"""
struct Rotated270Copula{d,CT} <: AbstractReflectedCopula{d,CT}
    C::CT
    Rotated270Copula{2}(C::Copula{2}) = new{2,typeof(C)}(C)
end
Rotated270Copula(C::Copula{2}) = Rotated270Copula{2}(C)
function Rotated270Copula(d::Integer, C::Copula{2})
    d == 2 || throw(DimensionMismatch("Rotated270Copula is only defined in dimension 2"))
    return Rotated270Copula{2}(C)
end

"""
    basecopula(C)

Return the underlying copula transformed by the reflected copula `C`.

See also: [`flipmask`](@ref), [`flips`](@ref).
"""
basecopula(C::AbstractReflectedCopula) = C.C

"""
    flipmask(C)

Return the Boolean tuple identifying the reflected coordinates of `C`.
The tuple has one entry per coordinate and is suitable for dispatch-neutral,
type-stable internal algorithms. Use [`flips`](@ref) for the corresponding
one-based indices.
"""
flipmask(C::SurvivalCopula) = C.flipmask
flipmask(::Rotated90Copula) = (true, false)
flipmask(::Rotated180Copula) = (true, true)
flipmask(::Rotated270Copula) = (false, true)

"""
    flips(C)

Return the one-based indices of the reflected coordinates of `C`.

See also: [`flipmask`](@ref), [`basecopula`](@ref).
"""
flips(C::AbstractReflectedCopula) = _survival_flipindices(flipmask(C))

Base.eltype(C::AbstractReflectedCopula) = eltype(basecopula(C))
Paramorph.is_paramorph_type(::Type{<:AbstractReflectedCopula}) = true
Paramorph.transformation_schema(C::AbstractReflectedCopula) =
    Paramorph.transformation_schema(basecopula(C))
Paramorph.transformation_schema(C::AbstractReflectedCopula, ::NamedTuple) =
    Paramorph.transformation_schema(C)
Paramorph.parameter_values(C::AbstractReflectedCopula) =
    Paramorph.parameter_values(basecopula(C))
Paramorph.reconstruct_struct(C::SurvivalCopula{d}, values::NamedTuple) where {d} =
    SurvivalCopula{d}(Paramorph.reconstruct_struct(basecopula(C), values), flipmask(C))
Paramorph.reconstruct_struct(C::Rotated90Copula, values::NamedTuple) =
    Rotated90Copula(Paramorph.reconstruct_struct(basecopula(C), values))
Paramorph.reconstruct_struct(C::Rotated180Copula, values::NamedTuple) =
    Rotated180Copula(Paramorph.reconstruct_struct(basecopula(C), values))
Paramorph.reconstruct_struct(C::Rotated270Copula, values::NamedTuple) =
    Rotated270Copula(Paramorph.reconstruct_struct(basecopula(C), values))

function _survival_flipmask(::Val{d}, flips::NTuple{d,Bool}) where {d}
    return flips
end
function _survival_flipmask(::Val{d}, flips) where {d}
    indices = Tuple(flips)
    all(i -> i isa Integer && 1 <= i <= d, indices) || throw(ArgumentError("flip indices must belong to 1:$d"))
    length(unique(indices)) == length(indices) || throw(ArgumentError("flip indices must be unique"))
    return ntuple(i -> i in indices, d)
end
_survival_flipindices(mask::NTuple{d,Bool}) where {d} =
    Tuple(i for i in 1:d if mask[i])

SurvivalCopula(C::Copula{d}, flips) where {d} = SurvivalCopula{d}(C, flips)
SurvivalCopula(d::Integer, C::Copula, flips) = SurvivalCopula{d}(C, flips)
SurvivalCopula(C::Copula{d}) where {d} = SurvivalCopula{d}(C, ntuple(_ -> true, d))

copula_measure_style(::Type{<:SurvivalCopula{d,CT}}) where {d,CT} =
    copula_measure_style(CT)
copula_measure_style(::Type{<:AbstractReflectedCopula{d,CT}}) where {d,CT} =
    copula_measure_style(CT)
copula_measure_style(C::AbstractReflectedCopula) = copula_measure_style(basecopula(C))
_is_empirical_copula(C::AbstractReflectedCopula) =
    _is_empirical_copula(basecopula(C))

function _survival_reverse!(u, mask::Tuple)
    if ndims(u) == 1
        for i in eachindex(mask)
            mask[i] && (u[i] = 1 - u[i])
        end
    else
        for i in eachindex(mask)
            mask[i] && (u[i,:] .= 1 .- u[i,:])
        end
    end
    return u
end
_survival_reverse(u, mask::Tuple) =
    [mask[i] ? 1 - uᵢ : uᵢ for (i, uᵢ) in enumerate(u)]

function _survival_cdf(C, u, mask::NTuple{d,Bool}) where {d}
    i = findlast(identity, mask)
    isnothing(i) && return Distributions.cdf(C, u)
    remaining = ntuple(k -> k == i ? false : mask[k], d)
    v = collect(u)
    v[i] = 1 - v[i]
    r2 = _survival_cdf(C, v, remaining)
    v[i] = 1
    r1 = _survival_cdf(C, v, remaining)
    return r1 - r2
end
_cdf(C::AbstractReflectedCopula, u) = _survival_cdf(basecopula(C), u, flipmask(C))
Distributions._logpdf(C::AbstractReflectedCopula, u) =
    Distributions._logpdf(basecopula(C), _survival_reverse(u, flipmask(C)))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::AbstractReflectedCopula{d}, A::AbstractMatrix{T}) where {d,T<:Real}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    Distributions._rand!(rng, basecopula(C), A)
    return _survival_reverse!(A, flipmask(C))
end

# Fitting is structural: transform the observations, fit the base family, then
# reconstruct the reflected wrapper. Parameter geometry belongs to the base
# family and is not duplicated on the wrapper type.
Distributions.params(S::AbstractReflectedCopula) = Distributions.params(basecopula(S))

function _fit_reflected(::Type{subCT}, U, m, mask; kwargs...) where {subCT}
    Uflip = copy(U)
    _survival_reverse!(Uflip, mask)
    return _fit(subCT, Uflip, m; kwargs...)
end

function _fit_dispatch(
    CT::Type{<:SurvivalCopula{d,subCT}}, U, ::Val{d}, m::Val;
    flips=nothing, kwargs...,
) where {d,subCT}
    mask = isnothing(flips) ? ntuple(_ -> true, d) : _survival_flipmask(Val(d), flips)
    C = _fit_reflected(subCT, U, m, mask; kwargs...)
    return SurvivalCopula{d}(C, mask)
end

_rotation_type_flipmask(::Type{<:Rotated90Copula}) = (true, false)
_rotation_type_flipmask(::Type{<:Rotated180Copula}) = (true, true)
_rotation_type_flipmask(::Type{<:Rotated270Copula}) = (false, true)

function _fit_rotation(::Type{RT}, U, m; kwargs...) where {subCT,RT<:Rotated90Copula{2,subCT}}
    C = _fit_reflected(subCT, U, m, _rotation_type_flipmask(RT); kwargs...)
    return Rotated90Copula(C)
end
function _fit_rotation(::Type{RT}, U, m; kwargs...) where {subCT,RT<:Rotated180Copula{2,subCT}}
    C = _fit_reflected(subCT, U, m, _rotation_type_flipmask(RT); kwargs...)
    return Rotated180Copula(C)
end
function _fit_rotation(::Type{RT}, U, m; kwargs...) where {subCT,RT<:Rotated270Copula{2,subCT}}
    C = _fit_reflected(subCT, U, m, _rotation_type_flipmask(RT); kwargs...)
    return Rotated270Copula(C)
end

_fit_dispatch(CT::Type{<:Rotated90Copula}, U, ::Val{2}, m::Val; kwargs...) =
    _fit_rotation(CT, U, m; kwargs...)
_fit_dispatch(CT::Type{<:Rotated180Copula}, U, ::Val{2}, m::Val; kwargs...) =
    _fit_rotation(CT, U, m; kwargs...)
_fit_dispatch(CT::Type{<:Rotated270Copula}, U, ::Val{2}, m::Val; kwargs...) =
    _fit_rotation(CT, U, m; kwargs...)

_available_fitting_methods(::Type{<:SurvivalCopula{D,subCT}}, d) where {D,subCT} = _available_fitting_methods(subCT, d)
_available_fitting_methods(::Type{<:AbstractReflectedCopula{D,subCT}}, d) where {D,subCT} = _available_fitting_methods(subCT, d)

# Conditioning bindings colocated
function distortion(S::AbstractReflectedCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,<:Real}, i::Int) where {D,p}
    mask = flipmask(S)
    uⱼₛ′ = ntuple(k -> mask[js[k]] ? one(uⱼₛ[k]) - uⱼₛ[k] : uⱼₛ[k], p)
    base = distortion(basecopula(S), js, uⱼₛ′, i)
    return FlipDistortion(base, mask[i])
end
function conditional_copula(S::AbstractReflectedCopula{D}, js, uⱼₛ) where {D}
    mask = flipmask(S)
    uⱼₛ′ = Tuple(mask[j] ? 1 - float(u) : float(u) for (j,u) in zip(js, uⱼₛ))
    CC_base = conditional_copula(basecopula(S), js, uⱼₛ′)
    I = Tuple(setdiff(1:D, Tuple(collect(Int, js))))
    flip_positions = Tuple(p for (p, idx) in enumerate(I) if mask[idx])
    return SurvivalCopula(CC_base, flip_positions)
end

# Subsetting colocated: subset and remap flipped indices to the new positions
function SubsetCopula(C::AbstractReflectedCopula{d}, dims::NTuple{p, Int}) where {d,p}
    mask = flipmask(C)
    newflips = Tuple(k for (k, i) in enumerate(dims) if mask[i])
    return SurvivalCopula(subsetdims(basecopula(C), dims), newflips)
end

function τ(C::AbstractReflectedCopula{2})
    # For bivariate, flipping one margin negates tau, flipping both leaves tau unchanged
    if count(identity, flipmask(C)) % 2 == 1
        return -τ(basecopula(C))
    else
        return τ(basecopula(C))
    end
end
