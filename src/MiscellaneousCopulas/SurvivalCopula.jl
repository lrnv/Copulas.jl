"""
    SurvivalCopula(C, flips)
    SurvivalCopula{d}(C, flips)
    SurvivalCopula(d, C, flips)
Construct the survival (flipped) version of a copula by flipping the arguments at the given indices.

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
[`Distributions.fit`](@ref).

References:
* [nelsen2006](@cite) Nelsen (2006), An introduction to copulas.
"""
struct SurvivalCopula{d,CT} <: Copula{d}
    C::CT
    flipmask::NTuple{d,Bool}
    function SurvivalCopula{d}(C::Copula{d}, flips) where {d}
        mask = _survival_flipmask(Val(d), flips)
        return new{d,typeof(C)}(C, mask)
    end
end
Base.eltype(C::SurvivalCopula) = eltype(C.C)

function _survival_flipmask(::Val{d}, flips::NTuple{d,Bool}) where {d}
    return flips
end
function _survival_flipmask(::Val{d}, flips) where {d}
    indices = Tuple(flips)
    all(i -> i isa Integer && 1 <= i <= d, indices) ||
        throw(ArgumentError("flip indices must belong to 1:$d"))
    length(unique(indices)) == length(indices) ||
        throw(ArgumentError("flip indices must be unique"))
    return ntuple(i -> i in indices, d)
end
_survival_flipindices(mask::NTuple{d,Bool}) where {d} =
    Tuple(i for i in 1:d if mask[i])

SurvivalCopula(C::Copula{d}, flips) where {d} = SurvivalCopula{d}(C, flips)
SurvivalCopula(d::Integer, C::Copula, flips) = SurvivalCopula{d}(C, flips)

copula_measure_style(::Type{<:SurvivalCopula{d,CT}}) where {d,CT} =
    copula_measure_style(CT)
copula_measure_style(C::SurvivalCopula) = copula_measure_style(C.C)

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
_cdf(C::SurvivalCopula, u) = _survival_cdf(C.C, u, C.flipmask)
Distributions._logpdf(C::SurvivalCopula, u) =
    Distributions._logpdf(C.C, _survival_reverse(u, C.flipmask))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SurvivalCopula{d}, A::AbstractMatrix{T}) where {d,T<:Real}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    Distributions._rand!(rng, C.C, A)
    return _survival_reverse!(A, C.flipmask)
end

# Fitting: delegate to the base copula after flipping the requested indices in U
Distributions.params(S::SurvivalCopula) = Distributions.params(S.C)

# Twice the same function but cannot be joined... weirdly. 
function _fit(::Type{<:SurvivalCopula{d,subCT}}, U, m::Union{Val{:itau}, Val{:irho}, Val{:ibeta}}; flips=nothing, kwargs...) where {d,subCT}
    flips = isnothing(flips) ? ntuple(i -> true, d) : _survival_flipmask(Val{d}(),flips)
    Uflip = copy(U)
    _survival_reverse!(Uflip, flips)
    C, meta = _fit(subCT, Uflip, m; kwargs...)
    return SurvivalCopula{d}(C, flips), meta
end
function _fit(::Type{<:SurvivalCopula{d,subCT}}, U, m::Val{:mle}; flips=nothing, kwargs...) where {d,subCT}
    flips = isnothing(flips) ? ntuple(i -> true, d) : _survival_flipmask(Val{d}(),flips)
    Uflip = copy(U)
    _survival_reverse!(Uflip, flips)
    C, meta = _fit(subCT, Uflip, m; kwargs...)
    return SurvivalCopula{d}(C, flips), meta
end

_available_fitting_methods(::Type{<:SurvivalCopula{D,subCT}}, d) where {D,subCT} =
    _available_fitting_methods(subCT, d)
_example(::Type{<:SurvivalCopula{D,subCT}}, d) where {D,subCT} =
    SurvivalCopula(_example(subCT, d), ())


# Parameter transfer for fitting: delegate to underlying copula
function _unbound_params(::Type{<:SurvivalCopula{d,CT}}, d_, θ) where {d,CT}
    return _unbound_params(CT, d_, θ)
end

function _rebound_params(::Type{<:SurvivalCopula{d,CT}}, d_, α) where {d,CT}
    return _rebound_params(CT, d_, α)
end



# Conditioning bindings colocated
function distortion(S::SurvivalCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,p}
    uⱼₛ′ = ntuple(k -> (S.flipmask[js[k]] ? 1 - uⱼₛ[k] : uⱼₛ[k]), p)
    base = distortion(S.C, js, uⱼₛ′, i)
    return FlipDistortion(base, S.flipmask[i])
end
function conditional_copula(S::SurvivalCopula{D}, js, uⱼₛ) where {D}
    uⱼₛ′ = Tuple(S.flipmask[j] ? 1 - float(u) : float(u) for (j,u) in zip(js, uⱼₛ))
    CC_base = conditional_copula(S.C, js, uⱼₛ′)
    I = Tuple(setdiff(1:D, Tuple(collect(Int, js))))
    flip_positions = Tuple(p for (p, idx) in enumerate(I) if S.flipmask[idx])
    return SurvivalCopula(CC_base, flip_positions)
end

# Subsetting colocated: subset and remap flipped indices to the new positions
function SubsetCopula(C::SurvivalCopula{d}, dims::NTuple{p, Int}) where {d,p}
    newflips = Tuple(k for (k, i) in enumerate(dims) if C.flipmask[i])
    return SurvivalCopula(subsetdims(C.C, dims), newflips)
end


function τ(C::SurvivalCopula{2})
    # For bivariate, flipping one margin negates tau, flipping both leaves tau unchanged
    if count(identity, C.flipmask) % 2 == 1
        return -τ(C.C)
    else
        return τ(C.C)
    end
end
