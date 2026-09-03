"""
    SurvivalCopula(C, flips)
    SurvivalCopula{d,CT}

Construct the survival (flipped) version of a copula by flipping the arguments at the given indices.

The ergonomic constructor `SurvivalCopula(C, flips)` accepts the indices to flip:

    SurvivalCopula(ClaytonCopula(4, θ), (2,3))

The flip pattern is stored in the object rather than its type, so distinct
rotations of the same copula share one concrete type.

For a copula `C` in dimension `d` and indices `i₁, ..., iₖ ∈ 1:d`, the survival copula flips the corresponding arguments:

```math
    S(u_1,\\ldots,u_d) = C(v_1,\\ldots,v_d), \\quad v_j = \\begin{cases} 1-u_j & j \\in \\text{flips} \\\\ u_j & \\text{otherwise} \\end{cases}
```

Notes:
- In the bivariate case, this includes the usual 90/180/270-degree "rotations" of a copula family.
- The resulting object is handled like the base copula: same API (cdf, pdf/logpdf, rand, fit) and uniform marginals in ``[0,1]^d``.

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
function DistortionFromCop(S::SurvivalCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,p}
    uⱼₛ′ = ntuple(k -> (S.flipmask[js[k]] ? 1 - uⱼₛ[k] : uⱼₛ[k]), p)
    base = DistortionFromCop(S.C, js, uⱼₛ′, i)
    return FlipDistortion(base, S.flipmask[i])
end
function ConditionalCopula(S::SurvivalCopula{D}, js, uⱼₛ) where {D}
    uⱼₛ′ = Tuple(S.flipmask[j] ? 1 - float(u) : float(u) for (j,u) in zip(js, uⱼₛ))
    CC_base = ConditionalCopula(S.C, js, uⱼₛ′)
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
