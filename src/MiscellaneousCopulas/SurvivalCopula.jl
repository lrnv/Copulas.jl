"""
    SurvivalCopula(C, flips)
    SurvivalCopula{d,CT,flips}

Construct the survival (flipped) version of a copula by flipping the arguments at the given indices.

**Type-level encoding:**
The indices to flip are encoded at the type level as a tuple of integers, e.g. `SurvivalCopula{4,ClaytonCopula,(2,3)}`. This enables compile-time specialization and dispatch, and ensures that the flipping pattern is part of the type.

**Sugar constructor:**
The ergonomic constructor `SurvivalCopula(C, flips::Tuple)` infers the type parameters from the arguments, so you can write:

    SurvivalCopula(ClaytonCopula(4, θ), (2,3))

which is equivalent to the explicit type form:

    SurvivalCopula{4,ClaytonCopula,(2,3)}(ClaytonCopula(4, θ))

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
struct SurvivalCopula{d,CT,flips} <: Copula{d}
    C::CT
    function SurvivalCopula{d,CT,flips}(C::Copula{d}) where {d,CT,flips}
        if length(flips) == 0
            return C
        end
        if typeof(C) == IndependentCopula
            return C
        end
        return new{d,typeof(C),flips}(C)
    end
    SurvivalCopula(C::CT, flips::Tuple) where {d, CT<:Copula{d}} = SurvivalCopula{d,CT,flips}(C)
    SurvivalCopula(C::CT, flips) where {d, CT<:Copula{d}} = SurvivalCopula(C, tuple(flips...))
    SurvivalCopula{D,CT,flips}(d::Int, args...;kwargs...) where {D, CT, flips} = SurvivalCopula{d,CT,flips}(CT(d, args...; kwargs...))
end

function reverse!(u, idx::Tuple)
    if ndims(u) == 1
        for i in idx
            u[i] = 1 - u[i]
        end
    else
        for i in idx
            u[i,:] .= 1 .- u[i,:]
        end
    end
    return u
end
reverse(u, idx::Tuple) = [i ∈ idx ? 1-uᵢ : uᵢ for (i,uᵢ) in enumerate(u)]
function _cdf(C::SurvivalCopula{d,CT,flips}, u) where {d,CT,flips}
    i = flips[end]
    newC = SurvivalCopula{d,CT,Base.tuple(flips[1:end-1]...)}(C.C)
    v = reverse(u, (i,))
    r2 = _cdf(newC,v)
    v[i] = 1
    r1 = _cdf(newC,v)
    return r1 - r2
end
Distributions._logpdf(C::SurvivalCopula{d,CT,flips}, u) where {d,CT,flips} = Distributions._logpdf(C.C, reverse(u, flips))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SurvivalCopula{d,CT,flips}, x::AbstractVector{T}) where {d,CT,flips,T<:Real}
    Distributions._rand!(rng, C.C, x)
    reverse!(x, flips)
end

# Fitting: delegate to the base copula after flipping the requested indices in U
Distributions.params(S::SurvivalCopula) = Distributions.params(S.C)

# Twice the same function but cannot be joined... weirdly. 
function _fit(::Type{<:SurvivalCopula{d,subCT,flips}}, U, m::Union{Val{:itau}, Val{:irho}, Val{:ibeta}}; kwargs...) where {d,subCT,flips}
    dU = size(U, 1)
    @assert dU == d "Dimension mismatch in SurvivalCopula fit."
    Uflip = copy(U)
    reverse!(Uflip, flips)
    C, meta = _fit(subCT, Uflip, m; kwargs...)
    return SurvivalCopula{d,subCT,flips}(C), meta
end
function _fit(::Type{<:SurvivalCopula{d,subCT,flips}}, U, m::Val{:mle}; kwargs...) where {d,subCT,flips}
    dU = size(U, 1)
    @assert dU == d "Dimension mismatch in SurvivalCopula fit."
    Uflip = copy(U)
    reverse!(Uflip, flips)
    C, meta = _fit(subCT, Uflip, m; kwargs...)
    return SurvivalCopula{d,subCT,flips}(C), meta
end

_available_fitting_methods(::Type{<:SurvivalCopula{d,subCT,flips}}) where {d, subCT, flips} = _available_fitting_methods(subCT)
_example(CT::Type{<:SurvivalCopula{D,subCT,flips}}, d) where {D, subCT, flips} = SurvivalCopula(_example(subCT, d), flips)


# Parameter transfer for fitting: delegate to underlying copula
function _unbound_params(::Type{<:SurvivalCopula{d,CT,flips}}, d_, θ) where {d,CT,flips}
    return _unbound_params(CT, d_, θ)
end

function _rebound_params(::Type{<:SurvivalCopula{d,CT,flips}}, d_, α) where {d,CT,flips}
    return _rebound_params(CT, d_, α)
end



# Conditioning bindings colocated
function DistortionFromCop(S::SurvivalCopula{D,CT,flips}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,CT,flips,p}
    uⱼₛ′ = ntuple(k -> (js[k] in flips ? 1 - uⱼₛ[k] : uⱼₛ[k]), p)
    base = DistortionFromCop(S.C, js, uⱼₛ′, i)
    return (i in flips) ? FlipDistortion(base) : base
end
function ConditionalCopula(S::SurvivalCopula{D,CT,flips}, js, uⱼₛ) where {D,CT,flips}
    uⱼₛ′ = Tuple(j in flips ? 1 - float(u) : float(u) for (j,u) in zip(js, uⱼₛ))
    CC_base = ConditionalCopula(S.C, js, uⱼₛ′)
    I = Tuple(setdiff(1:D, Tuple(collect(Int, js))))
    flip_positions = Tuple(p for (p, idx) in enumerate(I) if idx in flips)
    return (length(flip_positions) == 0) ? CC_base : SurvivalCopula{length(I), typeof(CC_base), typeof(flip_positions)}(CC_base)
end

# Subsetting colocated: subset and remap flipped indices to the new positions
function SubsetCopula(C::SurvivalCopula{d,CT,flips}, dims::NTuple{p, Int}) where {d,CT,flips,p}
    newflips = Tuple(i for i in flips if i in dims)
    return SurvivalCopula(subsetdims(C.C, dims), newflips)
end


function τ(C::SurvivalCopula{2,CT,flips}) where {CT,flips}
    # For bivariate, flipping one margin negates tau, flipping both leaves tau unchanged
    if length(flips) % 2 == 1
        return -τ(C.C)
    else
        return τ(C.C)
    end
end