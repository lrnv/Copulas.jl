"""
    SurvivalCopula(C, indices)

Construct the survival version of a copula on selected indices. For a copula ``C`` in dimension ``d`` and indices ``i_1, \\ldots, i_k \\in \\{1,\\ldots,d\\}``, the survival copula flips the corresponding arguments. For example, for ``d=4`` and indices ``(2,3)``,

```math
S(u_1,\\ldots,u_4) = C(u_1, 1-u_2, 1-u_3, u_4).
```

Notes:
- In the bivariate case, this includes the usual 90/180/270-degree "rotations" of a copula family.
- The resulting object is handled like the base copula: same API (cdf, pdf/logpdf, rand, fit) and uniform marginals in ``[0,1]^d``.

References:
* [nelsen2006](@cite) Nelsen (2006), An introduction to copulas.
"""
struct SurvivalCopula{d,CT,VI} <: Copula{d}
    C::CT
    indices::VI
    function SurvivalCopula(C,indices)
        if typeof(C) == IndependentCopula
            return C
        end
        if length(indices) == 0
            return C
        end
        d = length(C)
        @assert all(indices .<= d)
        return new{d,typeof(C),typeof(indices)}(C,indices)
    end
end

function reverse!(u,idx)
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
reverse(u,idx) = [i ∈ idx ? 1-uᵢ : uᵢ for (i,uᵢ) in enumerate(u)]
function _cdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI}
    i = C.indices[end]
    newC = SurvivalCopula(C.C,C.indices[1:end-1])
    v = reverse(u, (i,))
    r2 = _cdf(newC,v)
    v[i] = 1
    r1 = _cdf(newC,v)
    return r1 - r2
end 
Distributions._logpdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI} = Distributions._logpdf(C.C,reverse(u, C.indices))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SurvivalCopula{d,CT,VI}, x::AbstractVector{T}) where {d,CT,VI,T<:Real}
    Distributions._rand!(rng,C.C,x)
    reverse!(x, C.indices)
end

# Fitting: delegate to the base copula after flipping the requested indices in U
Distributions.params(S::SurvivalCopula) = Distributions.params(S.C)

"""
    _fit(::Type{SurvivalCopula}, U, ::Val{method}; base::Type{<:Copula}, indices, kwargs...)

Fit a SurvivalCopula by flipping `U` on the given `indices` and delegating to `_fit(base, ...)` with the same method.
Required keywords:
  - base::Type{<:Copula}        underlying copula family type to fit
  - indices::Tuple{Vararg{Int}} tuple of indices to flip
Other keyword arguments are forwarded to the base `_fit`.
"""
function _fit(::Type{SurvivalCopula}, U,
              ::Val{method}; base::Type{<:Copula}, indices::Tuple{Vararg{Int}}, kwargs...) where {method}
    d = size(U, 1)
    @assert all(1 .<= indices .<= d) "indices must be in 1..d"
    Uflip = copy(U)
    @inbounds for i in indices
        Uflip[i, :] .= 1 .- U[i, :]
    end
    C, meta = _fit(base, Uflip, Val{method}(); kwargs...)
    return SurvivalCopula(C, indices), meta
end

# Convenience: dispatch on the SurvivalCopula type with embedded base type
function _fit(::Type{SurvivalCopula{d,subCT,VI}}, U, M::Val{:default}; indices::Tuple{Vararg{Int}}, kwargs...) where {d,subCT,VI}
    return _fit(SurvivalCopula, U, M; base=subCT, indices=indices, kwargs...)
end

# Conditioning bindings colocated
function DistortionFromCop(S::SurvivalCopula{D,CT,VI}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,Float64}, i::Int) where {D,CT,VI,p}
    flips = S.indices
    uⱼₛ′ = ntuple(k -> (js[k] in flips ? 1 - uⱼₛ[k] : uⱼₛ[k]), p)
    base = DistortionFromCop(S.C, js, uⱼₛ′, i)
    return (i in flips) ? FlipDistortion(base) : base
end
function ConditionalCopula(S::SurvivalCopula{D,CT,VI}, js, uⱼₛ) where {D,CT,VI}
    flips = S.indices
    uⱼₛ′ = Tuple(j in flips ? 1 - float(u) : float(u) for (j,u) in zip(js, uⱼₛ))
    CC_base = ConditionalCopula(S.C, js, uⱼₛ′)
    I = Tuple(setdiff(1:D, Tuple(collect(Int, js))))
    flip_positions = Tuple(p for (p, idx) in enumerate(I) if idx in flips)
    return (length(flip_positions) == 0) ? CC_base : SurvivalCopula(CC_base, flip_positions)
end

# Subsetting colocated: subset and remap flipped indices to the new positions
function SubsetCopula(C::SurvivalCopula{d, CT,VI}, dims::NTuple{p, Int}) where {d,CT,VI,p}
    return SurvivalCopula(subsetdims(C.C, dims), Tuple(setdiff(C.indices, setdiff(1:d,dims))))
end