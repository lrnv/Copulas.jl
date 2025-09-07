"""
    SubsetCopula{d,CT}

Fields:
  - `C::CT` - The copula
  - `dims::Tuple{Int64}` - a Tuple representing which dimensions are used. 

Constructor

    SubsetCopula(C::Copula,dims)

This class allows to construct a random vector corresponding to a few dimensions of the starting copula. If ``(X_1,...,X_n)`` is the random vector corresponding to the copula `C`, this returns the copula of `(` ``X_i`` `for i in dims)`. The dependence structure is preserved. There are specialized methods for some copulas. 
"""
struct SubsetCopula{d,CT} <: Copula{d}
    C::CT
    dims::NTuple{d,Int64}

    function SubsetCopula(C::Copula{d}, dims::NTuple{p, Int64}) where {d, p}
        @assert 2 <= p <= d "You cannot construct a subsetcopula with dimension p=1 or p > d (d = $d, p = $p provided)"
        dims == Tuple(1:d) && return C
        @assert all(dims .<= d)
        return new{p, typeof(C)}(C,Tuple(Int.(dims)))
    end
end
function Base.show(io::IO, C::SubsetCopula)
    print(io, "SubsetCopula($(C.C), $(C.dims))")
end
Base.eltype(C::SubsetCopula{d,CT}) where {d,CT} = Base.eltype(C.C)
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SubsetCopula{d,CT}, x::AbstractVector{T}) where {T<:Real, d,CT}
    u = Random.rand(rng,C.C)
    x .= (u[i] for i in C.dims)
    return x
end
function _cdf(C::SubsetCopula{d,CT},u) where {d,CT}
    # Simplyu saturate dimensions that are not choosen.
    v = ones(length(C.C))
    for (i,j) in enumerate(C.dims)
        v[j] = u[i]
    end 
    return Distributions.cdf(C.C,v)
end

# Kendall tau and spearman rho are symetric measures in bivaraite cases: 
τ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = τ(C.C)
ρ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = ρ(C.C)

"""
    subsetdims(C::Copula, dims::NTuple{p, Int64})
    subsetdims(D::SklarDist, dims)

Return a new copula or Sklar distribution corresponding to the subset of dimensions specified by `dims`.

# Arguments
- `C::Copula`: The original copula object.
- `D::SklarDist`: The original Sklar distribution.
- `dims::NTuple{p, Int64}`: Tuple of indices representing the dimensions to keep.

# Returns
- A `SubsetCopula` or a new `SklarDist` object corresponding to the selected dimensions. If `p == 1`, returns a `Uniform` distribution or the corresponding marginal.

# Details
This function extracts the dependence structure among the specified dimensions from the original copula or Sklar distribution. Specialized methods exist for some copula types to ensure efficiency and correctness.
"""
function subsetdims(C::Copula{d},dims::NTuple{p, Int64}) where {d,p}
    p==1 && return Distributions.Uniform()
    dims==tuple(1:d) && return C
    @assert p < d
    @assert length(unique(dims))==length(dims)
    @assert all(dims .<= d)
    return SubsetCopula(C,dims)
end
function subsetdims(D::SklarDist, dims::NTuple{p, Int64}) where p
    p==1 && return D.m[dims[1]]
    return SklarDist(subsetdims(D.C,dims), Tuple(D.m[i] for i in dims))
end

##################
### Specialized constructors are colocated in each copula file

###########################################################################
#####  Conditioning and subsetting bindings for SubsetCopula colocated here
###########################################################################
function SubsetCopula(CS::SubsetCopula{d,CT}, dims2::NTuple{p, Int64}) where {d,CT,p}
    @assert 2 <= p <= d
    return SubsetCopula(CS.C, ntuple(i -> CS.dims[dims2[i]], p))
end

@inline function DistortionFromCop(S::SubsetCopula, js::NTuple{p,Int64}, uⱼₛ::NTuple{p,T}, i::Int64) where {p,T}
    ibase = S.dims[i]
    jsbase = ntuple(k -> S.dims[js[k]], p)
    return DistortionFromCop(S.C, jsbase, uⱼₛ, ibase)
end

function ConditionalCopula(S::SubsetCopula{d,CT}, js, uⱼₛ) where {d,CT}
    Jbase = Tuple(S.dims[j] for j in js)
    CC_base = ConditionalCopula(S.C, Jbase, uⱼₛ)
    D = length(S.C); I = Tuple(setdiff(1:D, Jbase))
    dims_remain = Tuple(i for i in S.dims if !(i in Jbase))
    posmap = Dict(i => p for (p,i) in enumerate(I))
    dims_positions = Tuple(posmap[i] for i in dims_remain)
    return (length(dims_positions) == length(I)) ? CC_base : SubsetCopula(CC_base, dims_positions)
end