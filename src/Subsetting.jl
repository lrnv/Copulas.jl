
###############################################################################
#####  Subsetting framework.
#####  User-facing function: `subsetdims()`
#####
#####  When implementing a new copula, you can overwrite: 
#####   - `SubsetCopula(C::Copula{d}, dims::NTuple{p, Int}) where {d, p}`
###############################################################################
"""
    SubsetCopula{d,CT}

Fields:
  - `C::CT` - The copula
  - `dims::Tuple{Int}` - a Tuple representing which dimensions are used. 

Constructor

    SubsetCopula(C::Copula,dims)

This class allows to construct a random vector corresponding to a few dimensions of the starting copula. If ``(X_1,...,X_n)`` is the random vector corresponding to the copula `C`, this returns the copula of `(` ``X_i`` `for i in dims)`. The dependence structure is preserved. There are specialized methods for some copulas. 
"""
struct SubsetCopula{d,CT} <: Copula{d}
    C::CT
    dims::NTuple{d,Int}
    function SubsetCopula(C::Copula{d}, dims::NTuple{p, Int}) where {d, p}
        @assert 2 <= p <= d "You cannot construct a subsetcopula with dimension p=1 or p > d (d = $d, p = $p provided)"
        dims == Tuple(1:d) && return C
        @assert all(dims .<= d)
        return new{p, typeof(C)}(C,Tuple(Int.(dims)))
    end
end
function SubsetCopula(CS::SubsetCopula{d,CT}, dims2::NTuple{p, Int}) where {d,CT,p}
    @assert 2 <= p <= d
    return SubsetCopula(CS.C, ntuple(i -> CS.dims[dims2[i]], p))
end
_available_fitting_methods(::Type{<:SubsetCopula}, d) = Tuple{}() # cannot be fitted. 
Base.eltype(C::SubsetCopula{d,CT}) where {d,CT} = Base.eltype(C.C)
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SubsetCopula{d,CT}, x::AbstractVector{T}) where {T<:Real, d,CT}
    u = Random.rand(rng,C.C)
    x .= (u[i] for i in C.dims)
    return x
end
function _cdf(C::SubsetCopula{d,CT},u) where {d,CT}
    # Simplyu saturate dimensions that are not choosen.
    v = ones(eltype(u), length(C.C))
    for (i,j) in enumerate(C.dims)
        v[j] = u[i]
    end 
    return Distributions.cdf(C.C,v)
end
function Distributions._logpdf(S::SubsetCopula{d,<:Copula{D}}, u) where {d,D}
    return log(_partial_cdf(S.C, Tuple(setdiff(1:D, S.dims)), S.dims, ones(D-d), u))
end

# Dependence metrics are symetric in bivariate cases: 
τ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = τ(C.C)
ρ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = ρ(C.C)
β(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = β(C.C)
γ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = γ(C.C)
ι(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = ι(C.C)
λₗ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = λₗ(C.C)
λᵤ(C::SubsetCopula{2,CT}) where {CT<:Copula{2}} = λᵤ(C.C)

"""
    subsetdims(C::Copula, dims::NTuple{p, Int})
    subsetdims(D::SklarDist, dims)

Return a new copula or Sklar distribution corresponding to the subset of dimensions specified by `dims`.

# Arguments
- `C::Copula`: The original copula object.
- `D::SklarDist`: The original Sklar distribution.
- `dims::NTuple{p, Int}`: Tuple of indices representing the dimensions to keep.

# Returns
- A `SubsetCopula` or a new `SklarDist` object corresponding to the selected dimensions. If `p == 1`, returns a `Uniform` distribution or the corresponding marginal.

# Details
This function extracts the dependence structure among the specified dimensions from the original copula or Sklar distribution. Specialized methods exist for some copula types to ensure efficiency and correctness.
"""
function subsetdims(C::Copula{d},dims::NTuple{p, Int}) where {d,p}
    p==1 && return Distributions.Uniform()
    dims==ntuple(i->i, d) && return C
    @assert p < d
    @assert length(unique(dims))==length(dims)
    @assert all(dims .<= d)
    return SubsetCopula(C,dims)
end
function subsetdims(D::SklarDist, dims::NTuple{p, Int}) where p
    p==1 && return D.m[dims[1]]
    return SklarDist(subsetdims(D.C,dims), Tuple(D.m[i] for i in dims))
end
subsetdims(C::Union{Copula, SklarDist}, dims) = subsetdims(C, Tuple(collect(Int, dims)))

# Pairwise dependence metrics, leveraging subsetting: 
function _as_biv(f::F, C::Copula{d}) where {F, d}
    first_val = f(SubsetCopula(C, (1,2)))
    K = ones(eltype(first_val),d,d)
    K[1,2] = first_val
    K[2,1] = first_val
    for i in 1:d
        for j in i+1:d
            if (i,j) != (1,2)
                K[i,j] = f(SubsetCopula(C, (i,j)))
                K[j,i] = K[i,j]
            end
        end
    end
    return K
end
StatsBase.corkendall(C::Copula)  = _as_biv(τ, C)
StatsBase.corspearman(C::Copula) = _as_biv(ρ, C)
corblomqvist(C::Copula)          = _as_biv(β, C)
corgini(C::Copula)               = _as_biv(γ, C)
corentropy(C::Copula)            = _as_biv(ι, C)
coruppertail(C::Copula)          = _as_biv(λᵤ, C)
corlowertail(C::Copula)          = _as_biv(λₗ, C)
