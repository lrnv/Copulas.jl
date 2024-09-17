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
    function SubsetCopula(C::Copula{d},dims) where d
        if Tuple(dims) == Tuple(1:d)
            return C
        elseif length(dims)==1
            return Distributions.Uniform()
        end
        @assert all(dims .<= d)
        return new{length(dims), typeof(C)}(C,Tuple(Int.(dims)))
    end
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

# A few specialized constructors: 
SubsetCopula(C::GaussianCopula,dims) = length(dims) == 1 ? Distributions.Uniform() : GaussianCopula(C.Σ[collect(dims),collect(dims)])
SubsetCopula(C::TCopula{d,df,MT},dims) where {d,df,MT} = length(dims) == 1 ? Distributions.Uniform() : TCopula(df, C.Σ[collect(dims),collect(dims)])
SubsetCopula(C::ArchimedeanCopula{d,TG},dims) where {d,TG} = length(dims) == 1 ? Distributions.Uniform() : ArchimedeanCopula(length(dims), C.G) # in particular for the independence this will work. 

# We could add a few more for performance if needed: EmpiricalCopula, others... 



"""
    subsetdims(C::Copula,dims)
    subsetdims(D::SklarDist, dims)

If ``(X_1,...,X_n)`` is the random vector corresponding to the model `C` or `D`, this returns the distribution on `(` ``X_i`` `for i in dims)`, preserving the dependence structure between the dimensions in `dims`. There are specialized methods for some copulas. 
"""
subsetdims(C::Copula{d},dims) where d = SubsetCopula(C,dims)
function subsetdims(D::SklarDist, dims)
    if length(dims)==1 
        return D.m[dims[1]]
    else
        return SklarDist(subsetdims(D.C,dims), Tuple(D.m[i] for i in dims))
    end
end