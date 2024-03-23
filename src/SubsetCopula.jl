"""
    SubsetCopula{d,CT}

Fields:
  - `C::CT` - The copula
  - `dims::Tuple{Int64}` - a Tuple representing which dimensions are used. 

Constructor

    SubsetCopula(C::Copula,dims)

This class allows to construct a random vector specified as X[dims] where X is the random vector correspnding to C and dims is a selection of dimensions. 
"""
struct SubsetCopula{d,CT} <: Copula{d}
    C::CT
    dims::NTuple{d,Int64}
    function SubsetCopula(C::Copula{d},dims) where d
        # if Tuple(dims) == Tuple(1:d)
        #     return C
        # end
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
SubsetCopula(C::GaussianCopula,dims) = GaussianCopula(C.Σ[collect(dims),collect(dims)])
SubsetCopula(C::TCopula{d,df,MT},dims) where {d,df,MT} = TCopula(df, C.Σ[collect(dims),collect(dims)])
SubsetCopula(C::ArchimedeanCopula{d,TG},dims) where {d,TG} = ArchimedeanCopula(length(dims), C.G) # in particular for the independence this will work. 

# We could add a few more for performance if needed: EmpiricalCopula, others... 



"""
    subsetdims(C::Copula,dims)
    subsetdims(D::SklarDist, dims)

If X is the random vector corresponding to `C` or `D`, this returns the distributions of  C[dims]. Has specialized methods for some copulas. 
"""
subsetdims(C::Copula{d},dims) where d = SubsetCopula(C,dims)
subsetdims(D::SklarDist, dims) = SklarDist(subsetdims(D.C,dims), Tuple(D.m[i] for i in dims))