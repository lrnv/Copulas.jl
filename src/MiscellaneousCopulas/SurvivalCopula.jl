struct SurvivalCopula{d,CT,VI} <: Copula{d}
    C::CT
    indices::VI
    function SurvivalCopula(C,indices)
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
function reverse(u,idx)
    v = deepcopy(u)
    reverse!(v,idx)
    return v
end
Distributions.cdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI} = Distributions.cdf(C.C,reverse(u,C.indices))
Distributions._logpdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI} = Distributions._logpdf(C.C,reverse(u,C.indices))
function Distributions._rand!(rng::Distributions.AbstractRNG, C::SurvivalCopula{d,CT,VI}, x::AbstractVector{T}) where {d,CT,VI,T}
    Distributions._rand!(rng,C.C,x)
    reverse!(x,C.indices)
end
function Distributions.fit(T::Type{CT},u) where {CT <: SurvivalCopula}
    # d = size(u,1)
    d,subCT,indices = T.parameters
    subfit = Distributions.fit(subCT,reverse(u,indices))
    return SurvivalCopula(subfit,indices)
end