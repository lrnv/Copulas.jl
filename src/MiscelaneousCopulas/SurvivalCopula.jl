struct SurvivalCopula{d,CT,VI} <: Copula{d}
    C::CT
    indices::VI
    function SurvivalCopula(C,indices)
        if length(indices) == 0
            return C
        end
        d = length(C)
        @assert all(indices <= d)
        return new{d,eltype(C),eltype(indices)}(C,indices)
    end
end
function reverse!(u,idx)
    u[idx] .= 1 - u[idx]
end
function reverse(u,idx)
    reverse!(u,idx)
    return u
end
Distributions.cdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI} = Distributions.cdf(C.C,reverse(u,C.indices))
Distributions._logpdf(C::SurvivalCopula{d,CT,VI},u) where {d,CT,VI} = Distributions._logpdf(C.C,reverse(u,C.indices))
function Distributions._rand!(rng::Distributions.AbstractRNG, ::SurvivalCopula{d,CT,VI}, x::AbstractVector{T}) where {d,CT,VI}
    Distributions._rand!(rng,C.C,x)
    reverse!(x,C.indices)
end
function Base.rand(rng::Distributions.AbstractRNG,C::SurvivalCopula{d,CT,VI}) where {d,CT,VI}
    x = Base.rand(rnd,C.C)
    reverse!(x,C.indices)
    return x
end