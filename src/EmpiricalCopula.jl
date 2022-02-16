struct EmpiricalCopula{d,MT} <: Copula{d}
    u::MT
end
Base.eltype(C::EmpiricalCopula{d,MT}) where {d,MT} = Base.eltype(C.u)
function EmpiricalCopula(u;pseudos=true)
    d = size(u,1)
    if !pseudos
        u = pseudos(u)
    end
    return EmpiricalCopula{d,typeof(u)}(u)
end
function Distributions.cdf(C::EmpiricalCopula{d,MT},u) where {d,MT}
   return mean(all(C.u .<= u,dims=1)) # might not be very efficient implementation. 
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::EmpiricalCopula{d,MT}, x::AbstractVector{T}) where {d,MT,T<:Real}
    x .= C.u[:,Base.rand(rng,axes(C.u,2),1)[1]]
end
function Base.rand(rng::Distributions.AbstractRNG,C::EmpiricalCopula{d,MT}) where {d,MT}
    C.u[:,Base.rand(rng,axes(C.u,2),1)[1]]
end