"""
    EmpiricalCopula{d,MT} 

Fields:
  - u::MT - the matrix of observations. 

Constructor

    EmpiricalCopula(u;pseudos=true)

The EmpiricalCopula in dimension ``d`` is parameterized by a pseudo-data matrix wich should have shape (d,N). Its expression is given as :  

```math
C(\\mathbf x) = \\frac{1}{N}\\sum_{i=1}^n \\mathbf 1_{\\mathbf u_i \\le \\mathbf x}
```

This function is very practical, be be aware that this is not a true copula (since ``\\mathbf u`` are only pseudo-observations). The constructor allows you to pass dirctly pseudo-observations (the default) or will compute them for you. You can then compute the `cdf` of the copula, and sample it through the standard interface.
"""
struct EmpiricalCopula{d,MT} <: Copula{d}
    u::MT
end
Base.eltype(C::EmpiricalCopula{d,MT}) where {d,MT} = Base.eltype(C.u)
function EmpiricalCopula(u;pseudos=true)
    d = size(u,1)
    if !pseudos
        u = pseudos(u)
    else
        @assert all(0 .<= u .<= 1)
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