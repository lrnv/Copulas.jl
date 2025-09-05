"""
    EmpiricalCopula{d,MT} 

Fields:
* u::MT - the matrix of observations. 

Constructor

    EmpiricalCopula(u;pseudos=true)

The EmpiricalCopula in dimension ``d`` is parameterized by a pseudo-data matrix which should have shape (d,N). Its expression is given as :  

```math
C(\\mathbf x) = \\frac{1}{N}\\sum_{i=1}^n \\mathbf 1_{\\mathbf u_i \\le \\mathbf x}
```

This function is very practical, be be aware that this is not a true copula (since ``\\mathbf u`` are only pseudo-observations). The constructor allows you to pass dirctly pseudo-observations (the default) or will compute them for you. You can then compute the `cdf` of the copula, and sample it through the standard interface.

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct EmpiricalCopula{d,MT} <: Copula{d}
    u::MT
end
Base.eltype(C::EmpiricalCopula{d,MT}) where {d,MT} = Base.eltype(C.u)
function EmpiricalCopula(u;pseudo_values=true)
    T = promote_type(eltype(u), Float64)
    u = T.(u)
    d = size(u,1)
    if !pseudo_values
        u = pseudos(u)
    else
        @assert all(0 .<= u .<= 1)
    end
    return EmpiricalCopula{d,typeof(u)}(u)
end
function Base.show(io::IO, C::EmpiricalCopula)
    print(io, "EmpiricalCopula{d}$(size(C.u))")
end
function _cdf(C::EmpiricalCopula{d,MT},u) where {d,MT}
   return sum(all(C.u .<= u,dims=1))/size(C.u,2) # might not be very efficient implementation. 
end
function Distributions._logpdf(C::EmpiricalCopula{d,MT}, u) where {d,MT}
    any(C.u .== u) ? -log(size(C.u,2)) : -Inf
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::EmpiricalCopula{d,MT}, x::AbstractVector{T}) where {d,MT,T<:Real}
    x .= C.u[:,Distributions.rand(rng,axes(C.u,2),1)[1]]
end
function Distributions.fit(::Type{CT},u) where {CT <: EmpiricalCopula}
    return EmpiricalCopula(u)
end
StatsBase.corkendall(C::EmpiricalCopula) = StatsBase.corkendall(C.u')
