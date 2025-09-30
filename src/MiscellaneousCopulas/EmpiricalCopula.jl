"""
    EmpiricalCopula{d, MT}

Fields:
- `u::MT` — pseudo-observation matrix of size `(d, N)`.

Constructor

    EmpiricalCopula(u; pseudo_values=true)

The empirical copula in dimension ``d`` is defined from a matrix of pseudo-observations
``\\mathbf u = (u_{i,j})_{1\\le i \\le d,\\ 1\\le j \\le N}`` with entries in ``[0,1]``.
Its distribution function is

```math
C(\\mathbf{x}) = \\frac{1}{N} \\sum_{j=1}^{N} \\mathbf{1}_{\\{ \\mathbf{u}_{\\cdot,j} \\le \\mathbf{x} \\}} ,
```

where the inequality is componentwise. If `pseudo_values=false`, the constructor first ranks the raw data into pseudo-observations; otherwise it assumes `u` already contains pseudo-observations in ``[0,1]``.

Notes:
- This is an empirical object based on pseudo-observations; it is not necessarily a true copula for finite ``N`` but is widely used for nonparametric inference.
- Supports `cdf`, `logpdf` at observed points, random sampling, and subsetting.

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
function _cdf(C::EmpiricalCopula{d,MT},u) where {d,MT}
   return sum(all(C.u .<= u,dims=1))/size(C.u,2) # might not be very efficient implementation. 
end
function Distributions._logpdf(C::EmpiricalCopula{d,MT}, u) where {d,MT}
    any(C.u .== u) ? -log(size(C.u,2)) : -Inf
end
function Distributions._rand!(rng::Distributions.AbstractRNG, C::EmpiricalCopula{d,MT}, x::AbstractVector{T}) where {d,MT,T<:Real}
    x .= C.u[:,Distributions.rand(rng,axes(C.u,2),1)[1]]
end
StatsBase.corkendall(C::EmpiricalCopula) = StatsBase.corkendall(C.u')

# Subsetting colocated
function SubsetCopula(C::EmpiricalCopula{d,MT}, dims::NTuple{p, Int}) where {d,MT,p}
    return EmpiricalCopula(C.u[collect(dims), :]; pseudo_values=true)
end

# Fitting colocated. 
StatsBase.dof(::Copulas.EmpiricalCopula)    = 0
_default_method(::Type{<:EmpiricalCopula}) = :deheuvels
function _fit(::Type{<:EmpiricalCopula}, U, ::Val{:deheuvels}; pseudo_values = true, kwargs...)
    C = EmpiricalCopula(U; pseudo_values=pseudo_values, kwargs...)
    return C, (; pseudo_values)
end