"""
    EmpiricalCopula(u; pseudo_values=true)

Nonparametric estimator of an underlying dependence structure; used as a building
block for rank‑based inference, bootstrap procedures, and goodness‑of‑fit tests.


Parameters
* `u::AbstractMatrix` – data matrix of shape `(d, N)`.
* `pseudo_values=true` – if `false`, raw data in `u` are converted to pseudo‑observations
    via ranks divided by `N+1`; if `true` the entries of `u` are assumed already in `[0,1]`.

Given pseudo‑observations `(u_{i,j})_{1\\le i\\le d,\\ 1\\le j\\le N}` with each
column in `[0,1]^d`, the empirical copula distribution function is
```math
C(\\boldsymbol x) = \\frac{1}{N} \\sum_{j=1}^{N} \\mathbf{1}_{\\{ u_{1,j} \\le x_1,\\ldots,u_{d,j} \\le x_d \\}}.
```


Notes
* For finite `N` the function above is grounded and has uniform margins but may
    fail some higher‑order smoothness properties; asymptotically it converges to the
    true copula under standard i.i.d. assumptions.
* `pdf`/`logpdf`: the mass function assigns probability `1/N` to each observed
    point; off‑sample points have zero density.
* `rand`: resamples observed pseudo‑observations with replacement.
* Subsetting returns the empirical copula of the projected pseudo‑observations.

See also: [`pseudos`](@ref), [`BetaCopula`](@ref), [`CheckerboardCopula`](@ref).

References
* [nelsen2006](@cite) Nelsen (2006), An Introduction to Copulas.
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
function Distributions.fit(::Type{CT},u) where {CT <: EmpiricalCopula}
    return EmpiricalCopula(u)
end
StatsBase.corkendall(C::EmpiricalCopula) = StatsBase.corkendall(C.u')

# Subsetting colocated
function SubsetCopula(C::EmpiricalCopula{d,MT}, dims::NTuple{p, Int}) where {d,MT,p}
    return EmpiricalCopula(C.u[collect(dims), :]; pseudo_values=true)
end
