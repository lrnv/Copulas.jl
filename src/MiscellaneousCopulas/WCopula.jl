"""
    WCopula

The [Lower Frechet-Hoeffding bound](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds) is the copula with the lowest value among all copulas. Note that ``W`` is only a proper copula when ``d=2``, in greater dimensions it is still the (pointwise) lower bound, but not a copula anymore. For any copula ``C``, if ``W`` and ``M`` are (respectively) the lower and uppder Frechet-Hoeffding bounds, we have that for all ``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u})
```

The two Frechet-Hoeffding bounds are also Archimedean copulas.

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct WCopula <: Copula{2}
    WCopula(d) = d!=2 ? error("WCopula only available in dimension 2") : new()
    WCopula() = new()
end
Distributions._logpdf(::WCopula,           u) = sum(u) == 1 ? zero(eltype(u)) : eltype(u)(-Inf)
_cdf(::WCopula, u) = max(sum(u)-1,0)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::WCopula, x::AbstractVector{T}) where {T<:Real}
    @assert length(x)==2
    x[1] = rand(rng)
    x[2] = 1-x[1]
    return x
end
τ(::WCopula) = -1
ρ(::WCopula) = -1
StatsBase.corkendall(::WCopula) = [1 -1; -1 1]
StatsBase.corspearman(::WCopula) = [1 -1; -1 1]

# Subsetting colocated
SubsetCopula(::WCopula, ::NTuple{p, Int}) where {p} = (p==2 ? C : error("WCopula only defined for p=2"))
DistortionFromCop(::WCopula, js::Tuple{Int}, uⱼₛ::Tuple{Float64}, i::Int) = WDistortion(float(uⱼₛ[1]), Int8(js[1]))