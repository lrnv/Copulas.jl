"""
    WCopula

The lower Fréchet–Hoeffding bound is the copula with the smallest value among all copulas. Note that ``W`` is a proper copula only when ``d = 2``; for ``d > 2`` it remains the pointwise lower bound but is not itself a copula. For any copula ``C`` and all ``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u}).
```

Both Fréchet–Hoeffding bounds are Archimedean copulas.

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
SubsetCopula(C::WCopula, ::NTuple{p, Int}) where {p} = (p==2 ? C : error("WCopula only defined for p=2"))
DistortionFromCop(::WCopula, js::Tuple{Int}, uⱼₛ::Tuple{Float64}, i::Int) = WDistortion(float(uⱼₛ[1]), Int8(js[1]))

# Fitting/params interface (no parameters)
Distributions.params(::WCopula) = (;)
_fit(::Type{<:WCopula}, U, ::Val{:mle}) = WCopula(size(U,1)), (;)
_fit(::Type{<:WCopula}, U, ::Val{:itau}) = WCopula(size(U,1)), (;)
_fit(::Type{<:WCopula}, U, ::Val{:irho}) = WCopula(size(U,1)), (;)
_fit(::Type{<:WCopula}, U, ::Val{:ibeta}) = WCopula(size(U,1)), (;)