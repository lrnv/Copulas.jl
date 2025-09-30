"""
    MCopula(d)

The upper Fréchet–Hoeffding bound is the copula with the largest value among all copulas; it corresponds to comonotone random vectors. For any copula ``C`` and all ``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u}).
```

Both Fréchet–Hoeffding bounds are Archimedean copulas.

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct MCopula{d} <: Copula{d} 
    MCopula(d) = new{d}()
end
Distributions.params(::MCopula) = ()
_example(::Type{<:MCopula}, d) = MCopula(d)
_unbound_params(::Type{<:MCopula}, d, θ) = Float64[]
_rebound_params(::Type{<:MCopula}, d, α) = (;)
_fit(::Type{<:MCopula}, U, ::Union{Val{:mle}, Val{:itau}, Val{:irho}, Val{:ibeta}}) = MCopula(size(U,1)), (;)

Distributions._logpdf(::MCopula{d}, u) where {d} = all(u == u[1]) ? zero(eltype(u)) : eltype(u)(-Inf)
_cdf(::MCopula{d}, u) where {d} = Base.minimum(u)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::MCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    x .= rand(rng)
end
τ(::MCopula) = 1
ρ(::MCopula) = 1
StatsBase.corkendall(::MCopula{d}) where d = ones(d,d)
StatsBase.corspearman(::MCopula{d}) where d = ones(d,d)

# Subsetting colocated
SubsetCopula(::MCopula{d}, ::NTuple{p, Int}) where {d,p} = MCopula(p)
DistortionFromCop(::MCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, i::Int) = MDistortion(float(uⱼₛ[1]), Int8(js[1]))