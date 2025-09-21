"""
    IndependentCopula(d)

The independent copula in dimension ``d`` has distribution function

```math
C(\\mathbf{x}) = \\prod_{i=1}^{d} x_i.
```

It is Archimedean with generator ``\\psi(s) = e^{-s}``.

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct IndependentCopula{d} <: Copula{d} 
    IndependentCopula(d) = new{d}()
end

# Fitting/params interface (no parameters)
Distributions.params(::IndependentCopula) = ()
_example(::Type{<:IndependentCopula}, d) = IndependentCopula(d)
_unbound_params(::Type{<:IndependentCopula}, d, θ) = Float64[]
_rebound_params(::Type{<:IndependentCopula}, d, α) = (;)
_fit(::Type{<:IndependentCopula}, U, ::Any) = IndependentCopula(size(U,1)), (;)

_cdf(::IndependentCopula{d}, u) where d = prod(u)
Distributions._logpdf(::IndependentCopula{d}, u) where {d} = all(0 .<= u .<= 1) ? zero(eltype(u)) : eltype(u)(-Inf)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::IndependentCopula{d}, x::AbstractVector{T}) where {d,T<:Real}
    Random.rand!(rng,x)
end
function Distributions._rand!(rng::Distributions.AbstractRNG, ::IndependentCopula{d}, A::DenseMatrix{T}) where {T<:Real, d}
    Random.rand!(rng,A)
    return A
end
rosenblatt(::IndependentCopula{d}, u::AbstractMatrix{<:Real}) where {d} = u
inverse_rosenblatt(::IndependentCopula{d}, u::AbstractMatrix{<:Real}) where {d} = u
τ(::IndependentCopula) = 0
ρ(::IndependentCopula) = 0
StatsBase.corkendall(::IndependentCopula{d}) where d = one(zeros(d,d))
StatsBase.corspearman(::IndependentCopula{d}) where d = one(zeros(d,d))

# Conditioning colocated
DistortionFromCop(::IndependentCopula, ::NTuple{p,Int}, ::NTuple{p,Float64}, ::Int) where {p} = NoDistortion()
ConditionalCopula(::IndependentCopula{D}, js, u) where D = IndependentCopula(D - length(js))
function condition(::IndependentCopula{D}, js::NTuple{p, Int}, uⱼₛ::NTuple{p, Float64}) where {D, p}
    d = D - length(js)
    return d==1 ? Distributions.Uniform() : IndependentCopula(d)
end

# Subsetting colocated
SubsetCopula(::IndependentCopula{d}, ::NTuple{p, Int}) where {d, p} = IndependentCopula(p)