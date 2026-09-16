"""
    IndependentCopula(d)
    IndependentCopula{d}()

The independent copula in dimension ``d`` has distribution function

```math
C(\\mathbf{x}) = \\prod_{i=1}^{d} x_i.
```

It is Archimedean with generator ``\\psi(s) = e^{-s}``.

The model is parameter free and available in every public copula dimension
`d ≥ 2`. Its density is one throughout the unit hypercube, sampling produces
independent uniform coordinates, and both Rosenblatt transforms are the
identity. It is also the zero-dependence limit of many parametric families;
constructing the named family at such a limit need not return this concrete
type.

# Example
```julia
using Copulas, Distributions

C = IndependentCopula(3)
cdf(C, [0.2, 0.5, 0.8]) == 0.2 * 0.5 * 0.8
```

See also: [`Copula`](@ref), [`IndependentGenerator`](@ref),
[`ArchimedeanCopula`](@ref).

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct IndependentCopula{d} <: Copula{d}
    function IndependentCopula{d}() where {d}
        d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))
        return new{d}()
    end
end
IndependentCopula(d) = IndependentCopula{d}()
_cdf(::IndependentCopula{d}, u) where d = prod(u)
Distributions._logpdf(::IndependentCopula{d}, u) where {d} = zero(eltype(u))

function Distributions._rand!(rng::Distributions.AbstractRNG, ::IndependentCopula{d}, A::AbstractMatrix{T}) where {T<:Real, d}
    size(A, 1) == d || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    Random.rand!(rng,A)
    return A
end

rosenblatt(::IndependentCopula{d}, u::AbstractMatrix{<:Real}) where {d} = u
inverse_rosenblatt(::IndependentCopula{d}, u::AbstractMatrix{<:Real}) where {d} = u

τ(::IndependentCopula) = 0
ρ(::IndependentCopula) = 0
γ(::IndependentCopula) = 0
ι(::IndependentCopula) = 0

StatsBase.corkendall(::IndependentCopula{d}) where d = one(zeros(d,d))
StatsBase.corspearman(::IndependentCopula{d}) where d = one(zeros(d,d))

# Conditioning colocated
distortion(::IndependentCopula, ::NTuple{p,Int}, ::NTuple{p,<:Real}, ::Int) where {p} = NoDistortion()
conditional_copula(::IndependentCopula{D}, js, u) where D = IndependentCopula{D - length(js)}()
function condition(::IndependentCopula{D}, js::NTuple{p,Int}, uⱼₛ::NTuple{p,<:Real}) where {D,p}
    d = D - length(js)
    return d==1 ? Distributions.Uniform() : IndependentCopula{D - p}()
end

# Subsetting colocated
SubsetCopula(::IndependentCopula{d}, ::NTuple{p, Int}) where {d, p} =
    p == 1 ? Distributions.Uniform() : IndependentCopula{p}()

# Fitting/params interface (no parameters)
Distributions.params(::IndependentCopula) = (;)
# A parameter-free family has nothing to weight, so the weights are accepted and unused.
_fit(::Type{<:IndependentCopula}, U, ::Val{:mle}; weights=nothing) = IndependentCopula(size(U,1))
_fit(::Type{<:IndependentCopula}, U, ::Val{:itau}) = IndependentCopula(size(U,1))
_fit(::Type{<:IndependentCopula}, U, ::Val{:irho}) = IndependentCopula(size(U,1))
_fit(::Type{<:IndependentCopula}, U, ::Val{:ibeta}) = IndependentCopula(size(U,1))
