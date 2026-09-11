"""
    WCopula{2}()
    WCopula(2)

The lower Fréchet–Hoeffding bound is the smallest bivariate copula and
represents complete negative dependence. For any copula ``C`` and all
``\\mathbf{u} \\in [0,1]^2``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u}).
```

`WCopula` is parameter free and exists only in dimension two: the analogous
formula is merely a pointwise lower bound, not a copula, in higher dimensions.
A sample has the form `(U, 1-U)`, so the law is singular on the anti-diagonal
and has no ordinary Lebesgue density. `pdf` and `logpdf` use Copulas.jl's
generalized-density convention and must not be integrated as a Lebesgue
density.

# Example
```julia
using Copulas, Distributions

C = WCopula()
isapprox(cdf(C, [0.7, 0.6]), 0.3)
```

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct WCopula{d} <: Copula{d}
    function WCopula{d}() where {d}
        d == 2 || error("WCopula only available in dimension 2")
        return new{2}()
    end
end
copula_measure_style(::Type{<:WCopula}) = NonAbsolutelyContinuousMeasure()
WCopula() = WCopula{2}()
WCopula(d) = WCopula{d}()
Distributions._logpdf(::WCopula,           u) = sum(u) == 1 ? zero(eltype(u)) : eltype(u)(-Inf)
_cdf(::WCopula, u) = max(sum(u)-1,0)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::WCopula, A::AbstractMatrix{T}) where {T<:Real}
    size(A, 1) == 2 || throw(ArgumentError("Dimension mismatch between copula and output matrix"))
    Random.rand!(rng, view(A, 1, :))
    @inbounds for col in axes(A, 2)
        A[2, col] = one(T) - A[1, col]
    end
    return A
end
τ(::WCopula) = -1
ρ(::WCopula) = -1
StatsBase.corkendall(::WCopula) = [1 -1; -1 1]
StatsBase.corspearman(::WCopula) = [1 -1; -1 1]

# Subsetting colocated
SubsetCopula(C::WCopula, ::NTuple{p, Int}) where {p} = (p==2 ? C : error("WCopula only defined for p=2"))
distortion(::WCopula, js::Tuple{Int}, uⱼₛ::Tuple{Float64}, i::Int) = WDistortion(float(uⱼₛ[1]), Int8(js[1]))

# Fitting/params interface (no parameters)
Distributions.params(::WCopula) = (;)
_fit(::Type{<:WCopula}, U, ::Val{:mle}) = WCopula(size(U,1)), (;)
_fit(::Type{<:WCopula}, U, ::Val{:itau}) = WCopula(size(U,1)), (;)
_fit(::Type{<:WCopula}, U, ::Val{:irho}) = WCopula(size(U,1)), (;)
_fit(::Type{<:WCopula}, U, ::Val{:ibeta}) = WCopula(size(U,1)), (;)
