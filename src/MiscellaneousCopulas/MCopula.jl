"""
    MCopula(d)
    MCopula{d}()

The upper Fréchet–Hoeffding bound is the pointwise largest copula and represents
complete positive dependence. For any copula ``C`` and all
``\\mathbf{u} \\in [0,1]^d``,

```math
W(\\mathbf{u}) \\le C(\\mathbf{u}) \\le M(\\mathbf{u}).
```

`MCopula` is parameter free and valid in every public copula dimension `d ≥ 2`.
A sample repeats one uniform variate across all coordinates, hence the
distribution is concentrated on the main diagonal and has no ordinary Lebesgue
density. `pdf` and `logpdf` follow Copulas.jl's generalized-density convention
for singular copulas and must not be integrated against Lebesgue measure as if
they were a density.

# Example
```julia
using Copulas, Distributions

C = MCopula(3)
cdf(C, [0.2, 0.5, 0.8]) == 0.2
```

See also: [`Copula`](@ref), [`WCopula`](@ref), [`measure`](@ref).

References:
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct MCopula{d} <: Copula{d}
    function MCopula{d}() where {d}
        d >= 2 || throw(ArgumentError("a public copula requires dimension d ≥ 2; got d=$d"))
        return new{d}()
    end
end
copula_measure_style(::Type{<:MCopula}) = NonAbsolutelyContinuousMeasure()
MCopula(d) = MCopula{d}()
Distributions._logpdf(::MCopula{d}, u) where {d} = all(u == u[1]) ? zero(eltype(u)) : eltype(u)(-Inf)
_cdf(::MCopula{d}, u) where {d} = Base.minimum(u)

function Distributions._rand!(rng::Distributions.AbstractRNG, ::MCopula{d}, A::AbstractMatrix{T}) where {d,T<:Real}
    size(A, 1) == d || throw(DimensionMismatch(
        "output matrix has $(size(A, 1)) rows; expected copula dimension $d",
    ))
    Random.rand!(rng, view(A, 1, :))
    @inbounds for row in 2:d
        A[row, :] .= view(A, 1, :)
    end
    return A
end
τ(::MCopula) = 1
ρ(::MCopula) = 1
γ(::MCopula) = 1
ι(::MCopula) = -Inf
StatsBase.corkendall(::MCopula{d}) where d = ones(d,d)
StatsBase.corspearman(::MCopula{d}) where d = ones(d,d)

# Subsetting colocated
SubsetCopula(::MCopula{d}, ::NTuple{p, Int}) where {d,p} =
    p == 1 ? Distributions.Uniform() : MCopula{p}()
distortion(::MCopula{2}, js::NTuple{1,Int}, uⱼₛ::NTuple{1,Float64}, i::Int) = MDistortion(float(uⱼₛ[1]), Int8(js[1]))

Paramorph.param_space(::Type{<:MCopula}, d) = ()

# Fitting/params interface (no parameters)
# A parameter-free family has nothing to weight, so the weights are accepted and unused.
_fit(
    ::Type{<:MCopula}, U, ::Val{d},
    ::Union{Val{:mle},Val{:itau},Val{:irho},Val{:ibeta}}; weights=nothing,
) where {d} = MCopula{d}()
