
###############################################################################
#####  SklarDist framework.
#####  User-facing function: `SklarDist(C::Copula{d}, m::NTuple{d, <:UnivariateDistribution}) where d`
#####
#####  Nothing here should be overwritten when defining new copulas. 
###############################################################################

"""
    SklarDist{CT,TplMargins} 

Fields:
  - `C::CT` - The copula
  - `m::TplMargins` - a Tuple representing the marginal distributions

Constructor

    SklarDist(C,m)

Construct a joint distribution via Sklar's theorem from marginals and a copula. See [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem):

!!! theorem "Sklar 1959"
    For every random vector ``\\boldsymbol X``, there exists a copula ``C`` such that 

    ``\\forall \\boldsymbol x\\in \\mathbb R^d, F(\\boldsymbol x) = C(F_{1}(x_{1}),...,F_{d}(x_{d})).``
    The copula ``C`` is uniquely determined on ``\\mathrm{Ran}(F_{1}) \\times ... \\times \\mathrm{Ran}(F_{d})``, where ``\\mathrm{Ran}(F_i)`` denotes the range of the function ``F_i``. In particular, if all marginals are absolutely continuous, ``C`` is unique.


The resulting random vector follows the `Distributions.jl` API (rand/cdf/pdf/logpdf). A `fit` method is also provided. Example:

```julia
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # Generate a dataset

# You may estimate a copula using the `fit` function:
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
```

References: 
* [sklar1959](@cite) Sklar, M. (1959). Fonctions de répartition à n dimensions et leurs marges. In Annales de l'ISUP (Vol. 8, No. 3, pp. 229-231).
* [nelsen2006](@cite) Nelsen, Roger B. An introduction to copulas. Springer, 2006.
"""
struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C::Copula{d}, m::NTuple{d, Any}) where d
        @assert all(mᵢ isa Distributions.UnivariateDistribution for mᵢ in m)
        return new{typeof(C),typeof(m)}(C,m)
    end    
end
SklarDist(C, m) = SklarDist(C, Tuple(m))
Base.length(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = length(S.C)
Base.eltype(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = Base.eltype(S.C)
@inline function _sklar_work_eltype(S::SklarDist, x)
    T = promote_type(eltype(S.C), eltype(x))
    for margin in S.m
        T = promote_type(T, eltype(margin))
    end
    return T
end
function Distributions.cdf(S::SklarDist{CT,TplMargins}, x) where {CT,TplMargins}
    d = length(S)
    T = _sklar_work_eltype(S, x)
    u = Vector{T}(undef, d)
    @inbounds for i in 1:d
        u[i] = Distributions.cdf(S.m[i], x[i])
    end
    return Distributions.cdf(S.C, u)
end
Distributions.logcdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins} = log(Distributions.cdf(S, x))
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, A::AbstractMatrix{T}) where {CT,TplMargins,T}
    size(A, 1) == length(S) || throw(ArgumentError("Dimension mismatch between distribution and output matrix"))
    Random.rand!(rng, S.C, A)
    lo, hi = nextfloat(T(0)), prevfloat(T(1))
    @inbounds for col in axes(A, 2), row in axes(A, 1)
        A[row, col] = Distributions.quantile(S.m[row], clamp(A[row, col], lo, hi))
    end
    return A
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist, x::AbstractVector{T}) where {T<:Real}
    Distributions._rand!(rng, S, reshape(x, length(S), 1))
    return x
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins}, u) where {CT,TplMargins}
    d = length(S)
    T = _sklar_work_eltype(S, u)
    # sum marginal logpdfs without generator comprehensions
    s = zero(T)
    @inbounds for i in 1:d
        s += Distributions.logpdf(S.m[i], u[i])
    end
    # compute cdf of marginals, clamped, without broadcasting temporaries
    U = Vector{T}(undef, d)
    @inbounds for i in 1:d
        U[i] = clamp(Distributions.cdf(S.m[i], u[i]), zero(T), one(T))
    end
    return s + Distributions.logpdf(S.C, U)
end
function StatsBase.dof(S::SklarDist)
    a = StatsBase.dof(S.C)
    b = sum(hasmethod(StatsBase.dof, Tuple{typeof(d)}) ? StatsBase.dof(d) : length(Distributions.params(d)) for d in S.m)
    return a+b
end
