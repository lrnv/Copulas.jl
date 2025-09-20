"""
    SklarDist{CT,TplMargins} 

Fields:
  - `C::CT` - The copula
  - `m::TplMargins` - a Tuple representing the marginal distributions

Constructor

    SklarDist(C,m)

Construct a joint distribution via Sklar's theorem from marginals and a copula. See [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem):

!!! theorem "Theorem (Sklar 1959):"
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
Base.length(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = length(S.C)
Base.eltype(S::SklarDist{CT,TplMargins}) where {CT,TplMargins} = Base.eltype(S.C)
  
function Distributions.cdf(S::SklarDist{CT,TplMargins},x) where {CT,TplMargins}
    return Distributions.cdf(S.C,Distributions.cdf.(S.m,x))
end
function Distributions._rand!(rng::Distributions.AbstractRNG, S::SklarDist{CT,TplMargins}, x::AbstractVector{T}) where {CT,TplMargins,T}
    Random.rand!(rng,S.C,x)
     x .= Distributions.quantile.(S.m,x)
end
function Distributions._logpdf(S::SklarDist{CT,TplMargins},u) where {CT,TplMargins}
    sum(Distributions.logpdf(S.m[i],u[i]) for i in eachindex(u)) + Distributions.logpdf(S.C,clamp.(Distributions.cdf.(S.m,u),0,1))
end
function Distributions.loglikelihood(S::SklarDist, X::AbstractMatrix)
    tot = 0.0
    @inbounds for j in axes(X,2)
        tot += Distributions.logpdf(S, @view X[:,j])
    end
    return tot
end
# total dof of the joint model = dof(copula) + sum dof(margins)
StatsBase.dof(S::SklarDist) = StatsBase.dof(S.C) + sum(_dof_marg, S.m)

_dof_marg(d) = hasmethod(StatsBase.dof, Tuple{typeof(d)}) ? StatsBase.dof(d) : length(Distributions.params(d))  # fallback si el marginal no tiene dof

# I am not sure this is really needed, maybe this is done by itself by Distributions.jl
Distributions.loglikelihood(S::SklarDist, X::AbstractMatrix{<:Real}) = sum(Base.Fix1(Distributions.logpdf, S), eachcol(X))