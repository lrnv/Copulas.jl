"""
    SklarDist{CT,TplMargins} 

Fields:
  - `C::CT` - The copula
  - `m::TplMargins` - a Tuple representing the marginal distributions

Constructor

    SklarDist(C,m)

This function allows to construct a random vector specified, through the Sklar Theorem, by its marginals and its copula separately. See [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem):

!!! theorem "Theorem (Sklar 1959):"
    For every random vector ``\\bm X``, there exists a copula ``C`` such that 

    ``\\forall \\bm x\\in \\mathbb R^d, F(\\bm x) = C(F_{1}(x_{1}),...,F_{d}(x_{d})).``
    The copula ``C`` is uniquely determined on ``\\mathrm{Ran}(F_{1}) \\times ... \\times \\mathrm{Ran}(F_{d})``, where ``\\mathrm{Ran}(F_i)`` denotes the range of the function ``F_i``. In particular, if all marginals are absolutely continuous, ``C`` is unique.


The obtain random vector follows `Distributions.jl`'s API and can be sampled, pdf and cdf can be evaluated, etc... We even provide a fit function. See the folowing exemple code : 

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
function Distributions.fit(::Type{SklarDist{CT,TplMargins}},x) where {CT,TplMargins}
    # The first thing to do is to fit the marginals : 
    @assert length(TplMargins.parameters) == size(x,1)
    m = Tuple(Distributions.fit(TplMargins.parameters[i],x[i,:]) for i in axes(x,1))
    u = pseudos(x)
    C = Distributions.fit(CT,u)
    return SklarDist(C,m)
end