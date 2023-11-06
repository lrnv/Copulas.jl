"""
    SklarDist{CT,TplMargins} 

Fields:
  - C::CT - The copula
  - m::TplMargins - a Tuple representing the marginal distributions

Constructor

    SklarDist(C,m)

This function allows to construct a random vector specified, through the Sklar Theorem, by its marginals and its copula separately. See [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem). 

The obtain random vector follows `Distributions.jl`'s API and can be sampled, pdf and cdf can be evaluated, etc... We even provide a fit function. See the folowing exemple code : 

```julia
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

# This generates a (3,1000)-sized dataset from the multivariate distribution D
simu = rand(D,1000)

# While the following estimates the parameters of the model from a dataset: 
D̂ = fit(SklarDist{FrankCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
# Increase the number of observations to get a beter fit (or not?)  
```


"""
struct SklarDist{CT,TplMargins} <: Distributions.ContinuousMultivariateDistribution
    C::CT
    m::TplMargins
    function SklarDist(C,m)
        d = length(C)
        @assert length(m) == d
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
    sum(Distributions.logpdf(S.m[i],u[i]) for i in 1:length(u)) + Distributions.logpdf(S.C,Distributions.cdf.(S.m,u))
end


function Distributions.fit(::Type{SklarDist{CT,TplMargins}},x) where {CT,TplMargins}
    # The first thing to do is to fit the marginals : 
    @assert length(TplMargins.parameters) == size(x,1)
    m = Tuple(Distributions.fit(TplMargins.parameters[i],x[i,:]) for i in 1:size(x,1))
    u = pseudos(x)
    C = Distributions.fit(CT,u)
    return SklarDist(C,m)
end