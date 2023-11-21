```@meta
CurrentModule = Copulas
```


## Installation

The package is available on Julia's general registry, and can be installed as follows: 

```julia
] add Copulas
```

## First example

The API contains random number generation, cdf and pdf evaluation, and the `fit` function from `Distributions.jl`. A typical use case might look like this: 

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

There are a lot of available copula families in the package, that can be regrouped into a few classes: 
- `EllipticalCopulas`: `GaussianCopula` and `TCopula`
- `ArchimedeanCopula`: The generic version (for user-specified generator), but also `ClaytonCopula`,`FrankCopula`, `AMHCopula`, `JoeCopula`, `GumbelCopula`,`GumbelBarnettCopula`,`InvGaussianCopula`, supporting the full ranges in every dimension (e.g., `ClaytonCopula` can be sampled with negative dependence in any dimension, not just d=2). 
- `WCopula`, `IndependentCopula` and `MCopula`, which are [Fréchet-Hoeffding bounds](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds),
- Others unclassable : `PlackettCopula`, `EmpiricalCopula` to follow closely a given dataset, etc..

