# Bayesian inference with Turing.jl


The compatibility with the broader `Distributions.jl`'s API allows a lot of interactions with the broader eco-system. One of the firsts examples that we discovered was the possibility to do bayesian inference of model parameters with [`Turing.jl`](https://turing.ml/). 

Consider that we have a model with :

- A given parametric dependence structure, say a Clayton Copula. 
- A few given parametric marginals. 

Then we can use Turing's `@addlogprob!` to compute the loglikelyhood of our model and maximize it around th eparameters alongside the chain as folows: 

```@example
using Copulas
using Distributions
using Random
using Turing
using StatsPlots

Random.seed!(123)
true_θ = 7
true_θ₁ = 1
true_θ₂ = 3
true_θ₃ = 2
D = SklarDist(ClaytonCopula(3,true_θ), (Exponential(true_θ₁), Pareto(true_θ₂), Exponential(true_θ₃)))
draws = rand(D, 200)

@model function copula(X)
    # Priors
    θ  ~ TruncatedNormal(1.0, 1.0, -1/3, Inf)
    θ₁ ~ TruncatedNormal(1.0, 1.0, 0, Inf)
    θ₂ ~ TruncatedNormal(1.0, 1.0, 0, Inf)
    θ₃ ~ TruncatedNormal(1.0, 1.0, 0, Inf)

    # Build the parametric model
    C = ClaytonCopula(3,θ)
    X₁ = Exponential(θ₁)
    X₂ = Pareto(θ₂)
    X₃ = Exponential(θ₃)
    D = SklarDist(C, (X₁, X₂, X₃))

    # Compute the final loglikelyhood
    Turing.Turing.@addlogprob! loglikelihood(D, X)
end

sampler = NUTS() # MH() works too
chain = sample(copula(draws), sampler, MCMCThreads(), 100, 4)
plot(chain)
```

This kind of model could be used to do a lot of stuff. here we use it for a parametric estimation of a sampling model, but it could be used for other things such as, e.g., a bayesian regression with an error structure given by a parmaetric copula.

Note that we truncated the θ parameter at -1/3 and not 0 as the ClaytonCopula can handle negative dependence structures. 




