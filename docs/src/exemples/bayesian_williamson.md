# Bayesian estimation of non-parametric WilliamsonCopula

The `WilliamsonCopula` allows to easily fit the williamson distribution of the model through Turing.jl. Take for exemple a look at the following code : 

[not working yet]

```julia
using Copulas
using Distributions
using Random
using Turing
using StatsPlots

# Construct the true model and sample from it
C = ClaytonCopula(3,4)
Random.seed!(123)
draws = rand(C, 2_000)
# we could also include some marginal distributions through SklarDist


@model function mymodel(X)
    # Priors
    atoms_increments ~ filldist(TruncatedNormal(0,1,0,Inf),4)
    atoms = cumsum(atoms_increments)

    weights ~ Dirichlet(4,1)
    C = WilliamsonCopula(DiscreteNonParametric(atoms,weights),3)

    # Compute the final loglikelyhood
    Turing.Turing.@addlogprob! loglikelihood(C, X)
end

chain = sample(mymodel(draws),IS(), MCMCThreads(), 1000, 4)
plot(chain)
mean(chain)

# Compare the distribution with the true williamson distribution : 
est_C = WilliamsonCopula(DiscreteNonparametric(atoms,weights),3)
plot(ϕ(C,x) for x in 0:0.1:10) # estimated generator. 
plot(ϕ(est_C,x) for x in 0:0.1:10) # true genreator

# what about printing the lambda function insterad of the generator ? 
```




