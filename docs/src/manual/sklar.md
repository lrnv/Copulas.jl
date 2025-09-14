```@meta
CurrentModule = Copulas
```

# Sklar's Distribution

Recall the following theorem from [sklar1959](@cite):

!!! theorem "Theorem (Sklar):"
    For every random vector $\boldsymbol X$, there exists a copula $C$ such that

    $\forall \boldsymbol x \in \mathbb{R}^d, F(\boldsymbol x) = C(F_{1}(x_{1}), ..., F_{d}(x_{d})).$
    The copula $C$ is uniquely determined on $\mathrm{Ran}(F_{1}) \times ... \times \mathrm{Ran}(F_{d})$, where $\mathrm{Ran}(F_i)$ denotes the range of the function $F_i$. In particular, if all marginals are absolutely continuous, $C$ is unique.


The implementation we have of this theorem allows building multivariate distributions by specifying separately their marginals and dependence structures as follows:


```@example 2
using Copulas, Distributions, Random
X₁, X₂, X₃ = Gamma(2,3), Pareto(), LogNormal(0,1) # Marginals
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution
```

The obtained multivariate random vector object are genuine multivariate random vector following the `Distributions.jl` API. They can be sampled (`rand()`), and their probability density function and distribution function can be evaluated (respectively `pdf` and `cdf`), etc:

```@example 2
x = rand(D,10)
p = pdf(D, x)
l = logpdf(D, x)
c = pdf(D, x)
[x' p l c]
```


Copulas live on the unit hypercube $[0,1]^d$, but, applying non-uniform marginals via `SklarDist` warps the sample space. However we can go back to the unit hypercube  This can be seen by plotting the models as follows: 

```@example 2
using Plots
plot(D)
```
By default, the main plots are on the copula hypercube scale You can get them on themarginal scale as follows: 

```@example 2
using Plots
plot(D, scale=:sklar)
```

```@docs
SklarDist
```

```@bibliography
Pages = [@__FILE__]
Canonical = false
```