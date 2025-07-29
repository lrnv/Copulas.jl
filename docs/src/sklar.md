```@meta
CurrentModule = Copulas
```

# Sklar's Distribution

Recall the following theorem from [sklar1959](@cite):

!!! theorem "Theorem (Sklar):"
    For every random vector $\bm X$, there exists a copula $C$ such that

    $\forall \bm x \in \mathbb{R}^d, F(\bm x) = C(F_{1}(x_{1}), ..., F_{d}(x_{d})).$
    The copula $C$ is uniquely determined on $\mathrm{Ran}(F_{1}) \times ... \times \mathrm{Ran}(F_{d})$, where $\mathrm{Ran}(F_i)$ denotes the range of the function $F_i$. In particular, if all marginals are absolutely continuous, $C$ is unique.


The implementation we have of this theorem allows building multivariate distributions by specifying separately their marginals and dependence structures as follows:


```@example 2
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution
```

Although the output is not formatted, the model is constructed and can be used in different ways: 

```@example 2
u = rand(D,10)
```

```@example 2
pdf(D, u)
```
```@example 2
cdf(D, u)
```


From this construction, the object `D` is a genuine multivariate random vector following the `Distributions.jl` API. It can be sampled (`rand()`), and its probability density function and distribution function can be evaluated (respectively `pdf` and `cdf`), etc.


```@docs
SklarDist
```

```@bibliography
Pages = [@__FILE__]
Canonical = false
```