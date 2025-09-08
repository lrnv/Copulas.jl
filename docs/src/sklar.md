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

### Visualizing the effect of marginals

Although copulas live on the unit square/cube, applying non-uniform marginals via `SklarDist`
warps the sample space. Here we draw from two models with the same copula but different marginals,
and compare their scatter plots after transforming back to uniforms with pseudo-observations:

```@example 2
using Plots
N = 1000
X = SklarDist(C, (Normal(), Normal()))
Y = SklarDist(C, (LogNormal(), Gamma(2,2)))
Ux = rand(X, N)
Uy = rand(Y, N)
P1 = scatter(Ux[1,:], Ux[2,:]; ms=2, title="X on original scale", legend=false)
P2 = scatter(Uy[1,:], Uy[2,:]; ms=2, title="Y on original scale", legend=false)
plot(P1, P2; layout=(1,2), size=(800,350))
```


From this construction, the object `D` is a genuine multivariate random vector following the `Distributions.jl` API. It can be sampled (`rand()`), and its probability density function and distribution function can be evaluated (respectively `pdf` and `cdf`), etc.

### Same copula, different marginals: back to uniforms

```@example 2
using StatsBase
Yx = rand(SklarDist(C, (Normal(), Gamma(2,2))), 1500)
Yy = rand(SklarDist(C, (LogNormal(), LogNormal())), 1500)
Ux = pseudos(Yx') # N×d
Uy = pseudos(Yy')
P1 = scatter(Yx[1,:], Yx[2,:]; ms=2, alpha=0.6, title="Original scale X", legend=false)
P2 = scatter(Yy[1,:], Yy[2,:]; ms=2, alpha=0.6, title="Original scale Y", legend=false)
P3 = scatter(Ux[:,1], Ux[:,2]; ms=2, alpha=0.6, title="Back to uniforms (X)", legend=false, xlim=(0,1), ylim=(0,1))
P4 = scatter(Uy[:,1], Uy[:,2]; ms=2, alpha=0.6, title="Back to uniforms (Y)", legend=false, xlim=(0,1), ylim=(0,1))
plot(P1, P2, P3, P4; layout=(2,2), size=(900,650))
```


```@docs
SklarDist
```

```@bibliography
Pages = [@__FILE__]
Canonical = false
```