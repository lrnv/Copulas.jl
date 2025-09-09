```@meta
CurrentModule = Copulas
```

# Vines Copulas

!!! todo "Not implemented yet!"
    Do not hesitate to join the discussion on [our GitHub](https://github.com/lrnv/Copulas.jl)!

One notable class of copulas is the Vines copulas. These distributions use a graph of conditional distributions to encode the distribution of the random vector. To define such a model, working with conditional densities, and given any ordered partition $\bm i_1, ..., \bm i_p$ of $1, ..., d$, we write:

$$f(\bm x) = f(x_{\bm i_1}) \prod\limits_{j=1}^{p-1} f(x_{\bm i_{j+1}} | x_{\bm i_j}).$$

Of course, the choice of partition, its order, and the conditional models is left to the practitioner. The goal when dealing with such dependency graphs is to tailor the graph to reduce the approximation error, which can be a challenging task. There exist simplifying assumptions that help with this, and we refer to [durante2017a, nagler2016, nagler2018, czado2013, czado2019, graler2014](@cite) for a deep dive into vine theory, along with results and extensions.

## Visual hints (work in progress)

```@example 1
using Copulas, Plots, StatsBase
# As a simple proxy, look at pairwise τ to guide first tree choices
C = GaussianCopula([1.0 0.7 0.2; 0.7 1.0 0.5; 0.2 0.5 1.0])
U = rand(C, 1500)
τ = StatsBase.corkendall(U')
heatmap(τ; title="Empirical Kendall τ (pairwise)", aspect_ratio=1, c=:blues, clim=(-1,1))
```

```@example 1
# Show a simple pair-copula composition idea via partial residuals (illustrative)
scatter(U[1,:], U[2,:]; ms=2, alpha=0.6, xlim=(0,1), ylim=(0,1), title="First pair (1,2)", legend=false)
```


```@bibliography
Pages = [@__FILE__]
Canonical = false
```