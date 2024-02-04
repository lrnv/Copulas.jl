```@meta
CurrentModule = Copulas
```

## Multivariate random vectors

This section gives some general definitions and tools about dependence structures. Along this journey through the mathematical theory of copulas, we link to the rest of the documentation for more specific and detailed arguments on particular points, or simply to link the technical documentation of our implementation. 

The interested theoretical reader can take a look at the standard books on the subject [joe1997,cherubini2004,nelsen2006,joe2014](@cite) or more recently [mai2017, durante2015a, czado2019,grosser2021](@cite). We start here by defining a few concepts about dependence structures and copulas.


Consider a real valued random vector $\bm X = \left(X_1,...,X_d\right): \Omega \to \mathbb R^d$. The random variables $X_1,...,X_d$ are called the marginals of the random vector $\bm X$. 

!!! info "Constructing random variables in Julia via `Distributions.jl`"
    Recall that you can construct random variables in Julia by the following code : 

    ```julia
    using Distributions
    X₁ = Normal()       # A standard gaussian random variable
    X₂ = Gamma(2,3)     # A Gamma random variable
    X₃ = Pareto(1)      # A Pareto random variable with no variance.
    X₄ = LogNormal(0,1) # A Lognormal random variable 
    ```
    
    We refer to [Distributions.jl's documentation](https://github.com/JuliaStats/Distributions.jl) for more details on what you can do with these objects, but here we assume that you are familiar with their API.


The distribution of the random vector $\bm X$ can be characterized by its distribution function $F$: 
```math
\begin{align*}
  F(\bm x) &= \mathbb P\left(\bm X \le \bm x\right)\\
  &= \mathbb P\left(\forall i \in \{1,...,d\},\; X_i \le x_i\right).
\end{align*}
```
For a function $F$ to be the distribution function of some random vector, it should be $d$-increasing, right-continuous and left-limited. 
For $i \in \{1,...,d\}$, the random variables $X_1,...,X_d$, called the marginals of the random vector, also have distribution functions denoted $F_1,...,F_d$ and defined by : 
```math
F_i(x_i) = F(+\infty,...,+\infty,x_i,+\infty,...,+\infty).
```

Note that the range $\mathrm{Ran}(F)$ of a distribution function $F$, univariate or multivariate, is always contained in $[0,1]$. When the random vector or random variable is absolutely continuous with respect to (w.r.t.) the Lebesgue measure restricted to its domain, the range is exactly $[0,1]$. When the distribution is discrete with $n$ atoms, the range is a finite set of $n+1$ values in $[0,1]$.

## Copulas and Sklar's Theorem

There is a fundamental functional link between the function $F$ and its marginals $F_1,...,F_d$. This link is expressed by the mean of *copulas*. 

> **Definition (Copula) :** A copula, usually denoted $C$, is the distribution function of a random vector with marginals that are all uniform on $[0,1]$, i.e.
>
> $C_i(u) = u\mathbb 1_{u \in [0,1]} \text{ for all }i \in 1,...,d.$

!!! note "Vocabulary"
    In this documentation but more largely in the literature, the term *Copula* refers both to the random vector and its distribution function. Usually, the distinction is clear from context. 

At the grounds of the theory of copulas lies Sklar's Theorem [sklar1959](@cite), dating back from 1959.

> **Theorem (Sklar):** For every random vector $\bm X$, there exists a copula $C$ such that 
>
> $\forall \bm x\in \mathbb R^d, F(\bm x) = C(F_{1}(x_{1}),...,F_{d}(x_{d})).$
> The copula $C$ is uniquely determined on $\mathrm{Ran}(F_{1}) \times ... \times \mathrm{Ran}(F_{d})$, where $\mathrm{Ran}(F_i)$ denotes the range of the function $F_i$. In particular, if all marginals are absolutely continuous, $C$ is unique.


This result allows to decompose the distribution of $\bm X$ into several components: the marginal distributions on one side, and the copula on the other side, which governs the dependence structure between the marginals. This object is central in our work, and therefore deserves a moment of attention. 

> **Example (Independence):** The function 
>
> $\Pi : \bm x \mapsto \prod_{i=1}^d x_i = \bm x^{\bm 1}$ is a copula, corresponding to independent random vectors.

The independence copula can be constructed using the [`IndependentCopula(d)`](@ref IndependentGenerator) syntax as follows: 

```julia
using Copulas
d = 4 # The dimension of the model
Π = IndependentCopula(d)
```

And then the Sklar's theorem can be applied to it as follows, using the previously-defined marginals : 

```julia
MyDistribution = SklarDist(Π, (X₁,X₂,X₃,X₄))
```

!!! note "Independent random vectors"

    Distributions.jl proposes the [`product_distribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#Product-distributions) function to create those independent random vectors with given marginals. But you'll see that our approach is much more powerfull. 

Copulas are bounded functions

> **Property (Fréchet-Hoeffding bounds [lux2017](@cite)):** For all $\bm x \in [0,1]^d$, every copula $C$ satisfies : 
>
>$\langle \bm 1, \bm x - 1 + d^{-1}\rangle_{+} \le C(\bm x) \le \min \bm x,$
>where $y_{+} = \max(0,y)$.

> **Example (Fréchet-Hoeffding bounds [lux2017](@cite)):** The function $M : \bm x \mapsto \min\bm x$, called the upper Fréchet-Hoeffding bound, is a copula. The function $W : \bm x \mapsto \langle \bm 1, \bm x - 1 + d^{-1}\rangle_{+}$, called the lower Fréchet-Hoeffding bound, is on the other hand a copula only when $d=2$. 


These two copulas can be constructed through [`MCopula(d)`](@ref MGenerator) and [`WCopula(2)`](@ref WGenerator). The upper Fréchet-Hoeffding bound corresponds to the case of comonotone random vector: a random vector $\bm X$ is said to be comonotone, i.e., to have copula $M$, when each of its marginals can be written as a non-decreasing transformation of the same random variable (say with $\mathcal U\left([0,1]\right)$ distribution). This is a simple but important dependence structure. See e.g.,[kaas2002,hua2017](@cite) on this particular copula.

Since copulas are distribution functions, like distribution functions of real-valued random variables and random vectors, there exists classical and useful parametric families of copulas. This is mostly the content of this package, and we refer to the rest of the documentation for more details on the models and their implementations. 

## Fitting copulas and compound distributions.

`Distributions.jl` proposes the `fit` function in their API for random ve tors and random variables. We used it to implement fitting of multivariate models (copulas, of course, but also compound distributions). It can be used as follows: 


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

!!! info "About fitting methods"
    [`Distributions.jl` documentation](https://juliastats.org/Distributions.jl/stable/fit/#Distribution-Fitting) states that : 

    > The fit function will choose a reasonable way to fit the distribution, which, in most cases, is maximum likelihood estimation.

    The results of this fitting function should then only be used as "quick-and-dirty" fits, since the fitting method is "hidden" to the user. We embrace this philosophy: from one copula to the other, the fitting method might not be the same. 

## Going further

There are a lot of available copula families in the package, that can be regrouped into a few classes:
- [Elliptical Copulas](@ref elliptical_copulas_header), including the Gaussian and Student cases. 
- [Archimedean Copulas](@ref archimedean_copulas_header), leveraging their [Archimedean Generators](@ref archimedean_copulas_header)
- [Fréchet-Hoeffding bounds](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds), 
- [Other Copulas](@ref)

Each of these classes more-or-less correspond to an abstract type in our type hierarchy, and to a section of this documentation. 


```@bibliography
Pages = ["getting_started.md"]
Canonical = false
```
