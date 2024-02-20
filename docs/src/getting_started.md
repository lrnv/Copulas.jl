```@meta
CurrentModule = Copulas
```

## Multivariate random vectors

This section gives some general definitions and tools about dependence structures, multivariate random vectors and copulas. Along this journey through the mathematical theory of copulas, we link to the rest of the documentation for more specific and detailed arguments on particular points, or simply to the technical documentation of the actual implementation. 
The interested theoretical reader can take a look at the standard books on the subject [joe1997,cherubini2004,nelsen2006,joe2014](@cite) or more recently [mai2017, durante2015a, czado2019,grosser2021](@cite). 

We start here by defining a few concepts about dependence structures and copulas.
Consider a real valued random vector $\bm X = \left(X_1,...,X_d\right): \Omega \to \mathbb R^d$. The random variables $X_1,...,X_d$ are called the marginals of the random vector $\bm X$. 

!!! info "Constructing random variables in Julia via `Distributions.jl`"
    Recall that you can construct random variables in Julia by the following code : 

    ```@example 1
    using Distributions
    X₁ = Normal()       # A standard gaussian random variable
    X₂ = Gamma(2,3)     # A Gamma random variable
    X₃ = Pareto(1)      # A Pareto random variable with no variance.
    X₄ = LogNormal(0,1) # A Lognormal random variable 
    nothing # hide
    ```
    
    We refer to [Distributions.jl's documentation](https://github.com/JuliaStats/Distributions.jl) for more details on what you can do with these objects. We assume here that you are familiar with their API.


The probability distribution of the random vector $\bm X$ can be characterized by its *distribution function* $F$: 
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

You may define a copula object in Julia by simply calling its constructor: 

```@example 1
using Copulas
d = 4 # The dimension of the model
θ = 7 # Parameter
C = ClaytonCopula(4,7) # A 4-dimensional clayton copula with parameter θ = 7.
```

This object is a random vector, and behaves exactly as you would expect a random vector from `Distributions.jl` to behave: you may sample it with `rand(C,100)`, compute its pdf or cdf with `pdf(C,x)` and `cdf(C,x)`, etc:

```@example 1
u = rand(C,10)
```
```@example 1
cdf(C,u)
```

One of the reasons that makes copulas so useful is discovered by Sklar [sklar1959](@cite) in 1959:

> **Theorem (Sklar):** For every random vector $\bm X$, there exists a copula $C$ such that 
>
> $\forall \bm x\in \mathbb R^d, F(\bm x) = C(F_{1}(x_{1}),...,F_{d}(x_{d})).$
> The copula $C$ is uniquely determined on $\mathrm{Ran}(F_{1}) \times ... \times \mathrm{Ran}(F_{d})$, where $\mathrm{Ran}(F_i)$ denotes the range of the function $F_i$. In particular, if all marginals are absolutely continuous, $C$ is unique.


This result allows to decompose the distribution of $\bm X$ into several components: the marginal distributions on one side, and the copula on the other side, which governs the dependence structure between the marginals. This object is central in our work, and therefore deserves a moment of attention. 

> **Example (Independence):** The function 
>
> $\Pi : \bm x \mapsto \prod_{i=1}^d x_i = \bm x^{\bm 1}$ is a copula, corresponding to independent random vectors.

The independence copula can be constructed using the [`IndependentCopula(d)`](@ref IndependentGenerator) syntax as follows: 

```@example 1
Π = IndependentCopula(d) # A 4-variate independence structure.
nothing # hide
```

We leverage the Sklar theorem to construct multivariate random vectors from a copula-marginals specification. This can be used as follows: 

```@example 1
MyDistribution = SklarDist(Π, (X₁,X₂,X₃,X₄))
MyOtherDistribution = SklarDist(C, (X₁,X₂,X₃,X₄))
nothing # hide
```

And the API is still the same: 
```@example 1
rand(MyDistribution,10)
```
```@example 1
rand(MyOtherDistribution,10)
```


On the other hand, the [`pseudo()`](@ref Pseudo-observations) function computes ranks, effectively using Sklar's theorem the other way around (from the marginal space to the unit hypercube).

!!! note "Independent random vectors"

    Distributions.jl proposes the [`product_distribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#Product-distributions) function to create those independent random vectors with given marginals. But you can already see that our approach generalizes to other dependence structres, and is thus much powerfull. 

Copulas are bounded functions with values in [0,1] since they correspond to probabilities. But their range can be bounded more precisely:

> **Property (Fréchet-Hoeffding bounds [lux2017](@cite)):** For all $\bm x \in [0,1]^d$, every copula $C$ satisfies : 
>
>$\langle \bm 1, \bm x - 1 + d^{-1}\rangle_{+} \le C(\bm x) \le \min \bm x,$
>where $y_{+} = \max(0,y)$.

The function $M : \bm x \mapsto \min\bm x$, called the upper Fréchet-Hoeffding bound, is a copula. The function $W : \bm x \mapsto \langle \bm 1, \bm x - 1 + d^{-1}\rangle_{+}$, called the lower Fréchet-Hoeffding bound, is on the other hand a copula only when $d=2$. 
These two copulas can be constructed through [`MCopula(d)`](@ref MGenerator) and [`WCopula(2)`](@ref WGenerator). 

The upper Fréchet-Hoeffding bound corresponds to the case of comonotone random vector: a random vector $\bm X$ is said to be comonotone, i.e., to have copula $M$, when each of its marginals can be written as a non-decreasing transformation of the same random variable (say with $\mathcal U\left([0,1]\right)$ distribution). This is a simple but important dependence structure. See e.g.,[kaas2002,hua2017](@cite) on this particular copula. Note that the implementation of their sampler was straightforward due to their particular shapes:

```@example 1
rand(MCopula(2),10) # sampled values are all equal, this is comonotony
```
```@example 1
u = rand(WCopula(2),10)
sum(u, dims=1) # sum is always equal to one, this is anticomonotony
```

Since copulas are distribution functions, like distribution functions of real-valued random variables and random vectors, there exists classical and useful parametric families of copulas. This is mostly the content of this package, and we refer to the rest of the documentation for more details on the models and their implementations. 

## Fitting copulas and compound distributions.

`Distributions.jl`'s API contains a `fit` function for random vectors and random variables. We propose an implementation of it for copulas and multivariate compound distributions (composed of a copula and some given marginals). It can be used as follows: 

```@example 2
using Copulas, Distributions, Random
# Construct a given model:
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # Generate a dataset

# You may estimate a copula using the `fit` function:
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
```

We see on the output that the parameters were correctly estimated from this sample. More details on the estimator, including, e.g., standard errors, may be obtained with more complicated estimation routines. For a Bayesian approach using  `Turing.jl`, see [this example](@ref Bayesian-inference-with-Turing.jl).

!!! info "Fitting procedures are not part of the API"
    [`Distributions.jl` documentation](https://juliastats.org/Distributions.jl/stable/fit/#Distribution-Fitting) states that: 

    > The fit function will choose a reasonable way to fit the distribution, which, in most cases, is maximum likelihood estimation.

    The results of this fitting function should then only be used as "quick-and-dirty" fits, since the fitting method is "hidden" to the user and might even change without breaking releases. We embrace this philosophy: from one copula to the other, the fitting method might not be the same. 

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
