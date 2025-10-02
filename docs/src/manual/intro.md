```@meta
CurrentModule = Copulas
```

# [Copulas and Sklar Distributions](@id intro)

This section gives some general definitions and tools about dependence structures, multivariate random vectors and copulas. Along this journey through the mathematical theory of copulas, we link to the rest of the documentation for more specific and detailed arguments on particular points, or simply to the technical documentation of the actual implementation. 
The interested reader can take a look at the standard books on the subject [joe1997,cherubini2004,nelsen2006,joe2014](@cite) or more recently [mai2017, durante2015a, czado2019,grosser2021](@cite). 

We start here by defining a few concepts about multivariate random vectors, dependence structures and copulas.

## Reminder on multivariate random vectors


Consider a real valued random vector $\boldsymbol X = \left(X_1,...,X_d\right): \Omega \to \mathbb R^d$. The random variables $X_1,...,X_d$ are called the marginals of the random vector $\boldsymbol X$. 

!!! info "Constructing random variables in Julia via `Distributions.jl`"
    Recall that you can construct random variables in Julia by the following code : 

    ```@example 1
    using Distributions
    X₁ = Normal()       # A standard Gaussian random variable
    X₂ = Gamma(2,3)     # A Gamma random variable
    X₃ = Pareto(1)      # A Pareto random variable with infinite variance.
    X₄ = LogNormal(0,1) # A Lognormal random variable 
    nothing # hide
    ```
    
    We refer to [Distributions.jl's documentation](https://github.com/JuliaStats/Distributions.jl) for more details on what you can do with these objects. We assume here that you are familiar with their API.


The probability distribution of the random vector $\boldsymbol X$ can be characterized by its *distribution function* $F$: 
```math
\begin{align*}
  F(\boldsymbol x) &= \mathbb P\left(\boldsymbol X \le \boldsymbol x\right)\\
  &= \mathbb P\left(\forall i \in \{1,...,d\},\; X_i \le x_i\right).
\end{align*}
```
For a function $F$ to be the distribution function of some random vector, it should be $d$-increasing, right-continuous and left-limited. 
For $i \in \{1,...,d\}$, the random variables $X_1,...,X_d$, called the marginals of the random vector, also have distribution functions denoted $F_1,...,F_d$ and defined by : 
```math
F_i(x_i) = F(+\infty,...,+\infty,x_i,+\infty,...,+\infty).
```

Note that the range $\mathrm{Ran}(F)$ of a distribution function $F$, univariate or multivariate, is always contained in $[0,1]$. When the random vector or random variable is absolutely continuous with respect to (w.r.t.) the Lebesgue measure restricted to its domain, the range is exactly $[0,1]$. When the distribution is discrete with $n$ atoms, the range is a finite set of $n+1$ values in $[0,1]$.

## [Copulas and Sklar's Theorem](@id copula_and_sklar)

There is a fundamental functional link between the function $F$ and its marginals $F_1,...,F_d$. This link is expressed by the mean of *copulas*. 

!!! definition "Copula" 
    A copula, usually denoted $C$, is the distribution function of a random vector with marginals that are all uniform on $[0,1]$, i.e.
    
    $C_i(u) = u\mathbb 1_{u \in [0,1]} \text{ for all }i \in 1,...,d.$


!!! info "Vocabulary"
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

You can also plot it:

```@example 1
using Plots
plot(C, :logpdf)
```

See [the visualizations page](@ref viz_page) for details on the visualisations tools. It’s often useful to get an intuition by looking at scatter plots.

One of the reasons that makes copulas so useful is the bijective map from the Sklar Theorem [sklar1959](@cite):

!!! theorem "Sklar (1959)"
    For every random vector $\boldsymbol X$, there exists a copula $C$ such that 

    $\forall \boldsymbol x\in \mathbb R^d, F(\boldsymbol x) = C(F_{1}(x_{1}),...,F_{d}(x_{d})).$
    The copula $C$ is uniquely determined on $\mathrm{Ran}(F_{1}) \times ... \times \mathrm{Ran}(F_{d})$, where $\mathrm{Ran}(F_i)$ denotes the range of the function $F_i$. In particular, if all marginals are absolutely continuous, $C$ is unique.


This result allows to decompose the distribution of $\boldsymbol X$ into several components: the marginal distributions on one side, and the copula on the other side, which governs the dependence structure between the marginals. This object is central in our work, and therefore deserves a moment of attention. 

!!! example "Independence"
    The function 

    $\Pi : \boldsymbol x \mapsto \prod_{i=1}^d x_i = \boldsymbol x^{\boldsymbol 1}$ is a copula, corresponding to independent random vectors.

The independence copula can be constructed using the [`IndependentCopula(d)`](@ref IndependentCopula) syntax as follows: 

```@example 1
Π = IndependentCopula(d) # A 4-variate independence structure.
nothing # hide
```

We can then leverage the Sklar theorem to construct multivariate random vectors from a copula-marginals specification. The implementation we have of this theorem allows building multivariate distributions by specifying separately their marginals and dependence structures as follows:


```@example 1
X₁, X₂, X₃ = Gamma(2,3), Pareto(), LogNormal(0,1) # Marginals
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C, (X₁,X₂,X₃)) # The final distribution
plot(D, scale=:sklar)
nothing # hide
```

The obtained multivariate random vector object are genuine multivariate random vector following the `Distributions.jl` API. They can be sampled (`rand()`), and their probability density function and distribution function can be evaluated (respectively `pdf` and `cdf`), etc:

```@example 1
x = rand(D,10)
p = pdf(D, x)
l = logpdf(D, x)
c = pdf(D, x)
[x' p l c]
```

Sklar's theorem can be used the other way around (from the marginal space to the unit hypercube): this is, for example, what the [`pseudo()`](@ref Pseudo-observations) function does, computing ranks.

!!! info "Independent random vectors"

    `Distributions.jl` provides the [`product_distribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#Product-distributions) function to create independent random vectors with given marginals. `product_distribution(args...)` is essentially equivalent to `SklarDist(Π, args)`, but our approach generalizes to other dependence structures.

Copulas are bounded functions with values in [0,1] since they correspond to probabilities. But their range can be bounded more precisely, and [lux2017](@cite) gives us:

!!! property "Fréchet-Hoeffding bounds" 
    For all $\boldsymbol x \in [0,1]^d$, every copula $C$ satisfies : 

    $$\langle \boldsymbol 1, \boldsymbol x - 1 + d^{-1}\rangle_{+} \le C(\boldsymbol x) \le \min \boldsymbol x,$$
    
    where $y_{+} = \max(0,y)$.

The function $M : \boldsymbol x \mapsto \min\boldsymbol x$, called the upper Fréchet-Hoeffding bound, is a copula. The function $W : \boldsymbol x \mapsto \langle \boldsymbol 1, \boldsymbol x - 1 + d^{-1}\rangle_{+}$, called the lower Fréchet-Hoeffding bound, is on the other hand a copula only when $d=2$. 
These two copulas can be constructed through [`MCopula(d)`](@ref MCopula) and [`WCopula(2)`](@ref WCopula). 

The upper Fréchet-Hoeffding bound corresponds to the case of comonotone random vector: a random vector $\boldsymbol X$ is said to be comonotone, i.e., to have copula $M$, when each of its marginals can be written as a non-decreasing transformation of the same random variable (say with $\mathcal U\left([0,1]\right)$ distribution). This is a simple but important dependence structure. See e.g.,[kaas2002,hua2017](@cite) on this particular copula. 

Here is a plot of the independence, a positive dependence (Clayton), and the Fréchet bounds in bivariate cases. You can visualize the strong alignment for `M` and the anti-diagonal pattern for `W`.

```@example 1
p1 = plot(IndependentCopula(2), title="IndependentCopula(2)")
p2 = plot(ClaytonCopula(2, 3.0), title="ClaytonCopula(2, 3.0)")
p3 = plot(MCopula(2), title="MCopula(2)")
p4 = plot(WCopula(2), title="WCopula(2)")
plot(p1,p2,p3,p4; layout=(2,2), size=(800,600))
```

Since copulas are distribution functions, like distribution functions of real-valued random variables and random vectors, there exists classical and useful parametric families of copulas (we already saw the Clayton family). You can browse the available families in this package in the [Bestiary](@ref bestiary_ref). Like any families of random variables or random vectors, copulas are fittable on empirical data. 

## A tour of the main API

The public API of `Copulas.jl` is quite small and easy to expose on a simple example, which is what we will do right now.

### Copulas and SklarDist

The most important objects of the package are of course copulas as sklar distributions. Both of these objects follow the `Distributions.jl`'s API, and so you can construct, sample, and evaluate copulas as standard `Distributions.jl` objects:

```@example api
using Copulas, Distributions, Random, StatsBase
C = ClaytonCopula(3, 2.0)
u = rand(C, 5)
Distributions.loglikelihood(C, u)
```

```@example api
X₁, X₂, X₃ = Gamma(2,3), Beta(1,5), LogNormal(0,1)
C2 = GumbelCopula(3, 1.7)
D = SklarDist(C2, (X₁, X₂, X₃))
rand(D, 3)
pdf(D, rand(3))
```

### Basic dependence metrics. 

Basic dependence summaries available on copulas, whatever their dimension `d`: 

```@example api
multivariate_stats = (
    kendall_tau = Copulas.τ(C),
    spearm_rho = Copulas.ρ(C),
    blomqvist_beta = Copulas.β(C),
    gini_gamma = Copulas.γ(C), 
    entropy_iota = Copulas.ι(C), 
    lower_tail_dep = Copulas.λₗ(C), 
    upper_tail_dep = Copulas.λᵤ(C)
)
```

The same functions have dispatches for `u::Abstractmatrix` of size `(d,d)` where `d` is the dimension of the copula and `n` is the number of observations, which provide sample versions of the same quantities. Moreover, since most of these statistics are more common in bivariate case, we provide the folllowing bindings for pairwise matrices of the same dependence metrics: 

```@example api
StatsBase.corkendall(C)
StatsBase.corspearman(C)
Copulas.corblomqvist(C)
Copulas.corgini(C)
Copulas.corentropy(C)
Copulas.corlowertail(C)
Copulas.coruppertail(C)
```

and of course once again the same functions dispatch on `u::Abstractmatrix`, but, for historical reasons, they require dataset to be `(n,d)`-shaped and not `(d,n)`, so you have to transpose.

### Measure function 

A `measure` function gives the measure of hypercubes from any copula as follows: 
```@example api
Copulas.measure(C, (0.1,0.2,0.3), (0.9,0.8,0.7))
```

### Subsetting (working with a subset of dimensions)

Extract lower-dimensional dependence without materializing new data:

```@example api
S23 = subsetdims(C2, (2,3))  # a bivariate copula view
StatsBase.corkendall(S23)
```

For Sklar distributions, subsetting returns a smaller joint distribution:

```@example api
D13 = subsetdims(D, (1,3))
rand(D13, 2)
```

### Conditioning (conditional marginals and joint conditionals)

On the uniform scale (copula): distortions and conditional copulas are provided:

```@example api
Dj = condition(C2, 2, 0.3)   # Distortion of U₁|U₂=0.3 when d=2
Distributions.cdf(Dj, 0.95)
```

On the original scale (Sklar distribution):

```@example api
Dc = condition(D, (2,3), (0.3, 0.2))
rand(Dc, 2)
```


And rosenblatt transfromations of the copula (or sklardist) can be obtained as follows: 
```@example api
u = rand(D, 10)
s = rosenblatt(D, u)
u2 = inverse_rosenblatt(D, s)
maximum(abs, u2 .- u) # should be approx zero.
```

These transformation leverage the conditioning mechanismes. 

### Fitting (copulas and Sklar distributions)

You can fit copulas from pseudo-observations U, and Sklar distributions from raw data X. Available methods vary by family; see the fitting manual for details.

```@example api
X = rand(D, 500)
M = fit(CopulaModel, SklarDist{GumbelCopula, Tuple{Gamma,Beta,LogNormal}}, X; copula_method=:mle)
```

A shortcut allows to directly get the fitting object (copula or sklardist) by simply ommiting the first `CopulaModel` argument: 

```@example api
U = pseudos(X)
Ĉ = fit(GumbelCopula, U; method=:itau)
Copulas.τ(Ĉ)
```

Notes:
- `fit` chooses a reasonable default per-family; pass `method`/`copula_method` to control it.
- Common copula methods include `:mle`, `:itau`, `:irho`, `:ibeta`; for Sklar fitting, `:ifm` (parametric CDFs) and `:ecdf` (pseudo-observations) are available.
- `CopulaModel` implements model stats: `nobs`, `coef`, `vcov`, `stderror`, `confint`, `aic/bic`, `nullloglikelihood`, and more.
- For a Bayesian workflow over Sklar models, see the examples section.

!!! info "About fitting procedures"
    The `Distributions.jl` documentation states:

    > The fit function will choose a reasonable way to fit the distribution, which, in most cases, is maximum likelihood estimation.

    We embrace this philosophy: from one copula family to another, the default fitting method may differ. Treat `fit` as a quick starting point; when you need control, specify `method`/`copula_method` explicitly.


## Next steps

The documentation of this package aims to combine theoretical information and references to the literature with practical guidance related to our specific implementation. It can be read as a lecture, or used to find the specific feature you need through the search function. We hope you find it useful.

!!! tip "Explore the bestiary!"
    The package contains *many* copula families. Classifying them is essentially impossible, since the class is infinite-dimensional, but the package proposes a few standard classes: elliptical, archimedean, extreme value, empirical...
    
    Each of these classes more or less corresponds to an abstract type in our type hierarchy, and to a section of this documentation. Do not hesitate to explore the bestiary !

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
