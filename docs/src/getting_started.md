```@meta
CurrentModule = Copulas
```

## Multivariate random vectors

This section gives some general definitions and tools about dependence structures. Along this journey through the mathematical theory of copulas, we link to the rest of the documentation for more specific and detailed arguments on particular points, or simply to link the technical documentation of our implementation. 

The interested theoretical reader can take a look at the standard books on the subject [joe1997,cherubini2004,nelsen2006,joe2014](@cite) or more recently [mai2017, durante2015a, czado2019,grosser2021](@cite). We start here by defining a few concepts about dependence structures and copulas.


Consider a real valued random vector $\bm X = \left(X_1,...,X_d\right): \Omega \to \mathbb R^d$. The random variables $X_1,...,X_d$ are called the marginals of the random vector $\bm X$. 

!!! info "Constructing random variables in Julia via Distributions.jl"
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
Π = IndependenceCopula(d)
```

And then the Sklar's theorem can be applied to it as follows, using the previously-defined marginals : 

```julia
MyDistribution = SklarDist(Π, (X₁,X₂,X₃,X₄))
```

!!! note "Independent random vectors"

    Distributions.jl proposes the [`product_distribution`](https://juliastats.org/Distributions.jl/stable/multivariate/#Product-distributions) function to create those independent random vectors with given marginals. But you'll see that our approach is much more powerfull. 

Copulas are bounded functions

> **Property (Fréchet-Hoeffding bounds [lux2017](@cite):** For all $\bm x \in [0,1]^d$, every copula $C$ satisfies : 
>
>$\langle \bm 1, \bm x - 1 + d^{-1}\rangle_{+} \le C(\bm x) \le \min \bm x,$
>where $y_{+} = \max(0,y)$.

> **Example (Fréchet-Hoeffding bounds [lux2017](@cite):** The function $M : \bm x \mapsto \min\bm x$, called the upper Fréchet-Hoeffding bound, is a copula. The function $W : \bm x \mapsto \langle \bm 1, \bm x - 1 + d^{-1}\rangle_{+}$, called the lower Fréchet-Hoeffding bound, is on the other hand a copula only when $d=2$. 


These two copulas can be constructed through [`MCopula(d)`](@ref MGenerator) and [`WCopula(2)`](@ref WGenerator). The upper Fréchet-Hoeffding bound corresponds to the case of comonotone random vector: a random vector $\bm X$ is said to be comonotone, i.e., to have copula $M$, when each of its marginals can be written as a non-decreasing transformation of the same random variable (say with $\mathcal U\left([0,1]\right)$ distribution). This is a simple but important dependence structure. See e.g.,[kaas2002,hua2017](@cite) on this particular copula.

Since copulas are distribution functions, like distribution functions of real-valued random variables and random vectors, there exists classical and useful parametric families of copulas. This is mostly the content of this package, and we refer to the rest of the documentaiton for more details on the models and their implementations. 

## Fitting copulas and compound distributions.

`Distributions.jl` proposes the  `fit` function in their API for random ve tors and random variables. We used it to implement fitting of multivariate models (copulas, of course, but also comppound distributions). It can be used as follows: 


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





#### SUite

Through the statistical process leading to the estimation of copulas, one usually observes the data and information on the marginals scale and not on the copula scale. This discrepancy between the observed information and the modeled distribution must be taken into account. A key concept is that of pseudo-observations. 


> **Definition (Pseudo-observations):** If $\bm x \in \mathbb R^{N\times d}$ is an $N$-sample of a $d$-variate real-valued random vector $\bm X$, then the pseudo-observations are the normalized ranks of the marginals of $\bm x$, defined as : 
>
> $$\bm u \in \mathbb [0,1]^{N\times d}:\; u_{i,j} = \frac{\mathrm{Rank}(x_{i,j},\,\bm x_{\cdot,j})}{N+1} = \frac{1}{N+1}\sum_{k=1}^N \mathbb 1_{x_{k,j} \le x_{i,j}},$$
>
> where $\mathrm{Rank}(y,\bm x)  = \sum\limits_{x_i \in \bm x} \mathbb 1_{x_i \le y}$.

From these pseudo-observations, an empirical copula is defined as follows:

> **Definition (Empirical Copula) [deheuvels1979](@cite):** The empirical distribution function of the normalized ranks,
>
> $$\hat{C}_N(\bm u) = \frac{1}{N} \sum_{i=1}^N \mathbb 1_{\bm u_i \le \bm u},$$ is called the empirical copula function.

> **Theorem (Exhaustivity and consistency [deheuvels1979](@cite)):** $\hat{C}_N$ is an exhaustive estimator of $C$, and moreover for any normalizing constants $\{\phi_N, N\in \mathbb N\}$ such that $\lim\limits_{N \to \infty} \phi_N \sqrt{N^{-1}\ln \ln N} = 0$, 
>
>$$\lim\limits_{N\to\infty} \phi_N \sup_{\bm u \in [0,1]^d} \lvert\hat{C}_N(\bm u) - C(\bm u) \rvert = 0 \text{ a.s.}$$


$\hat{C}_N$ then converges (weakly) to $C$, the true copula of the random vector $\bm X$, when the number of observations $N$ goes to infinity. However, despite its name, $\hat{C}_N$ is not a copula since it does not have uniform marginals:

```math
\begin{align*}
  \hat{C}_N(1,...,1,y_j,1,...,1) &= \frac{1}{N}\sum_{i=1}^N \mathbb 1_{u_{i,j} \le y_j}\\
  &= \frac{1}{N}\sum_{i=1}^N \mathbb 1_{\mathrm{Rank}(x_{i,j},\bm x_{\cdot,j}) \le (N+1)y_j}\\
  &= \frac{1}{N}\sum_{i=1}^N \mathbb 1_{i \le (N+1)y_j}\\
  &= \frac{\lfloor(N+1)y_j\rfloor}{N} \neq y_j 
\end{align*}
```


Thus, the empirical copula function is not a copula. An easy way to fix this problem is to smooth out the marginals with beta distribution functions: 

> **Definition (Beta Copula [segers2017](@cite)):** Denoting $F_{n,r}(x) = \sum_{s=r}^n \binom{n}{s} x^s(1-x)^{n-s}$ the distribution function of a $\mathrm{Beta}(r,n+1-r)$ random variable, the function 
>
> $$\hat{C}_N^\beta : \bm x \mapsto \frac{1}{N} \sum_{i=1}^N \prod\limits_{j=1}^d F_{n,(N+1)u_{i,j}}(x_j)$$ is a genuine copula, called the Beta copula. 

> **Property (Proximity of $\hat{C}_N$ and $\hat{C}_N^\beta$ [segers2017](@cite)):**
>
> $$\sup\limits_{\bm u \in [0,1]^d} \lvert \hat{C}_N(\bm u) - \hat{C}_N^\beta(\bm u) \rvert \le d\left(\sqrt{\frac{\ln n}{n}} + \sqrt{\frac{1}{n}} + \frac{1}{n}\right)$$



There are other nonparametric estimators of the copula function that are true copulas. Of interest to our work is the Checkerboard construction (see [cuberos2019,mikusinski2010](@cite)), detailed below.

First, for any $\bm m \in \mathbb N^d$, let $\left\{B_{\bm i,\bm m}, \bm i < \bm m\right\}$ be a partition of the unit hypercube defined by 

```math
B_{\bm i, \bm m} = \left]\frac{\bm i}{\bm m}, \frac{\bm i+1}{\bm m}\right].
```

Furthermore, for any copula $C$ (or more generally distribution function $F$), we denote $\mu_{C}$ (resp $\mu_F$) the associated measure.  For example, for the independence copula $Pi$, $\mu_{\Pi}(A) = \lambda(A \cup [\bm 0, \bm 1])$ where $\lambda$ is the Lebesgue measure.

> **Definition (Empirical Checkerboard copulas [cuberos2019](@cite)):** Let $\bm m \in \mathbb N^d$. The $\bm m$-Checkerboard copula $\hat{C}_{N,\bm m}$, defined by 
>
> $$\hat{C}_{N,\bm m}(\bm x) = \bm m^{\bm 1} \sum_{\bm i < \bm m} \mu_{\hat{C}_N}(B_{\bm i, \bm m}) \mu_{\Pi}(B_{\bm i, \bm m} \cap [0,\bm x])$$ s a genuine copula as soon as $m_1,...,m_d$ all divide $N$.

> **Property (Consistency of $\hat{C}_{N,\bm m}$ [cuberos2019](@cite)):** If all $m_1,..,m_d$ divide $N$, 
>
> $$\sup\limits_{\bm u \in [0,1]^d} \lvert \hat{C}_{N,\bm m}(\bm u) - C(\bm u) \rvert \le \frac{d}{2m} + \mathcal O_{\mathbb P}\left(n^{-\frac{1}{2}}\right).$$

This copula is called *Checkerboard*, as it fills the unit hypercube with hyperrectangles of same shapes $B_{\bm i, \bm m}$, conditionally on which the distribution is uniform, and the mixing weights are the empirical frequencies of the hyperrectangles. 

It can be noted that there is no need for the hyperrectangles to be filled with a uniform distribution ($\mu_{\Pi}$), as soon as they are filled with copula measures and weighted according to the empirical measure in them (or to any other copula). The direct extension is then the more general patchwork copulas, whose construction is detailed below.


Denoting $B_{\bm i, \bm m}(\bm x) = B_{\bm i, \bm m} \cap [0,\bm x]$, we have : 

```math
\begin{align}
  m^d\mu_{\Pi}(B_{\bm i, \bm m} \cap [0,\bm x]) &= \frac{\mu_{\Pi}(B_{\bm i, \bm m} \cap [0,\bm x])}{\mu_{\Pi}(B_{\bm i, \bm m})}\\
  &= \frac{\mu_{\Pi}(B_{\bm i, \bm m}(\bm x))}{\mu_{\Pi}(B_{\bm i, \bm m})}\\
  &= \mu_{\Pi}(\bm m B_{\bm i, \bm m}(\bm x))
\end{align}
```

where we intend $\bm m ]\bm a, \bm b] = ] \bm m \bm a, \bm m \bm b]$ (recall that products between vectors are componentwise).

This allows for an easy generalization in the framework of patchwork copulas: 

> **Definition (Patchwork copulas [durante2012,durante2013,durante2015](@cite)):** Let $\bm m \in \mathbb N^d$ all divide $N$, and let $\mathcal C = \{C_{\bm i}, \bm i < \bm m\}$ be a given collection of copulas. The distribution function
>
> $$\hat{C}_{N,\bm m, \mathcal C}(\bm x) = \sum_{\bm i < \bm m} \mu_{\hat{C}_N}(B_{\bm i, \bm m}) \mu_{C_{\bm i}}(\bm m B_{\bm i, \bm m}(\bm x))$$ is a copula. 


In fact, replacing $\hat{C}_N$ by any copula in Definition \ref{intro:def:patchwork} still yields a genuine copula, with no more conditions that all components of $\bm m$ divide $N$. The Checkerboard grids are practical in the sense that computations associated to a Checkerboard copula can be really fast: if the grid is large, the number of boxes is small, and otherwise if the grid is very refined, many boxes are probably empty. On the other hand, the grid is fixed a priori. There exists potential extensions to adaptative grids, see e.g. [laverny2020](@cite).

Convergence results for this kind of copulas can be found in [durante2015](@cite), with a slightly different parametrization. 

One more noticeable class of copulas are the Vines copulas. These distributions use a graph of conditional distributions to encode the distribution of the random vector. To define such a model, working with conditional densities, and given any ordered partition $\bm i_1,...\bm i_p$ of $1,...d$, we write 
```math
f(\bm x) = f(x_{\bm i_1}) \prod\limits_{j=1}^{p-1} f(x_{\bm i_{j+1}} | x_{\bm i_j}).
```

Of course, the choice of the partition, of its order, and of the conditional models is left to the practitioner. The goal when dealing with such dependency graphs is to tailor the graph to reduce the error of approximation, which can be a tricky task. There exists simplifying assumptions that help with this matter, and we refer to [durante2017a,nagler2016,nagler2018,czado2013,czado2019,graler2014](@cite) for a deep dive into the vine theory, along with some nice results and extensions. 

Although the copula is an object that summarizes completely the dependence structure of any random vector, it is an infinite dimensional object and the interpretation of its properties can be difficult when the dimension gets high. Therefore, the literature has come up with some quantifications of the dependence structure that might be used as univariate summaries, of course imperfect, of certain properties of the copula at hand. 


We now define measures of dependency level.

> **Definition (Kendall's $\tau$ and Spearman's $\rho$):** For a copula $C$ with a density $c$, we define:
>
> $\tau = 4 \int C(\bm u) \, c(\bm u) \;d\bm u -1$
> $\rho &= 12 \int C(\bm u) d\bm u -3.$

Kendall's $\tau$ and Spearman's $\rho$ have values between -1 and 1, and are -1 in case of complete anticonomotony and 1 in case of comonotony. Moreover, they are 0 in case of independence. This is why we say that they measure the 'strength' of the dependency. They make more sense in the bivariate case than in other cases, and therefore we sometimes refer to the Kendall's matrix or the Spearman's matrix for the collection of bivariate coefficients associated to a multivariate copula. Many copula estimators are based on these coefficients, see e.g., [genest2011,fredricks2007](@cite).

There are also tail coefficients: 
> **Definition (Upper tail dependency):** For a copula $C$, we define (when they exist):
>
> $\lambda = \lim\limits_{u \to 1} \frac{1 - 2u - C(u,..,u)}{1- u} \in [0,1]$
> $\chi(u) = \frac{2 \ln(1-u)}{\ln(1-2u-C(u,...,u))} -1$
> $\chi = \lim\limits_{u \to 1} \chi(u) \in [-1,1]$
>
> When $\lambda > 0$, we say that there is a strong upper tail dependency, and $\chi = 1$. When $\lambda = 0$, we say that there is no strong upper tail dependency, and if furthermore $\chi \neq 0$ we say that there is weak upper tail dependency.


The graph of $u \to \chi(u)$ over $[\frac{1}{2},1]$ is an interesting tool to assess the existence and strength of the tail dependency. The same kind of tools can be constructed for the lower tail. 

All these coefficients are useful to quantify the behavior of the dependence structure, both generally and in the extremes, and are therefore widely used in the literature either as verification tools to assess the quality of fits, or even as parameters. Many parametric copulas families have simple surjections, injections, or even bijections between these coefficients and their parametrizations, allowing matching procedures of estimation (a lot like moments matching algorithm for fitting standard random variables). 

There would have been many more things to say about dependence structures and copulas. This short literature review, however, seems enough to understand the matters of this package. We refer the interested reader to the numerous references.




```@bibliography
Pages = ["getting_started.md"]
Canonical = false
```
