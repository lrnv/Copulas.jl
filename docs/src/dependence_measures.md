```@meta
CurrentModule = Copulas
```

# Measures of dependency


The copula of a random vector fully encodes its dependence structure. 
However, copulas are infinite dimensional object and the interpretation of their properties can be difficult when the dimension gets high. 
Therefore, the literature has come up with some quantification of the dependence structure that might be used as univariate summaries, of course imperfect, of certain properties of the copula at hand. 
We implement the few most known ones in this package. 

## Kendall's τ and Spearman's ρ: bivariate and multivariate cases

!!! definition "Definition (Kendall' τ):"
    For a copula $C$ with a density $c$, **whatever its dimension $d$**, its Kendall's τ is defined as: 

    $$\tau = 4 \int C(\bm u) \, c(\bm u) \;d\bm u -1$$

!!! definition "Definition (Spearman's ρ):"
    For a copula $C$ with a density $c$, **whatever its dimension $d$**, the Spearman's ρ is defined as: 

    $$\rho = 12 \int C(\bm u) d\bm u -3.$$

These two dependence measures make more sense in the bivariate case than in other cases, and therefore we sometimes refer to the Kendall's matrix or the Spearman's matrix for the collection of bivariate coefficients associated to a multivariate copula. 
We thus provide two different interfaces:
* `Copulas.τ(C::Copula)` and `Copulas.ρ(C::Copula)`, providing true multivariate Kendall taus and Spearman rhos
* `StatsBase.corkendall(C::Copula)` and `StatsBase.corspearman(C::Copula)` provides on the other hand matrices of bivariate Kendall taus and Spearman rhos. 

Thus, for a given copula `C`, the theoretical dependence measures can be obtained by `τ(C), ρ(C)` (for the multivariate versions) and `StatsBase.corkendall(C), StatsBase.corspearman(C)` (for the matrix versions).
Similarly, from `StatsBase`, empirical versions of the matrices of dependence measures car be obtained from a matrix of observations `data::Matrix{n,d}` by `StatsBase.corkendall(data)` and `StatsBase.corspearman(data)`.

!!! note "Specific values of tau and rho"
    Kendall's $\tau$ and Spearman's $\rho$ have values between -1 and 1, and are -1 in case of complete anticomonotony and 1 in case of comonotony. 
    Moreover, they are 0 in case of independence. 
    Moreover, their values only depends on the dependence structure and not the marginals. 
    This is why we say that they measure the 'strength' of the dependency.

Many copula estimators are based on these coefficients, see e.g., [genest2011,fredricks2007,derumigny2017](@cite).

A few remarks on the state of the implementation:

* Bivariate elliptical cases use $\tau = asin(\rho) * 2 / \pi$ where $\rho$ is the spearman correlation, as soon as the radial part does not have atoms. See [fang2002meta](@cite) for historical credits and [lindskog2003kendall](@cite) for a good review.
* Many Archimedean copulas have specific formulas for their Kendall tau's, but generic ones use [mcneil2009](@cite).
* Extreme values copulas have a specific generic method.
* Generic copulas use directly the upper formula. 
* Estimation is done for some copulas wia inversion of Kendall's tau or Spearman's rho.

!!! note "Spearman's rho: work in progress"
    If most of the efficient family-specific formulas for Kendall's tau are already implemented in the package, Spearman's $\rho$'s tend to leverage the generic (slow) implementation much more. If you feel like a specific method for a certain copula is missing, do not hesitate to open an issue !

## Tail dependency

Many people are interested in the tail behavior of their dependence structures. Tail coefficients summarize this tail behavior.

!!! definition "Definition (Tail dependency):"
    For a copula $C$, we define (when they exist):
    ```math
     \begin{align}
       \lambda &= \lim\limits_{u \to 1} \frac{1 - 2u - C(u,..,u)}{1- u} \in [0,1]\\
       \chi(u) &= \frac{2 \ln(1-u)}{\ln(1-2u-C(u,...,u))} -1\\
       \chi &= \lim\limits_{u \to 1} \chi(u) \in [-1,1]
     \end{align}
    ```
    When $\lambda > 0$, we say that there is a strong upper tail dependency, and $\chi = 1$. When $\lambda = 0$, we say that there is no strong upper tail dependency, and if furthermore $\chi \neq 0$ we say that there is weak upper tail dependency.

The graph of $u \to \chi(u)$ over $[\frac{1}{2},1]$ is an interesting tool to assess the existence and strength of the tail dependency. The same kind of tools can be constructed for the lower tail. 

!!! note "Tail dependencies: WIP"
    The formalization of an interface for obtaining the tail dependence coefficients of copulas is still a work a in progress in the package. Do not hesitate to reach us on github if you want to discuss it!


All these coefficients quantify the behavior of the dependence structure, generally or in the extremes, and are therefore widely used in the literature either as verification tools to assess the quality of fits, or even as parameters.
Many parametric copulas families have simple surjections, injections, or even bijections between these coefficients and their parametrization, allowing matching procedures of estimation (a lot like moments matching algorithm for fitting standard random variables).


```@bibliography
Pages = [@__FILE__]
Canonical = false
```