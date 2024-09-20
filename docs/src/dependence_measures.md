# Measures of dependency


Although Copulas summarize completely the dependence structure of corresponding random vector, it is an infinite dimensional object and the interpretation of its properties can be difficult when the dimension gets high. Therefore, the literature has come up with some quantification of the dependence structure that might be used as univariate summaries, of course imperfect, of certain properties of the copula at hand. 

## Multivariate and Bivariate Kendall's Tau and Spearman rhos. 

> **Definition (Kendall' τ):** For a copula $C$ with a density $c$, **whatever its dimension $d$**, its Kendall's τ is defined as: 
> 
>$$\tau = 4 \int C(\bm u) \, c(\bm u) \;d\bm u -1$$

> **Definition (Spearman's ρ):** For a copula $C$ with a density $c$, **whatever its dimension $d$**, the Spearman's ρ is defined as: 
>
> $$\rho = 12 \int C(\bm u) d\bm u -3.$$

These two dependence measures make more sense in the bivariate case than in other cases, and therefore we sometimes refer to the Kendall's matrix or the Spearman's matrix for the collection of bivariate coefficients associated to a multivariate copula. We thus provide two different interfaces, one internal through `Copulas.τ(C::Copula)` and `Copulas.ρ(C::Copula)`, providing true multivariate Kendall taus and Spearman rhos, and one implementing `StatsBase.corkendall(C::Copula)` and `StatsBase.corspearman(C::Copula)` for matrices of bivariate dependence measures. 

Thus, for a given copula `C`, its Kendall's tau can be obtained through `τ(C::Copula)` and its Spearman's Rho through `ρ(C::Copula)`. 

In the literature however, for multivariate copulas, the matrices of bivariate Kendall taus and bivariate Spearman rhos are sometimes used, and this is these matrices that are obtained by `StatsBase.corkendall(data)` and `StatsBase.corspearman(data)` where `data::Matrix{n,d}` is a matrix of observations. The corresponding theoretical values for copulas in this package can be obtained through the `StatsBase.corkendall(C::Copula)` and `StatsBase.corspearman(C::Copula)` interface.


!!! note "Specific values of tau and rho"
    Kendall's $\tau$ and Spearman's $\rho$ have values between -1 and 1, and are -1 in case of complete anticomonotony and 1 in case of comonotony. Moreover, they are 0 in case of independence. Moreover, their values only depends on the dependence structure and not the marginals. This is why we say that they measure the 'strength' of the dependency.

Many copula estimators are based on these coefficients, see e.g., [genest2011,fredricks2007,derumigny2017](@cite).


## Tail dependency

Many people are interested in the tail behavior of their dependence structures. Tail coefficients summarize this tail behavior.

>**Definition (Tail dependency):** For a copula $C$, we define (when they exist):
> ```math
>  \begin{align}
>    \lambda &= \lim\limits_{u \to 1} \frac{1 - 2u - C(u,..,u)}{1- u} \in [0,1]\\
>    \chi(u) &= \frac{2 \ln(1-u)}{\ln(1-2u-C(u,...,u))} -1\\
>    \chi &= \lim\limits_{u \to 1} \chi(u) \in [-1,1]
>  \end{align}
>```
> When $\lambda > 0$, we say that there is a strong upper tail dependency, and $\chi = 1$. When $\lambda = 0$, we say that there is no strong upper tail dependency, and if furthermore $\chi \neq 0$ we say that there is weak upper tail dependency.

The graph of $u \to \chi(u)$ over $[\frac{1}{2},1]$ is an interesting tool to assess the existence and strength of the tail dependency. The same kind of tools can be constructed for the lower tail. 

All these coefficients quantify the behavior of the dependence structure, generally or in the extremes, and are therefore widely used in the literature either as verification tools to assess the quality of fits, or even as parameters. Many parametric copulas families have simple surjections, injections, or even bijections between these coefficients and their parametrization, allowing matching procedures of estimation (a lot like moments matching algorithm for fitting standard random variables).


```@bibliography
Pages = ["dependence_measures.md"]
Canonical = false
```