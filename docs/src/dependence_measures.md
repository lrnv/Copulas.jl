# Measures of dependency


Although the copula is an object that summarizes completely the dependence structure of any random vector, it is an infinite dimensional object and the interpretation of its properties can be difficult when the dimension gets high. Therefore, the literature has come up with some quantification of the dependence structure that might be used as univariate summaries, of course imperfect, of certain properties of the copula at hand. 


!!! note "Unfinished work"
    Unfortunately these dependence measures are not yet well-specified in the package and their implementation is experimental for the moment. These functions might change in the future, in particular see https://github.com/lrnv/Copulas.jl/issues/134 for future improvements. 


## Kendall's Tau 

> **Definition (Kendall' τ):** For a copula $C$ with a density $c$, Kendall's τ is defined as: 
> 
>$$\tau = 4 \int C(\bm u) \, c(\bm u) \;d\bm u -1$$

Kendall's tau can be obtained through `τ(C::Copula)`. Its value only depends on the dependence structure and not the marginals. 

!!! warn "Multivariate case"
    There exists several multivariate extensions of Kendall's tau. The one implemented here is the one we just defined what ever the dimension $d$, be careful as the normalization might differ from other places in the literature.



## Spearman's Rho 

> **Definition (Spearman's ρ):** For a copula $C$ with a density $c$, the Spearman's ρ is defined as: 
>
> $$\rho = 12 \int C(\bm u) d\bm u -3.$$

Spearman's Rho can be obtained through `ρ(C::Copula)`. Its value only depends on the dependence structure and not the marginals. 

!!! warn "Multivariate case"
    There exists several multivariate extensions of Spearman's rho. The one implemented here is the one we just defined what ever the dimension $d$, be careful as the normalization might differ from other places in the literature.

!!! note "Specific values of tau and rho"
    Kendall's $\tau$ and Spearman's $\rho$ have values between -1 and 1, and are -1 in case of complete anticomonotony and 1 in case of comonotony. Moreover, they are 0 in case of independence. This is 
    why we say that they measure the 'strength' of the dependency.

!!! tip "More-that-bivariate cases"
    These two dependence measures make more sense in the bivariate case than in other cases, and therefore we sometimes refer to the Kendall's matrix or the Spearman's matrix for the collection of bivariate coefficients associated to a multivariate copula. Many copula estimators are based on these coefficients, see e.g., [genest2011,fredricks2007,derumigny2017](@cite).

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