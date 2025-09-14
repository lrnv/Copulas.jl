```@meta
CurrentModule = Copulas
```

# Dependence measures


The copula of a random vector fully encodes its dependence structure. 
However, copulas are infinite-dimensional objects and interpreting their properties can be difficult as the dimension increases. 
Therefore, the literature has introduced quantifications of the dependence structure that may be used as univariate (imperfect but useful) summaries of certain copula properties. 
We implement the most well-known ones in this package. 

## Kendall's τ and Spearman's ρ

!!! definition "Definition (Kendall' τ):"
    For a copula $C$ with a density $c$, **regardless of its dimension $d$**, Kendall's τ is defined as: 

    $$\tau = 4 \int C(\bm u) \, c(\bm u) \;d\bm u -1$$

!!! definition "Definition (Spearman's ρ):"
    For a copula $C$ with a density $c$, **regardless of its dimension $d$**, Spearman's ρ is defined as: 

    $$\rho = 12 \int C(\bm u) d\bm u -3.$$

These two dependence measures are most meaningful in the bivariate case, and we sometimes refer to the Kendall's matrix or the Spearman's matrix for the collection of bivariate coefficients associated with a multivariate copula. 
We thus provide two different interfaces:
* `Copulas.τ(C::Copula)` and `Copulas.ρ(C::Copula)`, providing true multivariate Kendall taus and Spearman rhos
* `StatsBase.corkendall(C::Copula)` and `StatsBase.corspearman(C::Copula)` provide matrices of bivariate Kendall taus and Spearman rhos. 

Thus, for a given copula `C`, the theoretical dependence measures can be obtained by `τ(C), ρ(C)` (for the multivariate versions) and `corkendall(C), corspearman(C)` (for the matrix versions).
Similarly, empirical versions of these metrics can be obtained from a matrix of observations `data` of size `(d,n)` by  `Copulas.τ(data)`, `Copulas.ρ(data)`, `StatsBase.corkendall(data)` and `StatsBase.corspearman(data)`.

!!! note "Ranges of $\tau$ and $\rho$."
    Kendall's $\tau$ and Spearman's $\rho$ belong to $[-1, 1]$. They are equal to :
    * 0 if and only if the copula is a `IndependentCopula`.
    * -1 is and only if the copula is a `WCopula`.
    * 1 if and only if the copula is a `MCopula`. 
    
    They do not depend on the marginals. This is why we say that they measure the 'strength' of the dependency.



!!! todo "Work in progress"
    The package implements generic version of the dependence metrics, but some families have specific formulas for the Kendall $\tau$ and Spearman $\rho$, allowing faster implementations. However, all the potential fast-paths are not implemented yet. If you feel a specific method for a certain copula is missing, do not hesitate to open an issue !

    Moreover, many copula estimators are based on the relationship between parameters and these coefficients (see e.g., [genest2011,fredricks2007,derumigny2017](@cite)), but once again our implementation is not complete yet. 

Here is for example the relationship between the Kendall $\tau$ and the parameter of a Clayton copula:  

```@example dep
using Copulas, Plots, Distributions
θs = -1:0.1:5
τs = [Copulas.τ(ClaytonCopula(2, θ)) for θ in θs]
plot(θs, τs; xlabel="θ", ylabel="τ", title="θ -> τ for bivariate Clayton", legend=false)
```

Remark the clear and easy to exploit bijection. 


## Tail dependency

Many people are interested in the tail behavior of their dependence structures. Tail coefficients summarize this tail behavior.

!!! definition "Definition (Tail dependency):"
    For a copula $C$, we define the upper tail statisticss (when they exist):

    ```math
    \begin{align}
        \lambda_U(u) &= \frac{1 - 2u - C(u,..,u)}{1- u}\\
        \lambda_U &= \lim\limits_{u \to 1_-} \lambda_U(u) \in [0,1]\\
        \chi_U(u) &= \frac{2 \ln(1-u)}{\ln(1-2u+C(u,...,u))} -1\\
        \chi_U &= \lim\limits_{u \to 1_-} \chi_U(u) \in [-1,1]
    \end{align}
    ```

    Simetric tools can be constructed for the lower tail: 

    ```math
    \begin{align}
        \lambda_L(u) &= \frac{C(u,..,u)}{u}\\
        \lambda_L &= \lim\limits_{u \to 0_+} \lambda_L(u) \in [0,1]\\
        \chi_L(u) &= \frac{2 \ln(u)}{\ln(C(u,...,u))} -1\\
        \chi_L &= \lim\limits_{u \to 0_+} \chi(u) \in [-1,1]
    \end{align}
    ```

    
When $\lambda_U > 0$ (resp $\lambda_L > 0$), we say that there is strong upper (resp lower) tail dependency, and $\chi_U = 1$ (resp $\chi_L = 1$).
When $\lambda_U > 0$ (resp $\lambda_L > 0$), if furthermore $\chi_U \neq 0$ (resp $\chi_L \neq 0$), we say that there is weak upper tail dependency.
Otherwise we ay there is no tail dependency. Thus, the graph of $\lambda_L(u), \chi_L(u)$ over $[0, \frac{1}{2}]$, and the graph of  $\lambda_U(u), \chi_U(u)$ over $[\frac{1}{2},1]$ are usefull tools to diagnose the potential limits.

```@example chi_graph
using Copulas, Distributions, Plots
λᵤ(C::Copulas.Copula{d}, u) where d = (1 - 2u - cdf(C, fill(u,d)))/(1-u)
χᵤ(C::Copulas.Copula{d}, u) where d = 2 * log1p(- u) / log1p(- 2u + cdf(C, fill(u,d))) - 1

C = GumbelCopula(2, 2.5)
plot(0.9:0.001:0.999, Base.Fix1(λᵤ, C); xlabel="u", label="λᵤ(u)", title="Graph of λᵤ(u) and χᵤ(u)  for Gumbel Copula")
plot!(0.9:0.001:0.999, Base.Fix1(χᵤ, C); label="χᵤ(u)")
```

```@example chi_graph
C = ClaytonCopula(2, 2.5)
plot(0.9:0.001:0.999, Base.Fix1(λᵤ, C); xlabel="u", label="λᵤ(u)", title="Graph of λᵤ(u) and χᵤ(u) for Clayton Copula")
plot!(0.9:0.001:0.999, Base.Fix1(χᵤ, C); label="χᵤ(u)")
```

All these coefficients quantify the behavior of the dependence structure, generally or in the extremes, and are therefore widely used in the literature either as verification tools to assess the quality of fits, or even as parameters.
Many parametric copula families have simple surjections, injections, or even bijections between these coefficients and their parametrization, allowing matching procedures of estimation (similar to moment matching algorithms for fitting standard random variables).


!!! todo "Work in progress"
    The formalization of an interface for obtaining the tail dependence coefficients of copulas is still a work in progress in the package. Do not hesitate to reach us on GitHub if you want to discuss it!



```@bibliography
Pages = [@__FILE__]
Canonical = false
```