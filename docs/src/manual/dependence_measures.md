```@meta
CurrentModule = Copulas
```

# [Dependence measures](@id dep_metrics)

The copula of a random vector fully encodes its dependence structure. 
However, copulas are infinite-dimensional objects and interpreting their properties can be difficult as the dimension increases. 
Therefore, the literature has introduced quantifications of the dependence structure that may be used as univariate (imperfect but useful) summaries of certain copula properties. 
We implement the most well-known ones in this package. 

## Core dependence metrics τ, ρ, β, γ and ι

!!! definition "Kendall' τ"
    For a copula $C$ with a density $c$, **regardless of its dimension $d$**, Kendall's τ is defined as: 

    $$\tau = \frac{2^d}{2^{d-1} - 1} \int C(\boldsymbol u) \, c(\boldsymbol u) \;d\boldsymbol u - \frac{1}{2^{d-1}-1}$$

!!! definition "Spearman's ρ"
    For a copula $C$ with a density $c$, **regardless of its dimension $d$**, Spearman's ρ is defined as: 

    $$\rho = \frac{2^d * (d+1)}{2^d -d-1} \int C(\boldsymbol u) d\boldsymbol u - \frac{d+1}{2^d - (d+1)}.$$

!!! definition "Definition (Blomqvist's β):"
    For a copula $C$ with a density $c$, **regardless of its dimension $d$**, Blomqvist's β is defined as: 

    $$\beta = \frac{2^{d-1}}{2^{d-1} -1} \left(C(\frac{\boldsymbol{1}}{\boldsymbol{2}}) + \bar{C}(\frac{\boldsymbol{1}}{\boldsymbol{2}})\right) - \frac{1}{2^{d-1} - 1}.$$

    where $\bar{C}$ is the survival copula associated with $C$. 

!!! definition "Definition (Gini's γ):"
    For a copula $C$ with a density $c$, **regardless of its dimension $d$**, the multivariate Gini’s gamma is defined as [behboodian2007multivariate](@cite):

    $$\gamma = \frac{1}{b(d)-a(d)}\left[\int_{[0,1]^d}\{A(\boldsymbol{u}) + \bar{A}(\boldsymbol{u})\}dC(\boldsymbol{u}) - a(d)\right],$$

    with

    $$A(u)=\frac{1}{2}\left(\min(u)+\max\Big(\textstyle\sum_{i=1}^d u_i-d+1,0\Big)\right), \quad \bar{A}(u)=\frac{1}{2}\left(1-\max(u)+\max\Big(1-\sum_{i=1}^d u_i,0\Big)\right),$$

    where the normalizing constants depend only on the dimension $d$ and match our implementation:

    $$a(d) = \frac{1}{d+1} + \frac{1}{(d+1)!} \quad \text{(independence)}, \qquad b(d) = \frac{2 + 4^{\,1-d}}{3} \quad \text{(comonotonicity)}.$$

!!! definition "Definition (Copula entropy ι):"
    For a copula $C$ with density $c$, the copula entropy is

    $$\iota(C) = - \int_{[0,1]^d} c(u) \log c(u)\,du.$$

    It satisfies $I(X_1,\dots,X_d) = -\iota(C)$ (see [ma2011mutual](@cite)).


These dependence measures are very common when $d=2$, and a bit less when $d > 2$. We sometimes refer to the Kendall's matrix or the Spearman's matrix for the collection of bivariate coefficients associated with a multivariate copula. 
We thus provide two different interfaces:
* `Copulas.τ()`, `Copulas.ρ()`, `Copulas.β()`, `Copulas.γ()` and `Copulas.ι()` provide the upper formulas, yielding a scalar whatever the dimension of the copula.
* `StatsBase.corkendall()`, `StatsBase.corspearman()`, `Copulas.corblomqvist()`, `Copulas.corgini()` and `Copulas.corentropy()` provide matrices of pairwise dependence metrics. 
* All these functions have methods for a single argument `C::Copula`, yielding theoretical quantities, and for a dataset `data::AbstractMatrix` yielding empirical estimates.

For historical reasons, `τ(data)`, `ρ(data)`, `β(data)`, `γ(data)`, `ι(data)` require `(d,n)`-shaped datasets (observations or pseudo-observations), while  `corkendall(data)`, `corspearman(data)`, `corblomqvist(data)`, `corgini(data)`, and `corentropy(data)` do require transposed `(n,d)`-shaped datasets. 

!!! note "Ranges of τ, ρ, β and γ."
    Kendall's $\tau$, Spearman's $\rho$, Blomqvist's $\beta$ and Gini's $\gamma$ all belong to $[-1, 1]$. They are equal to :
    * 0 if and only if the copula is a `IndependentCopula`.
    * -1 is and only if the copula is a `WCopula`.
    * 1 if and only if the copula is a `MCopula`. 
    
    They do not depend on the marginals. This is why we say that they measure the 'strength' of the dependency.

!!! todo "Work in progress"
    The package implements generic version of the dependence metrics, but some families have faster versions (closed form formulas or better integration paths). 
    However, all the potential fast-paths are not implemented yet. If you feel a specific method for a certain copula is missing, do not hesitate to open an issue !

Many copula estimators are based on the relationship between parameters and these coefficients (see e.g., [genest2011,fredricks2007,derumigny2017](@cite)).
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

!!! definition "Tail dependency"
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

The package provides both theoretical limits (for a given copula object) and empirical estimators (from data matrices).  
In addition, pairwise tail-dependence matrices can be computed for multivariate samples.  

* **Theoretical λ**: `Copulas.λₗ(C::Copula)` and  `Copulas.λᵤ(C::Copula)`
  Shortcuts: `Copulas.λₗ(C)`, `Copulas.λᵤ(C)`  

* **Empirical λ**: `Copulas.λₗ(U::AbstractMatrix; p=1/√m)` and `Copulas.λᵤ(U::AbstractMatrix; p=1/√m)`

* **Pairwise λ-matrix**: `Copulas.coruppertail(data; method=:SchmidtStadtmueller, p=1/√m)` and `Copulas.corlowertail(data; method=:SchmidtStadtmueller, p=1/√m)`  

These follow the approach of Schmidt & Stadtmüller (see [schmidt2006non](@cite)).



## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```