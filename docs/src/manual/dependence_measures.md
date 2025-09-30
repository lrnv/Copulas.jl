```@meta
CurrentModule = Copulas
```

# Dependence measures


The copula of a random vector fully encodes its dependence structure. 
However, copulas are infinite-dimensional objects and interpreting their properties can be difficult as the dimension increases. 
Therefore, the literature has introduced quantifications of the dependence structure that may be used as univariate (imperfect but useful) summaries of certain copula properties. 
We implement the most well-known ones in this package. 

## Kendall's τ, Spearman's ρ, Blomqvist's β and Gini's γ

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
    For a copula $C$ with a density $c$, **regardless of its dimension $d$**, the multivariate Gini’s gamma is defined as (Behboodian–Dolati–Úbeda, 2007):

    $$\gamma = \frac{1}{b(d)-a(d)}\left[\int_{[0,1]^d}\{A(\boldsymbold{u}) + \bar{A}(\boldsymbold{u})\}dC(\boldsymbold{u}) - a(d)\right],$$

    with

    $$A(u)=\frac{1}{2}\left(\min(u)+\max\Big(\textstyle\sum_{i=1}^d u_i-d+1,0\Big)\right), \quad \bar{A}(u)=\frac{1}{2}\left(1-\max(u)+\max\Big(1-\sum_{i=1}^d u_i,0\Big)\right),$$

    while $a_d, b_d$ are normalizing constants depending only on the dimension $d$.


These dependence measures are very common when $d=2$, and a bit less when $d > 2$. We sometimes refer to the Kendall's matrix or the Spearman's matrix for the collection of bivariate coefficients associated with a multivariate copula. 
We thus provide two different interfaces:
* `Copulas.τ(C::Copula)`, `Copulas.ρ(C::Copula)`, `Copulas.β(C::Copula)`, `Copulas.γ(C::Copula)` provide the upper formulas.
* `StatsBase.corkendall(data)`, `StatsBase.corspearman(data)`, `Copulas.corblomqvist(data)`, `Copulas.corgini(data)` provide matrices of bivariate versions. 

Thus, for a given copula `C`, the theoretical dependence measures can be obtained by `τ(C), ρ(C), β(C)` (for the multivariate versions) and `corkendall(C), corspearman(C), corblomqvist(C)` (for the matrix versions).
Similarly, empirical versions of these metrics can be obtained from a matrix of observations `data` of size `(d,n)` by  `Copulas.τ(data)`, `Copulas.ρ(data)`, `Copulas.β(data)`, `Copulas.γ(data)`, `StatsBase.corkendall(data)`, `StatsBase.corspearman(data)`, `Copulas.corblomqvist(data)` and `Copulas.corgini(data)`.

!!! note "Ranges of $\tau$, $\rho$, $\beta$ and $\gamma$."
    Kendall's $\tau$, Spearman's $\rho$, Blomqvist's $\beta$ and Gini's $\gamma$ all belong to $[-1, 1]$. They are equal to :
    * 0 if and only if the copula is a `IndependentCopula`.
    * -1 is and only if the copula is a `WCopula`.
    * 1 if and only if the copula is a `MCopula`. 
    
    They do not depend on the marginals. This is why we say that they measure the 'strength' of the dependency.

!!! todo "Work in progress"
    The package implements generic version of the dependence metrics, but some families have faster versions (closed form formulas or better integration paths). 
    However, all the potential fast-paths are not implemented yet. If you feel a specific method for a certain copula is missing, do not hesitate to open an issue !

    Moreover, many copula estimators are based on the relationship between parameters and these coefficients (see e.g., [genest2011,fredricks2007,derumigny2017](@cite)), but once again our implementation is not complete yet. 

Here is for example the relationship between the Kendall $\tau$ and the parameter of a Clayton copula:  

```@example dep
using Copulas, Plots, Distributions
θs = -1:0.1:5
τs = [Copulas.τ(ClaytonCopula(2, θ)) for θ in θs]
plot(θs, τs; xlabel="θ", ylabel="τ", title="θ -> τ for bivariate Clayton", legend=false)
```

Remark the clear and easy to exploit bijection. 

!!! info "Efficiency note"
    In practice, **Gini’s γ** is the most efficient dependence measure in our implementation (microseconds, no allocations).  
    **Kendall’s τ** is the most computationally expensive (\(O(n^2)\)), while **Spearman’s ρ** and **Blomqvist’s β** are intermediate.  
    For pairwise matrices, `corgini` is faster than both `corblomqvist` and the classical `corspearman`/`corkendall`.

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

### Implementations in `Copulas.jl`

The package provides both theoretical limits (for a given copula object) and empirical estimators (from data matrices).  
In addition, pairwise tail-dependence matrices can be computed for multivariate samples.  

* **Theoretical λ**: `λ(C::Copula; t=:lower|:upper, ε=1e-10)`  
  Shortcuts: `λₗ(C)`, `λᵤ(C)`  

* **Empirical λ**: `λ(U::AbstractMatrix; t=:lower|:upper, p=1/√m)`  
  Shortcuts: `λₗ(U)`, `λᵤ(U)`  

* **Pairwise λ-matrix**: `cortail(data; t=:lower|:upper, method=:SchmidtStadtmueller, p=1/√m)`  

These follow the approach of Schmidt & Stadtmüller [schmidt2006non](@cite).
 

!!! todo "Work in progress"
    The formalization of an interface for obtaining the tail dependence coefficients of copulas is still a work in progress in the package. Do not hesitate to reach us on GitHub if you want to discuss it!

## Copula entropy

!!! definition "Definition (Copula entropy):"
    For a copula $C$ with density $c$, the copula entropy is the Shannon entropy of $c$:

    $$H(C) = - \int_{[0,1]^d} c(u) \log c(u) ,du.$$
    Ma & Sun (2011) proved that the mutual information of a random vector equals the negative copula entropy:
    $$
    I(X_1,\dots,X_d) = - H(C).
    $$

**Basic Properties.**
- $H(C)\le 0$ with equality $H(C)=0$ if and only if $C$ is the `IndependentCopula` (because $c\equiv 1$).
- For **singular** copulas (without density), $H(C)=-\infty$.
- Since $I=-H$, the larger the $I$ $\Rightarrow$, the greater the dependence (linear, nonlinear, tailing, etc.).

### Implementations in `Copulas.jl`

* **Parametric (Monte Carlo)**: `entropy(C::Copula; nmc=100_000)` Returns `(; H, I=-H, r)`, with $r=\sqrt{\max(0,1-e^{2H})}$ as rescaled by $[0,1]$.

* **Non-parametric (kNN)**: `entropy(U::AbstractMatrix; k=5, p=Inf)` 
Kozachenko–Leonenko estimator on **pseudo-observations** $U\in(0,1)^d$. 
Typical parameters: $k\in[5,15]$; norm $p\in\{1,2,\infty\}$.

* **Pairwise version**: `corentropy(data; k=5, p=Inf, signed=false)`
Matrices of $(H,I,r)$ for all pairs; `signed=true` multiplies $r$ by $\operatorname{sign}(\tau)$.

$r$ y su versión “signed” **no son PSD** (son re-escalas de $H$).

!!! note "Efficiency"

    While $\tau$, $\rho$, $\beta$ or $\gamma$ can be computed in microseconds, entropy-based measures are much slower due to simulation or kNN search.
    Benchmarks confirm this: `γ` is fastest, whereas `entropy` and `corentropy` are orders of magnitude slower.
    They are therefore recommended mainly for **validation, model selection, or feature screening**, not routine use.


```@bibliography
Pages = [@__FILE__]
Canonical = false
```