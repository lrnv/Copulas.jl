```@meta
CurrentModule = Copulas
```

# General Discussion

## Pseudo-observations


Through the statistical process leading to the estimation of copulas, one usually observes the data and information on the marginals scale and not on the copula scale. This discrepancy between the observed information and the modeled distribution must be taken into account. A key concept is that of pseudo-observations. 

>**Definition (Pseudo-observations):** If $\bm x \in \mathbb R^{N\times d}$ is an $N$-sample of a $d$-variate real-valued random vector $\bm X$, then the pseudo-observations are the normalized ranks of the marginals of $\bm x$, defined as : 
>
> $$\bm u \in \mathbb [0,1]^{N\times d}:\; u_{i,j} = \frac{\mathrm{Rank}(x_{i,j},\,\bm x_{\cdot,j})}{N+1} = \frac{1}{N+1}\sum_{k=1}^N \mathbb 1_{x_{k,j} \le x_{i,j}},$$
>
> where $\mathrm{Rank}(y,\bm x)  = \sum\limits_{x_i \in \bm x} \mathbb 1_{x_i \le y}$.


In `Copulas.jl`, we provide a function `pseudos` that implement this transformation directly. 

```@docs
pseudos
```

## Deheuvel's empirical copula

From these pseudo-observations, an empirical copula is defined as follows:

>**Definition (Deheuvel's empirical copula [deheuvels1979](@cite)):** The empirical distribution function of the normalized ranks,
>
>$$\hat{C}_N(\bm u) = \frac{1}{N} \sum_{i=1}^N \mathbb 1_{\bm u_i \le \bm u},$$ is called the empirical copula function.

>**Theorem (Exhaustivity and consistency [deheuvels1979](@cite)):** $\hat{C}_N$ is an exhaustive estimator of $C$, and moreover for any normalizing constants $\{\phi_N, N\in \mathbb N\}$ such that $\lim\limits_{N \to \infty} \phi_N \sqrt{N^{-1}\ln \ln N} = 0$, 
>
>$$\lim\limits_{N\to\infty} \phi_N \sup_{\bm u \in [0,1]^d} \lvert\hat{C}_N(\bm u) - C(\bm u) \rvert = 0 \text{ a.s.}$$

$\hat{C}_N$ then converges (weakly) to $C$, the true copula of the random vector $\bm X$, when the number of observations $N$ goes to infinity. 

!!! note "The empirical copula is not a true copula"
    Despite its name, $\hat{C}_N$ is not a copula since it does not have uniform marginals. Be carrefull. 

In the package, this copula is implemented as the `EmpiricalCopula`: 

```@docs; canonical=false
EmpiricalCopula
```

## Beta copula


The empirical copula function is not a copula. An easy way to fix this problem is to smooth out the marginals with beta distribution functions: 


>**Definition (Beta Copula [segers2017](@cite)):** Denoting $F_{n,r}(x) = \sum_{s=r}^n \binom{n}{s} x^s(1-x)^{n-s}$ the distribution function of a $\mathrm{Beta}(r,n+1-r)$ random variable, the function
>
> $$\hat{C}_N^\beta : \bm x \mapsto \frac{1}{N} \sum_{i=1}^N \prod\limits_{j=1}^d F_{n,(N+1)u_{i,j}}(x_j)$$ is a genuine copula, called the Beta copula. 

>**Property (Proximity of $\hat{C}_N$ and $\hat{C}_N^\beta$ [segers2017](@cite)):**
>
> $$\sup\limits_{\bm u \in [0,1]^d} \lvert \hat{C}_N(\bm u) - \hat{C}_N^\beta(\bm u) \rvert \le d\left(\sqrt{\frac{\ln n}{n}} + \sqrt{\frac{1}{n}} + \frac{1}{n}\right)$$

!!! note "Not implemeted yet!"
    Do not hesitate to come talk on [our github](https://github.com/lrnv/Copulas.jl) !

## Bernstein Copula

Bernstein copula are simply another smoothing of the empirical copula using Bernstein polynomials. 

!!! note "Not implemeted yet!"
    Do not hesitate to come talk on [our github](https://github.com/lrnv/Copulas.jl) !

## Checkerboard Copulas

There are other nonparametric estimators of the copula function that are true copulas. Of interest to our work is the Checkerboard construction (see [cuberos2019,mikusinski2010](@cite)), detailed below.

First, for any $\bm m \in \mathbb N^d$, let $\left\{B_{\bm i,\bm m}, \bm i < \bm m\right\}$ be a partition of the unit hypercube defined by

$$B_{\bm i, \bm m} = \left]\frac{\bm i}{\bm m}, \frac{\bm i+1}{\bm m}\right].$$

Furthermore, for any copula $C$ (or more generally distribution function $F$), we denote $\mu_{C}$ (resp $\mu_F$) the associated measure.  For example, for the independence copula $Pi$, $\mu_{\Pi}(A) = \lambda(A \cup [\bm 0, \bm 1])$ where $\lambda$ is the Lebesgue measure.

>**Definition (Empirical Checkerboard copulas [cuberos2019](@cite)):** Let $\bm m \in \mathbb N^d$. The $\bm m$-Checkerboard copula $\hat{C}_{N,\bm m}$, defined by 
>
>$$\hat{C}_{N,\bm m}(\bm x) = \bm m^{\bm 1} \sum_{\bm i < \bm m} \mu_{\hat{C}_N}(B_{\bm i, \bm m}) \mu_{\Pi}(B_{\bm i, \bm m} \cap [0,\bm x]),$$
>
>is a genuine copula as soon as $m_1,...,m_d$ all divide $N$.


>**Property (Consistency of $\hat{C}_{N,\bm m}$ [cuberos2019](@cite):** If all $m_1,..,m_d$ divide $N$,
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
where we intend $\bm m ]\bm a, \bm b] = ] \bm m \bm a, \bm m \bm b]$ (products between vectors are componentwise).

This allows for an easy generalization in the framework of patchwork copulas: 

> **Definition (Patchwork copulas [durante2012,durante2013,durante2015](@cite):**) Let $\bm m \in \mathbb N^d$ all divide $N$, and let $\mathcal C = \{C_{\bm i}, \bm i < \bm m\}$ be a given collection of copulas. The distribution function
>
>$$\hat{C}_{N,\bm m, \mathcal C}(\bm x) = \sum_{\bm i < \bm m} \mu_{\hat{C}_N}(B_{\bm i, \bm m}) \mu_{C_{\bm i}}(\bm m B_{\bm i, \bm m}(\bm x))$$
>is a copula. 

In fact, replacing $\hat{C}_N$ by any copula in the patchwork construct still yields a genuine copula, with no more conditions that all components of $\bm m$ divide $N$. The Checkerboard grids are practical in the sense that computations associated to a Checkerboard copula can be really fast: if the grid is large, the number of boxes is small, and otherwise if the grid is very refined, many boxes are probably empty. On the other hand, the grid is fixed a priori, see [laverny2020](@cite) for a construction with an adaptive grid.

Convergence results for this kind of copulas can be found in [durante2015](@cite), with a slightly different parametrization. 

!!! note "Not implemeted yet!"
    Do not hesitate to come talk on [our github](https://github.com/lrnv/Copulas.jl) !


```@bibliography
Pages = ["generalities.md"]
Canonical = false
```