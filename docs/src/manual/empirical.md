```@meta
CurrentModule = Copulas
```

# Empirical models

## Pseudo-observations


Through the statistical process leading to the estimation of copulas, one usually observes the data and information on the marginals scale and not on the copula scale. This discrepancy between the observed information and the modeled distribution must be taken into account. A key concept is that of pseudo-observations. 

!!! definition "Pseudo-observations"
    If $\boldsymbol x \in \mathbb{R}^{N \times d}$ is an $N$-sample of a $d$-variate real-valued random vector $\boldsymbol X$, then the pseudo-observations are the normalized ranks of the marginals of $\boldsymbol x$, defined as:

    $$\boldsymbol u \in [0,1]^{N \times d}:\; u_{i,j} = \frac{\mathrm{Rank}(x_{i,j}, \boldsymbol x_{\cdot,j})}{N+1} = \frac{1}{N+1} \sum_{k=1}^N \mathbb{1}_{x_{k,j} \le x_{i,j}},$$

    where $\mathrm{Rank}(y, \boldsymbol x) = \sum_{x_i \in \boldsymbol x} \mathbb{1}_{x_i \le y}$.


In `Copulas.jl`, we provide a function `pseudos` that implement this transformation directly. 

```@docs
pseudos
```

## Deheuvel's empirical copula

From these pseudo-observations, an empirical copula is defined and anlysed in [deheuvels1979](@cite) as follows:

!!! definition "Deheuvel's empirical copula" 
    The empirical distribution function of the normalized ranks,

    $$\hat{C}_N(\boldsymbol u) = \frac{1}{N} \sum_{i=1}^N \mathbb 1_{\boldsymbol u_i \le \boldsymbol u},$$ is called the empirical copula function.

!!! theorem "Exhaustivity and consistency" 
    $\hat{C}_N$ is an exhaustive estimator of $C$, and moreover for any normalizing constants $\{\phi_N, N\in \mathbb N\}$ such that $\lim\limits_{N \to \infty} \phi_N \sqrt{N^{-1}\ln \ln N} = 0$, 

    $$\lim\limits_{N\to\infty} \phi_N \sup_{\boldsymbol u \in [0,1]^d} \lvert\hat{C}_N(\boldsymbol u) - C(\boldsymbol u) \rvert = 0 \text{ a.s.}$$

$\hat{C}_N$ then converges (weakly) to $C$, the true copula of the random vector $\boldsymbol X$, when the number of observations $N$ goes to infinity. 

!!! note "The empirical copula is not a true copula"
    Despite its name, $\hat{C}_N$ is not a copula since it does not have uniform marginals. Be careful. 

In the package, this copula is implemented as the `EmpiricalCopula`: 

```@docs; canonical=false
EmpiricalCopula
```

!!! note "Conditionals and distortions"
    - Distortions: available via the generic implementation (partial-derivative ratios). For the empirical copula, derivatives are stepwise; interpret results carefully near sample jumps.
    - Conditional copulas: available via the generic implementation. No specialized fast path is provided.

### Visual: empirical copula from pseudo-observations

```@example 1
using Copulas, Distributions, Plots
# generate data with known dependence, then compute pseudo-observations
X = SklarDist(ClaytonCopula(2, 1.2), (Normal(), Beta(1, 4)))
x = rand(X, 1000)
Ĉ = EmpiricalCopula(x, pseudo_values=false)
plot(plot(X.C), plot(Ĉ); layout=(1,2))
```

## Beta copula


The empirical copula function is not a copula. An easy way to fix this problem is to smooth out the marginals with beta distribution functions. The Beta copula is thus defined and analysed in [segers2017](@cite) as follows:


!!! definition "Definition (Beta Copula):"
    Denoting $F_{n,r}(x) = \sum_{s=r}^n \binom{n}{s} x^s(1-x)^{n-s}$ as the distribution function of a $\mathrm{Beta}(r, n+1-r)$ random variable, the function

    $$\hat{C}_N^\beta : \boldsymbol x \mapsto \frac{1}{N} \sum_{i=1}^N \prod_{j=1}^d F_{n,(N+1)u_{i,j}}(x_j)$$ is a genuine copula, called the Beta copula. 

!!! property "Proximity of $\hat{C}_N$ and $\hat{C}_N^\beta$"

    $$\sup_{\boldsymbol u \in [0,1]^d} |\hat{C}_N(\boldsymbol u) - \hat{C}_N^\beta(\boldsymbol u)| \le d\left(\sqrt{\frac{\ln n}{n}} + \sqrt{\frac{1}{n}} + \frac{1}{n}\right)$$

In the package, this copula is implemented as `BetaCopula`:

```@docs; canonical=false
BetaCopula
```

!!! note "Conditionals and distortions"
    - Distortions: specialized fast path returning a MixtureModel of Beta components for efficient evaluation and sampling.
    - Conditional copulas: available via the generic implementation (no dedicated fast path).

### Performance notes
- Construction is O(d·n) after pseudo-observations are computed. Evaluation at a point uses O(d·n) basis lookups; consider subsampling for very large n.

## Bernstein Copula

Bernstein copula are simply another smoothing of the empirical copula using Bernstein polynomials. 

Mathematically, given a base copula $C$ and degrees $\boldsymbol m=(m_1,\ldots,m_d)$, the (cdf) Bernstein copula is

```math
B_{\boldsymbol m}(C)(\boldsymbol u)
= \sum_{s_1=0}^{m_1}\cdots\sum_{s_d=0}^{m_d}
 C\!\left(\tfrac{s_1}{m_1},\ldots,\tfrac{s_d}{m_d}\right)
 \prod_{j=1}^d \binom{m_j}{s_j} u_j^{s_j} (1-u_j)^{m_j-s_j}.
```

It is a multivariate Bernstein polynomial approximation of $C$ on the uniform grid. Larger $m_j$ increase smoothness and accuracy at higher computational cost.

In the package, this copula is implemented as `BernsteinCopula`:

```@docs; canonical=false
BernsteinCopula
```

!!! note "Conditionals and distortions"
    - Distortions: specialized fast path returning a MixtureModel of Beta components (weights from Bernstein grid finite differences conditioned on $\boldsymbol u_J$).
    - Conditional copulas: available via the generic implementation (no dedicated fast path).

### Performance notes
- Complexity grows with the grid size ∏_j (m_j+1) for cdf and ∏_j m_j for pdf. In higher dimensions, keep m small or prefer the 2D specialized paths provided.
- Small negative finite differences from numerical noise are clipped to zero before normalization.

## Checkerboard Copulas

There are other nonparametric estimators of the copula function that are true copulas. Of interest to our work is the Checkerboard construction (see [cuberos2019,mikusinski2010](@cite)), detailed below.

First, for any $\boldsymbol m \in \mathbb N^d$, let $\left\{B_{\boldsymbol i,\boldsymbol m}, \boldsymbol i < \boldsymbol m\right\}$ be a partition of the unit hypercube defined by

$$B_{\boldsymbol i, \boldsymbol m} = \left]\frac{\boldsymbol i}{\boldsymbol m}, \frac{\boldsymbol i+1}{\boldsymbol m}\right].$$

Furthermore, for any copula $C$ (or more generally distribution function $F$), we denote $\mu_{C}$ (resp $\mu_F$) the associated measure.  For example, for the independence copula $Pi$, $\mu_{\Pi}(A) = \lambda(A \cup [\boldsymbol 0, \boldsymbol 1])$ where $\lambda$ is the Lebesgue measure.

!!! definition "Empirical Checkerboard copulas"
    Let $\boldsymbol m \in \mathbb{N}^d$. The $\boldsymbol m$-Checkerboard copula $\hat{C}_{N,\boldsymbol m}$, defined by

    $$\hat{C}_{N,\boldsymbol m}(\boldsymbol x) = \boldsymbol m^{\boldsymbol 1} \sum_{\boldsymbol i < \boldsymbol m} \mu_{\hat{C}_N}(B_{\boldsymbol i, \boldsymbol m}) \mu_{\Pi}(B_{\boldsymbol i, \boldsymbol m} \cap [0,\boldsymbol x]),$$

    is a genuine copula as soon as $m_1, ..., m_d$ all divide $N$.


!!! property "Consistency of $\hat{C}_{N,\boldsymbol m}$"
    If all $m_1, ..., m_d$ divide $N$,

    $$\sup_{\boldsymbol u \in [0,1]^d} |\hat{C}_{N,\boldsymbol m}(\boldsymbol u) - C(\boldsymbol u)| \le \frac{d}{2m} + \mathcal{O}_{\mathbb{P}}\left(n^{-1/2}\right).$$


This copula is called *Checkerboard*, as it fills the unit hypercube with hyperrectangles of same shapes $B_{\boldsymbol i, \boldsymbol m}$, conditionally on which the distribution is uniform, and the mixing weights are the empirical frequencies of the hyperrectangles. 

It can be noted that there is no need for the hyperrectangles to be filled with a uniform distribution ($\mu_{\Pi}$), as soon as they are filled with copula measures and weighted according to the empirical measure in them (or to any other copula). The direct extension is then the more general patchwork copulas, whose construction is detailed below.

Denoting $B_{\boldsymbol i, \boldsymbol m}(\boldsymbol x) = B_{\boldsymbol i, \boldsymbol m} \cap [0,\boldsymbol x]$, we have : 

```math
\begin{align}
  m^d\mu_{\Pi}(B_{\boldsymbol i, \boldsymbol m} \cap [0,\boldsymbol x]) &= \frac{\mu_{\Pi}(B_{\boldsymbol i, \boldsymbol m} \cap [0,\boldsymbol x])}{\mu_{\Pi}(B_{\boldsymbol i, \boldsymbol m})}\\
  &= \frac{\mu_{\Pi}(B_{\boldsymbol i, \boldsymbol m}(\boldsymbol x))}{\mu_{\Pi}(B_{\boldsymbol i, \boldsymbol m})}\\
  &= \mu_{\Pi}(\boldsymbol m B_{\boldsymbol i, \boldsymbol m}(\boldsymbol x))
\end{align}
```
where we intend $\boldsymbol m ]\boldsymbol a, \boldsymbol b] = ] \boldsymbol m \boldsymbol a, \boldsymbol m \boldsymbol b]$ (products between vectors are componentwise).

This allows for an easy generalization in the framework of patchwork copulas [durante2012,durante2013,durante2015](@cite):

!!! definition "Patchwork copulas"
    Let $\boldsymbol m \in \mathbb{N}^d$ all divide $N$, and let $\mathcal{C} = \{C_{\boldsymbol i}, \boldsymbol i < \boldsymbol m\}$ be a given collection of copulas. The distribution function:

    $$\hat{C}_{N,\boldsymbol m, \mathcal{C}}(\boldsymbol x) = \sum_{\boldsymbol i < \boldsymbol m} \mu_{\hat{C}_N}(B_{\boldsymbol i, \boldsymbol m}) \mu_{C_{\boldsymbol i}}(\boldsymbol m B_{\boldsymbol i, \boldsymbol m}(\boldsymbol x))$$
    is a copula. 

In fact, replacing $\hat{C}_N$ by any copula in the patchwork construct still yields a genuine copula, with no more conditions that all components of $\boldsymbol m$ divide $N$. The Checkerboard grids are practical in the sense that computations associated to a Checkerboard copula can be really fast: if the grid is large, the number of boxes is small, and otherwise if the grid is very refined, many boxes are probably empty. On the other hand, the grid is fixed a priori, see [laverny2020](@cite) for a construction with an adaptive grid.

Convergence results for this kind of copulas can be found in [durante2015](@cite), with a slightly different parametrization. 

In the package, this copula is implemented as `CheckerboardCopula`:

```@docs; canonical=false
CheckerboardCopula
```

!!! note "Conditionals and distortions"
    - Distortions: specialized for conditioning on a single coordinate (p=1) via a histogram-bin distortion on the corresponding slice.
    - Conditional copulas: specialized projection onto remaining axes, renormalizing the mass in the fixed bins, still returns a Checkerboard. 

### Performance notes
- Construction cost scales with sample size but stores only occupied boxes (sparse). CDF evaluation is O(#occupied boxes) at query time.
- For large n choose coarser m to reduce occupied boxes; for small n a finer grid is possible but may leave many empty boxes.


## Empirical Extreme-Value copula (Pickands estimator)

In addition to the empirical, beta, Bernstein and checkerboard constructions, we provide a nonparametric bivariate Extreme Value copula built from data by estimating the Pickands dependence function. The tail implementation [`EmpiricalEVTail`](@ref) supports several classical estimators (Pickands, CFG, OLS intercept), and a convenience constructor `EmpiricalEVCopula` builds the corresponding `ExtremeValueCopula` directly from pseudo-observations.

Typical workflow:


See the Extreme Value manual page for background and the bestiary entry for the full API of [`EmpiricalEVTail`](@ref available_extreme_models).

```@docs; canonical=false
EmpiricalEVTail
```


## Empirical Archimedean generator (Kendall inversion)

Beyond copula estimators, we also provide a nonparametric estimator of a $d$-Archimedean generator from data, based on the empirical Kendall distribution following [genest2011a](@cite). For a $d$-Archimedean copula with generator $\varphi$, there exists a nonnegative random variable $R$ (the radial law) such that

$$\varphi(t) \,=\, \mathbb{E}\,\big[\,(1 - t/R)_{+}^{\,d-1}\,\big].$$

Approximating the (unknown) $R$ by a discrete measure $\widehat R = \sum_j w_j\,\delta_{r_j}$ yields a piecewise-polynomial generator

$$\widehat\varphi(t) \,=\, \sum_j w_j\,\big(1 - t/r_j\big)_{+}^{\,d-1}.$$

The radii and weights are recovered from the empirical Kendall distribution via a triangular recursion (see the example page for details), and the resulting generator is exposed as `EmpiricalGenerator`.

Usage:

- Build from data `u::d×n` (raw or pseudos): `Ĝ = EmpiricalGenerator(u; pseudo_values=true)`
- Use directly in an Archimedean copula: `Ĉ = ArchimedeanCopula(d, Ĝ)`
- Access the fitted radial law: `R̂ = williamson_dist(Ĝ, Val{d}())`

```@docs; canonical=false
EmpiricalGenerator
```

### Performance notes
- The Kendall sample computation is currently O(n^2) in the number of observations. For large n, future versions may switch to Fenwick-tree–based sweeps to reach ~O(n log n) in bivariate cases and ~O(n log^{d-1} n) for higher d.

See [This example page](@ref nonpar_archi_gen_example) for more details and example usages. 


## See also

- Bestiary: [Empirical models list](@ref empirical_cops)
- Extreme Value: [Extreme Value family](@ref Extreme_theory), [Implemented tails](@ref available_extreme_models)
- Manual: [Sklar's Distribution](@ref), [Conditioning and Subsetting](@ref)


```@bibliography
Pages = [@__FILE__]
Canonical = false
```