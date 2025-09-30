````markdown
```@meta
CurrentModule = Copulas
````

# Fitting interface

This section summarizes **how to fit** copulas (and Sklar distributions) in `Copulas.jl`, without going into the details of each family.

---

## Data Conventions

* We work with **pseudo-observations** `U ∈ (0,1)^{d×n}` (rows = dimension, columns = observations).
* The *rank* routines (τ/ρ/β/γ) assume pseudo-observations.
* The `StatsBase` functions for pairwise correlations use the `n×d` convention; when called internally, they are transposed (`U`) as appropriate.

---

## Main Calls

### Copula Only (Object)

```julia
Ĉ = fit(CT, U; method=:mle, kwargs...)
```

Returns **only** the fitted copula `Ĉ::CT`. This is a high-level shortcut.

### Full Model (with metadata)

```julia
M = fit(CopulaModel, CT, U; method=:default, summaries=true, kwargs...)
```

Returns a `CopulaModel` with:

* `result` (the fitted copula), `n`, `ll` (log-likelihood),
* `method`, `converged`, `iterations`, `elapsed_sec`,
* `vcov` (if available),
* `method_details` (a named tuple with method metadata and, if `summaries=true`, **pairwise summaries**: means, deviations, minima, and maxima of empirical τ/ρ/β/γ).

### Joint margin fitting + copula (Sklar)

```julia
Ŝ = fit(CopulaModel, SklarDist{CT,TupleOfMargins}, X;
sklar_method=:ifm, # or :ecdf
copula_method=:default,
margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple())
```

* `:ifm`: fits parametric margins and projects to pseudo-data with their CDFs.
* `:ecdf`: uses pseudo-empirical observations (ranks).

---

## Available fitting methods

Availability depends on the family; you can check it with:

```julia
_available_fitting_methods(CT) # e.g. (:mle, :itau, :irho, :ibeta)
```

### Short description

* `:mle` — **Maximum likelihood** over `U`. Recommended when there is a stable density and good reparameterization.
* `:itau` — **Kendall's inverse**: matches `τ_emp(U)` with `τ_theo(θ)`. Ideal for **single-parameter families** or when a reliable monotone inverse exists.
* `:irho` — **Spearman's inverse**: analogous to ρ; can operate with scalar or matrix objectives (e.g., multivariate Gaussians).
* `:ibeta` — **Blomqvist's inverse**: scalar; **only** works if the family has **≤ 1 free parameter**.

> **Note:** Rank-based methods require that the number of free parameters not exceed the information of the coefficient(s) used; in the case of `:ibeta`, this is explicitly enforced.

In **extreme value** copulas, the `:mle`/`:iupper` variants can rely on the Pickands function (A(t)) and its derivatives (`A`, `dA`, `d²A`) with Brent-type numerical inversion.

---

## `CopulaModel` Interface (summary)

The

```julia
CopulaModel{CT} <: StatsBase.StatisticalModel
```

stores the result and supports the standard `StatsBase` interface:

| Function           | Description                     |
| ------------------ | ------------------------------- |
| `nobs(M)`          | Number of observations          |
| `loglikelihood(M)` | Log-likelihood at the optimum   |
| `deviance(M)`      | Deviance (= −2 loglikelihood)   |
| `coef(M)`          | Estimated parameters            |
| `coefnames(M)`     | Parameter names                 |
| `vcov(M)`          | Var–cov matrix                  |
| `stderror(M)`      | Standard errors                 |
| `confint(M)`       | Confidence intervals            |
| `aic(M)`           | Akaike Information Criterion    |
| `bic(M)`           | Bayesian Information Criterion  |
| `aicc(M)`          | Corrected AIC (small-sample)    |
| `hqc(M)`           | Hannan–Quinn criterion          |


Quick access to the contained copula: `_copula_of(M)` (returns the copula even if `result` is a `SklarDist`).

---

## Minimal examples

Below are minimal examples illustrating the main fitting strategies.

```julia
using Random, Copulas, StatsBase, Distributions
Random.seed!(1234)

# True copula for simulation
Ctrue = GumbelCopula(2, 3.0)
U = rand(Ctrue, 2_000)

# 1) itau — inverse Kendall (fast, uniparametric)
C_itau = fit(GumbelCopula, U; method=:itau)

# 2) mle — maximum likelihood (robust, multiparametric)
M_mle = fit(CopulaModel, GumbelCopula, U; method=:mle)

# 3) Sklar — joint margins + copula
X = [rand(Normal(), 2000) rand(LogNormal(), 2000)]'  # raw data, d×n
M_sklar = fit(CopulaModel, SklarDist{GumbelCopula,(Normal,LogNormal)}, X; sklar_method=:ecdf, copula_method=:itau)

# 4) Empirical copula — nonparametric
Uemp = pseudos(rand(Normal(), 2, 300))  # pseudo-observations
C_emp = fit(CopulaModel, BetaCopula, U)
```

* **1) itau**: inversion of Kendall’s τ, quick and stable for one-parameter families.
* **2) mle**: maximizes the likelihood, suitable for more complex families.
* **3) Sklar**: estimates both marginals and copula jointly.
* **4) Empirical**: nonparametric alternative, no parametric assumptions.

For more details on empirical methods see [Empirical models](@ref empirical_copulas).

```@bibliography
Pages = [@__FILE__]
Canonical = false
```