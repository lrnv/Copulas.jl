````markdown
```@meta
CurrentModule = Copulas
````

# Fitting interface

This section summarizes **how to fit** copulas (and Sklar distributions) in `Copulas.jl`, without going into the details of each family.

---

## Data Conventions

* We work with **pseudo-observations** `U ∈ (0,1)^{d×n}` (rows = dimension, columns = observations). You can use the `pseudo()` function to get such normalized ranks. 
* The *rank* routines (τ/ρ/β/γ) assume pseudo-observations.
* The `StatsBase` functions for pairwise correlations use the `n×d` convention; when called internally, they are transposed (`U`) as appropriate.

---

## Main Calls

### Copula Only (Object)

```@example fitting_interface
using Copulas, Random, StatsBase, Distributions
Random.seed!(123) # hide
Ctrue = GumbelCopula(2, 3.0)
U = rand(Ctrue, 2000)
Ĉ = fit(GumbelCopula, U; method=:mle)
Ĉ
```

Returns **only** the fitted copula `Ĉ::CT`. This is a high-level shortcut.

```@example fitting_interface
using Plots
plot(Ĉ)
```

### Full Model (with metadata)

```@example fitting_interface
M = fit(CopulaModel, GumbelCopula, U; method=:default, summaries=true)
M
```

Returns a `CopulaModel` with:

* `result` (the fitted copula), `n`, `ll` (log-likelihood),
* `method`, `converged`, `iterations`, `elapsed_sec`,
* `vcov` (if available),
* `method_details` (a named tuple with method metadata and, if `summaries=true`, **pairwise summaries**: means, deviations, minima, and maxima of empirical τ/ρ/β/γ).

---

## Behavior & conventions (important)

- fit operates on types, not on pre-constructed parameterized instances. Always pass a Copula or SklarDist *type* to `fit`, e.g. `fit(GumbelCopula, U)` or `fit(CopulaModel, SklarDist{ClaytonCopula,Tuple{Normal,LogNormal}}, X)`. If you already have a constructed instance `C0`, re-estimate its parameters by calling `fit(typeof(C0), U)`.

- Default method selection: each family exposes the list of available fitting strategies via `_available_fitting_methods(CT)`. When `method = :default` the first element of that tuple is used. Example: `Copulas._available_fitting_methods(MyCopula)`.

- `CopulaModel` is the full result object returned by the fits performed via `Distributions.fit(::Type{CopulaModel}, ...)`. The light-weight shortcut `fit(MyCopula, U)` returns only a copula instance; use `fit(CopulaModel, ...)` to get diagnostics and metadata.

## `CopulaModel` Interface (summary)

The `CopulaModel{CT} <: StatsBase.StatisticalModel` type stores the result and supports the standard `StatsBase` interface:

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


### Pairwise summaries and `method_details`

When you request `summaries=true` (default) the returned `CopulaModel` contains extra pre-computed pairwise statistics inside `M.method_details`. Typical keys are:

- `:tau_mean`, `:tau_sd`, `:tau_min`, `:tau_max`
- `:rho_mean`, `:rho_sd`, `:rho_min`, `:rho_max`
- `:beta_mean`, `:beta_sd`, `:beta_min`, `:beta_max`
- `:gamma_mean`, `:gamma_sd`, `:gamma_min`, `:gamma_max`

Access example:

```@example fitting_interface
M = fit(CopulaModel, GumbelCopula, U; summaries=true)
M.method_details.tau_mean  # average pairwise Kendall's tau
```

If `summaries=false` these keys will be absent and `method_details` will be smaller.

### `vcov` and inference notes

- The `CopulaModel` field `vcov` (exposed as `StatsBase.vcov(M)`) contains the estimated covariance matrix of the fitted copula parameters when available. Many families supply `:vcov` via the `meta` NamedTuple returned by `_fit` (for example `meta.vcov`). If `vcov === nothing` the package cannot compute `stderror` or `confint` and those helpers will throw.

- If a family does not provide `vcov`, you can obtain approximate standard errors by computing a numerical Hessian of the negative loglikelihood with respect to the raw/unbound parameters and inverting it. Minimal runnable example (adapt to your family internals):

```@example fitting_interface
using ForwardDiff
CT = GumbelCopula
d = 2
ex = _example(CT, d)
U = rand(ex, 200)
α̂ = _unbound_params(CT, d, Distributions.params(ex))
lossα(α) = -Distributions.loglikelihood(CT(d, _rebound_params(CT, d, α)...), U)
H = ForwardDiff.hessian(lossα, α̂)
V_α = inv(H)
println("approx vcov size: ", size(V_α))
```

Consider adding a small bootstrap wrapper for robust CIs when the Hessian is unreliable.

## Joint margin fitting + copula (Sklar)

* `:ifm`: fits parametric margins and projects to pseudo-data with their CDFs.
* `:ecdf`: uses pseudo-empirical observations (ranks).

**Notes:**

- `sklar_method=:ifm` fits parametric margins first and converts observations to pseudo-scale via the fitted marginal CDFs. `sklar_method=:ecdf` uses empirical pseudo-observations (ranks). Choose `:ifm` when margins are believed parametric and you want a model-based projection; choose `:ecdf` to avoid margin misspecification.

- `margins_kwargs` is currently a single `NamedTuple` applied to every marginal fit. If you need different options per margin, fit margins manually (call `Distributions.fit` for each marginal type) and then call the copula fit on the pseudo-data yourself.

- The model's `null_ll` field (used in LR tests) is computed as the sum of the marginal logpdfs under the fitted margins; LR tests compare the fitted copula vs independence while keeping the same margins.

```@example fitting_interface
S = SklarDist(ClaytonCopula(2, 5), (Normal(), LogNormal(0, 0.5)))
X = rand(S, 1000)
Ŝ = fit(CopulaModel, SklarDist{ClaytonCopula,Tuple{Normal,LogNormal}}, X;
	sklar_method=:ifm, # or :ecdf
	copula_method=:default, # see next section. 
	margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple()) # options will be passed down to fitting functions. 
Ŝ
```

```@example fitting_interface
plot(Ŝ.result)
```

---

## Available fitting methods

The names and availiability of fitting methods depends on the model. You can check what is available with the following internal call : 

```@example fitting_interface
Copulas._available_fitting_methods(ClaytonCopula)
```

The first method in the list is the one used by default. 

### Short description

* `:mle` — **Maximum likelihood** over `U`. Recommended when there is a stable density and good reparameterization.
* `:itau` — **Kendall's inverse**: matches the theoretical `τ(C)` with the empirical `τ(U)`. Ideal for **single-parameter families** or when a reliable monotone inverse exists.
* `:irho` — **Spearman's inverse**: analogous to `ρ`; can operate with scalar or matrix objectives (e.g., multivariate Gaussians).
* `:ibeta` — **Blomqvist's inverse**: scalar; **only** works if the family has **≤ 1 free parameter**.

> **Note:** Rank-based methods require that the number of free parameters not exceed the information of the coefficient(s) used; in the case of `:ibeta`, this is explicitly enforced.

In **extreme value** copulas, the `:mle`/`:iupper` variants can rely on the Pickands function (A(t)) and its derivatives (`A`, `dA`, `d²A`) with Brent-type numerical inversion.

---

## Implementing fitting for a new family (contributor checklist)

When you add a new copula family, implement the following so the generic `fit` flow works seamlessly:

1. `_example(CT, d)` — return a representative instance (used to obtain default params and initial values).
2. `_unbound_params(CT, d, params)` — transform the family `NamedTuple` parameters to an unconstrained `Vector{Float64}` used by optimizers.
3. `_rebound_params(CT, d, α)` — invert `_unbound_params`, returning a `NamedTuple` suitable for `CT(d, ...)` construction.
4. `_available_fitting_methods(::Type{<:YourCopula})` — declare supported methods (examples:  `:mle, :itau, :irho, :ibeta, ...`).
5. `_fit(::Type{<:YourCopula}, U, ::Val{:mle})` (and other `Val{}` methods) — implement the method and return `(fitted_copula, meta::NamedTuple)`; include keys such as `:θ̂`, `:optimizer`, `:converged`, `:iterations` and optionally `:vcov`.

Place this checklist and a minimal `_fit` skeleton in `docs/src/manual/developer_fitting.md` where contributors can copy/paste and adapt.
````

