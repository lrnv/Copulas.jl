```@meta
CurrentModule = Copulas
```

# [Fitting interface](@id fitting_interface)

This section summarizes how to **fit** copulas (and Sklar distributions) in `Copulas.jl`, without going into family-specific details.

---

## Data conventions

- We work with **pseudo-observations** `U ∈ (0,1)^{d×n}` (rows = dimensions, columns = observations).  
  Use `pseudos(X)` to obtain normalized ranks from raw data `X`.
- Rank-based routines (tau / rho / beta / gamma) assume pseudo-observations.
- `StatsBase` pairwise correlation helpers use the `n×d` convention rather than
  the package's `d×n` fitting convention.

---

## Main calls

### Copula only (object)

```@example fitting_interface
using Copulas, Random, StatsBase, Distributions, Plots
Random.seed!(123) # hide
Ctrue = GumbelCopula(2, 3.0)
U = rand(Ctrue, 300)
Ĉ = fit(GumbelCopula, U; method=:mle)
Ĉ
```

Returns **only** the fitted copula `Ĉ::CT` (high-level shortcut).

### Full model (with metadata)

```@example fitting_interface
M = fit(CopulaModel, GumbelCopula, U; method=:default)
M
```

Returns a [`CopulaModel`](@ref) with:
- `result` (the fitted copula), `n`, `ll` (log-likelihood),
- `method`, `converged`, `iterations`, `elapsed_sec`,
- `vcov` (if available),
- `method_details` (a `NamedTuple` with method-specific metadata).

## Automatic copula-family selection

When the copula family is unknown, `CopulaModel` can select it automatically
from a collection of candidate families:

```@example fitting_interface
Ctrue = ClaytonCopula(2, 4.0)
Usel = rand(Ctrue, 300)

Msel = fit(
    CopulaModel,
    Copulas.Copula,
    Usel;
    candidates=(ClaytonCopula, GumbelCopula, FrankCopula),
    criterion=:bic,
    vcov=false,
)
Msel
```

The available information criteria are:

- `:bic` — Bayesian information criterion,
- `:aic` — Akaike information criterion,
- `:aicc` — finite-sample corrected AIC,
- `:hqc` — Hannan–Quinn criterion.

BIC is the default criterion.

Candidate fits are compared using the requested criterion, and the family with
the smallest eligible finite value is selected. The winning fit is reused:
only its requested inference (such as covariance estimation) is computed afterwards.

The complete comparison can be inspected with [`selectiontable`](@ref):

```@example fitting_interface
selectiontable(Msel)
```

Each row stores the candidate family, fitting status and method,
log-likelihood, number of parameters, and all four information criteria.
Candidates that fail to fit can be skipped with `on_error=:skip` (the default)
or propagated immediately with `on_error=:throw`.

The candidate collection is required and explicit; choose families appropriate
for the scientific problem and the data dimension. `selectiontable` returns a
plain vector of comparison rows. Nonfinite scores and, by default, nonconverged
fits are excluded. Interruptions always propagate, even with `on_error=:skip`.

Use maximum-likelihood fitting for the usual information-criterion interpretation;
passing another fitting method merely compares the scores at those estimates.
The shorter `fit(Copulas.Copula, U; candidates=(...))` returns only the selected copula.

`GOFCopulaTest(Msel)` and `GOFCopulaTest(Msel, U)` are deliberately unsupported:
a valid selection-aware bootstrap must repeat family selection in every replicate,
not just refit the winning family. These calls throw rather than silently omit
the selection step.


---

## Behavior & conventions (important)

- ``fit`` usually operates on **types**. Structural models with runtime
  configuration, such as `NestedArchimedeanCopula`,
  are instead fitted from a template instance.
  Pass a copula or Sklar type, e.g. `fit(GumbelCopula, U)` or  
  `fit(CopulaModel, SklarDist{ClaytonCopula,Tuple{Normal,LogNormal}}, X)`.  
  If an instance `C0` is completely described by its type, re-estimate its
  parameters with `fit(typeof(C0), U)`. For a structural model, use `fit(C0, U)`.

- **Default method selection.** With `method=:default`, each family selects its
  documented default fitting strategy. Supported explicit methods depend on the
  family.

- `CopulaModel` is the full result returned by `Distributions.fit(::Type{CopulaModel}, ...)`.  
  The lightweight shortcut `fit(MyCopula, U)` returns only a copula; use `fit(CopulaModel, ...)` to get diagnostics and metadata.

---

## `CopulaModel` interface (summary)

`CopulaModel` implements `StatsBase.StatisticalModel` and supports the following
documented functions and properties:

| Function / property                            | Description                                                                                       |
|-----------------------------------------------|---------------------------------------------------------------------------------------------------|
| `M.ll`                                         | Log-likelihood at the optimum.                                                                    |
| `nobs(M)`                                      | Number of observations used in the fit.                                                           |
| `deviance(M)`                                  | Deviance (= −2 · `M.ll`).                                                                         |
| `nullloglikelihood(M)`                         | Log-likelihood under independence with same margins (available for Sklar fits).                   |
| `nulldeviance(M)`                              | Deviance of the null model (−2 · `nullloglikelihood(M)`).                                         |
| `aic(M)` / `bic(M)`						     | Information criteria from ``StatsBase.jl``                                                        |
| `coef(M)` / `coefnames(M)`                     | Estimated parameters and their names.                                                             |
| `vcov(M)`                                      | Parameter variance–covariance matrix (may be `nothing`).                                          |
| `stderror(M)` / `confint(M; level=0.95)`       | Standard errors and Wald confidence intervals; return `nothing` when `vcov(M) === nothing`.       |
| `residuals(M; transform=:uniform \| :normal)`  | Rosenblatt residuals on `[0,1]` or Normal scale when the fit retains the required observations.  |
| `predict(M; what=:cdf\|:pdf\|:simulate, ...)`  | CDF/PDF at `newdata`, or simulation (`nsim`; default `nsim = M.n` if `nsim == 0`).                |

**Examples**

```@example fitting_interface
# Information criteria
StatsBase.aic(M)
StatsBase.bic(M)
```

```@example fitting_interface
# Standard errors and Wald CIs
StatsBase.stderror(M)
StatsBase.confint(M; level=0.95)
```

```@example fitting_interface
# Rosenblatt residuals
R  = StatsBase.residuals(M; transform=:uniform)
RN = StatsBase.residuals(M; transform=:normal)
(size(R), size(RN))
```

```@example fitting_interface
# Predictions and simulation
P  = StatsBase.predict(M; what=:cdf, newdata=rand(2, 5))   # CDF at 5 points
F  = StatsBase.predict(M; what=:pdf, newdata=rand(2, 5))   # PDF at 5 points
X̂  = StatsBase.predict(M; what=:simulate, nsim=200)       # simulate 200 obs
(size(P), size(F), size(X̂))
```

---

## Covariance estimation (`vcov`) and inference

When fitting with `fit(CopulaModel, ...)`, the keyword `vcov=true` triggers estimation of the **parameter covariance matrix**.

> **Default.** `vcov = true`. Covariance is computed automatically unless the
> user disables it (`vcov=false`) or it is unavailable for the fitted family.

The `vcov_method` keyword selects the estimator:

| Symbol               | Description                                                                                      |
|----------------------|--------------------------------------------------------------------------------------------------|
| `:hessian`           | Inverse observed information (−Hessian of the log-likelihood). Default for `method = :mle`.     |
| `:godambe`           | Godambe (sandwich) estimator based on score-type functions. Used for rank-based fits.            |
| `:godambe_pairwise`  | Pairwise Godambe using all variable pairs.                                                       |
| `:jackknife`         | Leave-one-out jackknife approximation (robust fallback).                                        |
| `:bootstrap`         | Bootstrap approximation (√n resamples, up to 200).                                              |

You can override the choice via `vcov_method`:

```@example fitting_interface
M2 = fit(CopulaModel, GumbelCopula, U; method=:mle, vcov=true, vcov_method=:bootstrap, derived_measures=false)
StatsBase.vcov(M2) isa AbstractMatrix
```

Each method returns a symmetric positive semi-definite matrix exposed by
`StatsBase.vcov(M)`.

If the requested covariance calculation is numerically unavailable,
`StatsBase.vcov(M)` may be `nothing`.

**Note.** In the example above we set `derived_measures=false`, which disables the automatic calculation and storage of dependence measures (e.g., Kendall’s τ, Spearman’s ρ, Blomqvist’s β, , Gini's γ, upper/lower tail coefficients, entropy). By default this is enabled. Disabling it reduces computation and memory footprint and omits the *Dependence metrics* section in the REPL summary.
---

## Joint margins + copula (Sklar)

You can pass the `sklar_method` parameter as: 

- `:ifm`: fits parametric margins and maps data to pseudo-scale via their CDFs.  
- `:ecdf`: uses empirical pseudo-observations (ranks).

**Notes**

- Use `sklar_method = :ifm` when margins are plausibly parametric and you want a model-based projection; use `:ecdf` to avoid margin misspecification.
- `margins_kwargs` is a single `NamedTuple` applied to every marginal fit. For heterogeneous options, fit margins manually and then fit the copula on the resulting pseudo-data.
- The model’s `null_ll` (for LR tests) is the log-likelihood under independence with the **same margins**.

```@example fitting_interface
S = SklarDist(ClaytonCopula(2, 5), (Normal(), LogNormal(0, 0.5)))
X = rand(S, 300)
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

## Fitting methods

The names and availability of fitting methods depend on the family. Use
`method=:default` unless a family documents a more appropriate explicit method.

### Short descriptions

- `:mle` — **Maximum likelihood** over `U`. Recommended when a stable density and a good reparameterization exist.
- `:itau` — **Kendall inverse**: matches theoretical `tau(C)` to empirical `tau(U)`. Ideal for single-parameter families with a monotone inverse.
- `:irho` — **Spearman inverse**: analogous to `rho`; can use scalar or matrix objectives (e.g., multivariate Gaussians).
- `:ibeta` — **Blomqvist inverse**: scalar; only valid for families with **≤ 1** free parameter.

> **Remark.** Rank-based methods require that the number of free parameters does not exceed the information contained in the chosen coefficient(s); `:ibeta` enforces this explicitly.

For **extreme-value** copulas, `:mle` / `:iupper` use the documented Pickands
representation when supported by the family.

## Nonparametric fits (Empirical Copulas)

In addition to parametric families (MLE / rank-based), `Copulas.jl` exposes several **nonparametric** or **empirical** constructions that can be fit through the same high-level API:

- `EmpiricalCopula` (Deheuvels)
- `BetaCopula`
- `BernsteinCopula`
- `CheckerboardCopula`
- `EmpiricalEVCopula` (bivariate Pickands estimation or shape-constrained
  multivariate spectral estimation, selected from the data dimension)

See the dedicated page for theory, properties, and references: [Empirical models](@ref empirical_copulas).

For empirical models with a density, the **StatsBase / StatsModels** interface works identically:
you can call `coef`, `aic`, `bic`, `deviance`, `predict`, `residuals`, etc.,  
and obtain a full `CopulaModel` with the same documented model interface.

The multivariate `EmpiricalEVCopula` projection may contain singular spectral
components and therefore has no global Lebesgue density. Its quick `fit`
interface, CDF, and sampler remain available, but likelihood-based summaries
are not applicable.

The only difference is that empirical models are **parameter-free** (`dof(M) = 0`),  
so `vcov(M)`, `stderror(M)`, and `confint(M)` return `nothing`,  
and information criteria reduce to AIC = BIC = −2 · loglikelihood.  
Otherwise, all features—including the computation of dependence measures and the REPL summary—behave exactly the same.
