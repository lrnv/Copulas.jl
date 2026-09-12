```@meta
CurrentModule = Copulas
```

# [Fitting interface](@id fitting_interface)

Fitting turns an observed sample into a dependence model. This sounds like a
single operation, but it involves several distinct choices: whether the margins
are known or estimated, which copula family is plausible, which feature of the
data identifies its parameters, and how uncertainty should be reported. This
page introduces those choices before presenting their Julia interface.

Throughout this page, observations are columns of a ``d\times n`` matrix. A
copula fit therefore starts from pseudo-observations on ``[0,1]^d``; a
`SklarDist` fit starts from observations on their original marginal scales.

::: definition Copula estimator

Given pseudo-observations ``U_1,\ldots,U_n``, a copula estimator associates the
sample with a fitted copula ``C_{\widehat\theta}``. Different fitting methods
define ``\widehat\theta`` differently: maximum likelihood optimizes the joint
density, whereas inversion methods match one or more empirical dependence
coefficients to their theoretical values.

:::

## From a point estimate to a statistical model

### A fitted copula

When only the estimated distribution is needed, `fit` returns it directly:

```@example fitting_interface
using Copulas, Random, StatsBase, Distributions, Plots
Random.seed!(123) # hide
Ctrue = GumbelCopula(2, 3.0)
U = rand(Ctrue, 300)
Ĉ = fit(GumbelCopula, U; method=:mle)
```

### Keeping the evidence behind the fit

An estimated parameter without information about how it was obtained is often
not enough. `CopulaModel` retains the likelihood, fitting method, convergence
information and, when requested, an estimate of parameter uncertainty:

```@example fitting_interface
M = fit(CopulaModel, GumbelCopula, U; method=:default)
```

The fitted distribution is available through the standard model interface;
the printed report also summarizes convergence, elapsed time, the estimator
used and the available dependence measures. Use this form whenever the fit will
be compared, diagnosed or used for inference.

::: remark Two levels of interface

`fit(Family, U)` is deliberately lightweight and returns only the fitted
copula. `fit(CopulaModel, Family, U)` fits the same model while retaining the
statistical evidence and estimator specification. The latter is not a different
estimator unless different keywords are supplied.

:::

## Comparing candidate families

Choosing a family is part of modelling, not a consequence of optimization.
When several scientifically defensible families remain, they can be fitted to
the same data and compared by an information criterion.

::: definition Information-criterion selection

For each candidate family, fit a model and compute a penalized likelihood
criterion. The selected candidate is the successful, converged fit with the
smallest eligible finite criterion. AIC emphasizes estimated predictive loss;
BIC and HQC penalize model dimension more strongly as the sample grows, while
AICc corrects AIC in small samples.

:::

When the copula family is unknown, `CopulaModel` can select it automatically
from a collection of candidate families:

```@example fitting_interface
Ctrue = ClaytonCopula(2, 4.0)
data = rand(Ctrue, 300)

Msel = fit(
    CopulaModel,
    Copulas.Copula,
    data;
    candidates=(ClaytonCopula, GumbelCopula, FrankCopula),
    criterion=:bic,
    vcov=false,
)
Msel
```

The available criteria are:

- `:bic` — Bayesian information criterion,
- `:aic` — Akaike information criterion,
- `:aicc` — finite-sample corrected AIC,
- `:hqc` — Hannan–Quinn criterion.

BIC is the default criterion.

The winning fit is reused: expensive inference requested by the user, such as
covariance estimation, is computed only after comparison.

The complete comparison can be inspected with [`selectiontable`](@ref):

```@example fitting_interface
selectiontable(Msel)
```

Each row stores the candidate family, fitting status and method,
log-likelihood, number of parameters, and all four information criteria.
Candidates that fail to fit can be skipped with `on_error=:skip` (the default)
or propagated immediately with `on_error=:throw`.

::: warning Candidate lists are scientific assumptions

The candidate collection is intentionally explicit. Automatic selection does
not make every implemented family plausible for every dimension, tail regime or
scientific question. Nonfinite scores and, by default, nonconverged fits are
excluded; `on_error=:throw` is useful when a failed candidate should invalidate
the comparison rather than merely be recorded in `selectiontable`.

:::

Use maximum-likelihood fitting for the usual information-criterion interpretation;
passing another fitting method merely compares the scores at those estimates.
The shorter `fit(Copulas.Copula, U; candidates=(...))` returns only the selected copula.

::: remark Selection and goodness of fit

`GOFCopulaTest(Msel)` and `GOFCopulaTest(Msel, U)` are deliberately unsupported:
a valid selection-aware bootstrap must repeat family selection in every replicate,
not just refit the winning family. These calls throw rather than silently omit
the selection step.

:::

## What exactly is being fitted?

Usually, the model is identified by a copula or Sklar type, for example
`fit(GumbelCopula, U)` or
`fit(CopulaModel, SklarDist{ClaytonCopula,Tuple{Normal,LogNormal}}, X)`. With
`method=:default`, each family chooses its documented default estimator;
explicitly supported methods depend on the family.

The form `SklarDist{CopulaType,Tuple{MarginTypes...}}` is intentionally public
syntax for this purpose: it selects the copula family and the ordered marginal
families to estimate. This is a narrow exception to the usual rule that storage
type parameters are implementation details. It does not expose the fields,
additional representation choices, or arbitrary concrete type parameters of a
constructed `SklarDist`.

::: note Structural models

Most calls identify a model by its type. A structure chosen at runtime, such as
a nested Archimedean tree, cannot be reconstructed from its type alone and is
therefore fitted from a template instance. This distinction also matters when a
bootstrap must reproduce the original estimator. Use `fit(typeof(C0), U)` when
the type completely describes the model and `fit(C0, U)` when the instance
contains the structure to preserve.

:::



## Reading and diagnosing a fitted model

`CopulaModel` implements `StatsBase.StatisticalModel`. Its accessors answer
different questions about the fit:

| Function                                       | Description                                                                                       |
|:--|:--|
| `fitteddistribution(M)`                        | Fitted copula or Sklar distribution.                                                              |
| `nobs(M)`                                      | Number of observations used in the fit.                                                           |
| `deviance(M)`                                  | Deviance, equal to minus twice the fitted log-likelihood.                                         |
| `nullloglikelihood(M)`                         | Log-likelihood under independence with same margins (available for Sklar fits).                   |
| `nulldeviance(M)`                              | Deviance of the null model (−2 · `nullloglikelihood(M)`).                                         |
| `aic(M)` / `bic(M)`                            | Information criteria from `StatsBase.jl`.                                                            |
| `coef(M)` / `coefnames(M)`                     | Estimated parameters and their names.                                                             |
| `vcov(M)`                                      | Parameter variance–covariance matrix (may be `nothing`).                                          |
| `stderror(M)` / `confint(M; level=0.95)`       | Standard errors and Wald confidence intervals; return `nothing` when `vcov(M) === nothing`.       |
| `residuals(M; transform=:uniform \| :normal)`  | Rosenblatt residuals on `[0,1]` or Normal scale when the fit retains the required observations.  |
| `predict(M; what=:cdf\|:pdf\|:simulate, ...)`  | CDF/PDF at `newdata`, or simulation (`nsim`; defaults to `nobs(M)` when non-positive).            |

The table is a reference; in practice, diagnostics are best read together.
Information criteria compare fitted models on the same observations, confidence
intervals describe local parameter uncertainty, and Rosenblatt residuals probe
whether the fitted conditional structure has removed the dependence.

### Examples

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



## Parameter uncertainty

When fitting with `fit(CopulaModel, ...)`, `vcov=true` estimates the covariance
matrix of the fitted parameters. Its diagonal controls standard errors; its
off-diagonal terms describe local dependence between parameter estimates.

!!! note "Default"
    `vcov=true` is the default. Set `vcov=false` for exploratory fits or large
    candidate comparisons when uncertainty is not yet needed.

The `vcov_method` keyword selects the estimator:

| Symbol               | Description                                                                                      |
|:--|:--|
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

Each successful method returns a symmetric positive semi-definite matrix exposed by
`StatsBase.vcov(M)`.

If the requested covariance calculation is numerically unavailable,
`StatsBase.vcov(M)` may be `nothing`.

::: remark Derived dependence measures

In the example above, `derived_measures=false` disables the automatic
calculation of Kendall's τ, Spearman's ρ, Blomqvist's β, Gini's γ, tail
coefficients and entropy. This can reduce computation and memory use, but it
also removes a useful interpretation layer from the printed report.

:::


## Estimating margins and dependence together

For raw observations, fitting a `SklarDist` separates two questions: how each
margin should be estimated, and how the transformed observations should be used
to estimate dependence.

You can pass the `sklar_method` parameter as: 

- `:ifm`: fits parametric margins and maps data to pseudo-scale via their CDFs.  
- `:ecdf`: uses empirical pseudo-observations (ranks).

::: remark IFM or empirical margins?

- Use `sklar_method = :ifm` when margins are plausibly parametric and you want a model-based projection; use `:ecdf` to avoid margin misspecification.
- `margins_kwargs` is a single `NamedTuple` applied to every marginal fit. For heterogeneous options, fit margins manually and then fit the copula on the resulting pseudo-data.
- The model’s `null_ll` (for LR tests) is the log-likelihood under independence with the **same margins**.

:::

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
plot(fitteddistribution(Ŝ))
```



## Choosing an estimating principle

The names and availability of fitting methods depend on the family. Use
`method=:default` unless a family documents a more appropriate explicit method.

The fitting method determines which feature of the sample identifies the
parameters. No method dominates in every family and sample size.

- `:mle` — **Maximum likelihood** over `U`. Recommended when a stable density and a good reparameterization exist.
- `:itau` — **Kendall inverse**: matches theoretical `tau(C)` to empirical `tau(U)`. Ideal for single-parameter families with a monotone inverse.
- `:irho` — **Spearman inverse**: analogous to `rho`; can use scalar or matrix objectives (e.g., multivariate Gaussians).
- `:ibeta` — **Blomqvist inverse**: scalar; only valid for families with **≤ 1** free parameter.
- `:itau_irho` — **joint Kendall/Spearman matching** for a bivariate
  `TCopula`: Kendall's tau determines the correlation parameter and Spearman's
  rho determines the degrees of freedom.

::: property Identifiability of inversion estimators

Rank-based methods cannot identify more free parameters than the matched
coefficients contain independent information. In particular, a single scalar
coefficient cannot generally identify a multi-parameter family; `:ibeta`
enforces this restriction explicitly.

:::

For **extreme-value** copulas, `:mle` / `:iupper` use the documented Pickands
representation when supported by the family.

## When a parametric family is too restrictive

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
