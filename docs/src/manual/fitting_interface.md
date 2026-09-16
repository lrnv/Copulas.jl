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

::: definition Maximum likelihood and maximum pseudo-likelihood

Maximum likelihood (`method=:mle`) maximizes the copula density over values
already observed on the uniform copula scale. Maximum pseudo-likelihood
(`method=:mpl`) first replaces raw marginal observations by their empirical
ranks and then maximizes the same numerical objective. The point optimizer is
the same, but the two estimators make different assumptions about how the
uniform observations were obtained.

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
not enough. `CopulaModel` deliberately retains only four ingredients: the
fitted distribution, the original data supplied by the user, the fitted
log-likelihood, and the minimal recipe needed to replay the estimator:

```@example fitting_interface
M = fit(CopulaModel, GumbelCopula, U; method=:mle)
```

The fitted distribution is available through the standard model interface;
the printed report summarizes the estimator, likelihood, information criteria,
parameters and available dependence measures. Transformed observations, the
independence likelihood and parameter blocks are reconstructed only when an
accessor needs them. Optimizer traces and convergence diagnostics are not model
state: an optimizer-backed estimator either returns an accepted fit or throws.

::: remark Two levels of interface

`fit(Family, U)` is deliberately lightweight and returns only the fitted
copula. `fit(CopulaModel, Family, U)` fits the same model while retaining the
statistical evidence and estimator specification. The latter is not a different
estimator unless different keywords are supplied.

:::


## Information criteria

`CopulaModel` deliberately exposes AIC, BIC, AICc, and HQC for every fitted
model, not only for ordinary maximum-likelihood estimates. This is a 1.0 API
choice: the functions are always operationally defined from the fitted model's
stored log-likelihood, number of fitted parameters, and number of observations.

For a fitted model `M`, with ``\ell`` the stored log-likelihood, ``k`` the value
of `dof(M)`, and ``n`` the value of `nobs(M)`, Copulas.jl uses

```math
\mathrm{AIC} = -2\ell + 2k,
```

```math
\mathrm{BIC} = -2\ell + k\log n,
```

with the package's existing AICc and HQC corrections applied analogously.
Automatic candidate selection uses exactly the same scores and remains
available for every supported fitting method.

### Statistical interpretation

!!! warning "Outside ordinary MLE these are operational comparison scores"
    When `M` comes from ordinary maximum likelihood for the likelihood being
    scored, the usual classical interpretation of AIC/BIC/AICc/HQC applies.
    For other estimators, Copulas.jl still evaluates the same algebraic score at
    the fitted parameters, but that number does **not** automatically inherit
    the classical MLE asymptotic justification.

This distinction applies in particular to:

- maximum pseudo-likelihood (`method=:mpl`), where empirical ranks are formed
  before maximizing the copula density;
- inversion estimators such as `:itau`, `:irho`, and `:ibeta`, which do not
  maximize the likelihood at all;
- sequential `SklarDist` IFM fitting, which is not joint maximum likelihood for
  all copula and marginal parameters;
- `SklarDist` ECDF/rank fitting and related semiparametric procedures.

The fitted log-likelihood remains useful in all of these cases, and the common
penalized score is useful for a stable comparison interface. Comparisons should
still be made between models evaluated on the same observations and the same
likelihood contribution.

For composite/pseudo-likelihood inference, estimator-specific information
criteria use a bias correction involving sensitivity and variability (Godambe
or sandwich) quantities rather than blindly substituting the ordinary AIC
penalty; see Varin and Vidoni [varin2005composite](@cite). For two-stage copula
estimation, Ko and Hjort develop a Copula Information Criterion (CIC) that
accounts for the IFM/two-stage structure [ko2019copula](@cite).

These adapted criteria are **not** implemented as part of the 1.0 contract.
Future implementations should use names that make their statistical meaning
explicit—for example a composite-likelihood information criterion using the
appropriate Godambe correction, and CIC (or a closely related criterion) for
IFM. They must not silently change the established operational meaning of
`aic`, `bic`, `aicc`, or `hqc` on `CopulaModel`.

### Examples

```@example information_criteria
using Copulas, Distributions, StatsBase

U = [
    0.12 0.31 0.54 0.73 0.89 0.42
    0.81 0.22 0.63 0.47 0.15 0.68
]

Mmle = fit(CopulaModel, ClaytonCopula, U; method=:mle)
Mτ   = fit(CopulaModel, ClaytonCopula, U; method=:itau)

(aic(Mmle), bic(Mmle), aic(Mτ), bic(Mτ))
```

The second pair remains valid API output. It should be read as the documented
penalized fitted-log-likelihood score at the Kendall-inversion estimate, not as
a claim that the classical MLE derivation of AIC/BIC applies unchanged to that
estimator.


## Comparing candidate families

Choosing a family is part of modelling, not a consequence of optimization.
When several scientifically defensible families remain, they can be fitted to
the same data and compared by an information criterion.

::: definition Information-criterion selection

For each candidate family, fit a model and compute a penalized likelihood
criterion. The selected candidate is the successful fit with the smallest
eligible finite criterion. AIC emphasizes estimated predictive loss;
BIC and HQC penalize model dimension more strongly as the sample grows, while
AICc corrects AIC in small samples.

:::

When the copula family is unknown, an explicit collection of candidate families
can be compared automatically. This returns a `CopulaSelection`, keeping the
comparison report separate from the winning `CopulaModel`:

```@example fitting_interface
Ctrue = ClaytonCopula(2, 4.0)
data = rand(Ctrue, 300)

Msel = fit(
    CopulaModel,
    Copulas.Copula,
    data;
    candidates=(ClaytonCopula, GumbelCopula, FrankCopula),
    criterion=:bic,
)
Msel
```

Retrieve the reusable fitted model with [`selected_model`](@ref):

```@example fitting_interface
Mbest = selected_model(Msel)
fitted_distribution(Mbest)
```

The available criteria are:

- `:bic` — Bayesian information criterion,
- `:aic` — Akaike information criterion,
- `:aicc` — finite-sample corrected AIC,
- `:hqc` — Hannan–Quinn criterion.

BIC is the default criterion.

The winning fit is reused. Selection itself performs no uncertainty
calculation.

The complete comparison can be inspected with [`selection_table`](@ref):

```@example fitting_interface
selection_table(Msel)
```

Each row stores the candidate family, fitting status and method,
log-likelihood, number of parameters, and all four information criteria.
Candidates that fail to fit can be skipped with `on_error=:skip` (the default)
or propagated immediately with `on_error=:throw`.

::: warning Candidate lists are scientific assumptions

The candidate collection is intentionally explicit. Automatic selection does
not make every implemented family plausible for every dimension, tail regime or
scientific question. Nonfinite scores and failed fits are excluded;
`on_error=:throw` is useful when a failed candidate should invalidate
the comparison rather than merely be recorded in `selection_table`.

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
direct copula calls default to `method=:mle`; explicitly supported alternatives
depend on the family.

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
| `fitted_distribution(M)`                        | Fitted copula or Sklar distribution.                                                              |
| `nobs(M)`                                      | Number of observations used in the fit.                                                           |
| `loglikelihood(M)`                             | Cached log-likelihood evaluated at the fitted distribution.                                       |
| `deviance(M)`                                  | Deviance, equal to minus twice the fitted log-likelihood.                                         |
| `nullloglikelihood(M)`                         | Lazily computed log-likelihood under independence, preserving fitted margins for Sklar models.    |
| `nulldeviance(M)`                              | Deviance of the null model (−2 · `nullloglikelihood(M)`).                                         |
| `aic(M)` / `bic(M)`                            | Information criteria from `StatsBase.jl`.                                                            |
| `coef(M)` / `coefnames(M)`                     | Estimated parameters and their names.                                                             |
| `residuals(M; transform=:uniform \| :normal)`  | Rosenblatt residuals on `[0,1]` or Normal scale when the fit retains the required observations.  |

The table is a reference; in practice, diagnostics are best read together.
Information criteria compare fitted models on the same observations, while
Rosenblatt residuals probe whether the fitted conditional structure has removed
the dependence. Parameter uncertainty belongs to a separate inference result.

### Examples

```@example fitting_interface
# Information criteria
StatsBase.aic(M)
StatsBase.bic(M)
```

```@example fitting_interface
# Rosenblatt residuals
R  = StatsBase.residuals(M; transform=:uniform)
RN = StatsBase.residuals(M; transform=:normal)
(size(R), size(RN))
```


## Inference after estimation

Fitting and uncertainty quantification are separate operations. `fit` produces
the point estimate and records how it was obtained; it never computes a
covariance matrix. Apply [`infer`](@ref) to that fitted model when uncertainty
is needed:

| Symbol               | Description                                                                                      |
|:--|:--|
| `:hessian`           | Inverse observed information (−Hessian of the log-likelihood). Default for `method = :mle`.     |
| `:godambe`           | Scalar-moment Godambe for supported bivariate rank-matching fits; accepts `nresamples` and `rng`.  |
| `:godambe_pairwise`  | Pairwise-moment Godambe for supported multivariate rank fits; accepts `nresamples` and `rng`.     |
| `:jackknife`         | Leave-one-out refitting of the complete recorded estimator.                                      |
| `:bootstrap`         | Bootstrap refitting of the complete recorded estimator; accepts `nresamples` and `rng`.          |

The default is `:hessian` after supported maximum-likelihood fits,
`:godambe` after supported bivariate rank-matching fits, and
`:godambe_pairwise` when a supported multivariate rank estimator is defined by
pairwise moments. A downstream fitting extension does not implicitly opt into
analytical inference. There is no generic silent fallback: if a method is
mathematically unavailable or its sensitivity matrix is rank deficient,
`infer` throws and the user must choose another procedure explicitly.

Maximum pseudo-likelihood currently has no implicit covariance method. A
sandwich estimator must reflect the rank preprocessing and is tracked
separately; use an explicit full-estimator bootstrap in the meantime when that
procedure is appropriate for the analysis.

```@example fitting_interface
I = infer(M)
StatsBase.vcov(I)
StatsBase.stderror(I)
StatsBase.confint(I; level=0.95)
```

The same fitted model can be passed to several inference procedures without
optimizing it again or mutating it. Bootstrap and Godambe procedures that use
resampling accept explicit controls, for example
`infer(M; method=:bootstrap, nresamples=500, rng=Xoshiro(42))` or
`infer(M; method=:godambe, nresamples=500, rng=Xoshiro(42))`.


## Estimating margins and dependence together

For raw observations, fitting a `SklarDist` separates two questions: how each
margin should be estimated, and how the transformed observations should be used
to estimate dependence.

You can pass the `sklar_method` parameter as:

- `:ifm`: fits parametric margins and maps data to pseudo-scale via their CDFs.  
- `:ecdf`: uses empirical pseudo-observations (ranks).

The default is `sklar_method=:ifm`. In either route the copula step defaults to
`copula_method=:mle` whenever that estimator is supported. Empirical or
extension-defined families without MLE retain their first advertised method.
The default can be replaced by any method supported by the chosen family, such
as `:itau` or `:irho`.

::: remark IFM or empirical margins?

- Use `sklar_method = :ifm` when margins are plausibly parametric and you want a model-based projection; use `:ecdf` to avoid margin misspecification.
- `margins_kwargs` is a single `NamedTuple` applied to every marginal fit. For heterogeneous options, fit margins manually and then fit the copula on the resulting pseudo-data.
- `nullloglikelihood(M)` reconstructs independence lazily and preserves the **same fitted margins**.
- Neither route jointly maximizes the complete Sklar likelihood: IFM is
  sequential, while ECDF estimates dependence from ranks.

:::

```@example fitting_interface
S = SklarDist(ClaytonCopula(2, 5), (Normal(), LogNormal(0, 0.5)))
X = rand(S, 300)
Ŝ = fit(CopulaModel, SklarDist{ClaytonCopula,Tuple{Normal,LogNormal}}, X;
	sklar_method=:ifm, # or :ecdf
	copula_method=:default, # MLE when available; otherwise the family's default
	margins_kwargs=NamedTuple(), copula_kwargs=NamedTuple()) # options will be passed down to fitting functions. 
Ŝ
```

For a fitted Sklar model, `infer(Ŝ)` defaults to a full-estimator bootstrap.
Each resample starts from `X`, refits every margin, rebuilds the transformed
sample according to `sklar_method`, and then refits the copula. Thus the full
covariance includes uncertainty in the margins, uncertainty in the copula, and
their cross-covariances:

`coef(Ŝ)` and `coefnames(Ŝ)` use the same order: copula parameters first, then
the parameters of each margin in coordinate order. This ordering is what makes
the covariance blocks interpretable.

```@example fitting_interface
Isklar = infer(Ŝ; method=:bootstrap, nresamples=10, rng=Xoshiro(43))
Vall = StatsBase.vcov(Isklar)
Vcopula = StatsBase.vcov(Isklar; component=:copula)
Vmargins = StatsBase.vcov(Isklar; component=:margins)
(size(Vall), size(Vcopula), size(Vmargins))
```

Analytical Sklar covariance methods throw explicitly. The arbitrary estimators
behind `Distributions.fit` for marginal families do not provide a common score,
Hessian, or parameter-transform contract from which such formulas could be
derived safely.

```@example fitting_interface
plot(fitted_distribution(Ŝ))
```



## Choosing an estimating principle

The names and availability of fitting methods depend on the family. Direct
parametric copula fitting defaults to `method=:mle` whenever MLE is available;
choose another estimator explicitly. Structural and empirical families without an MLE
retain the first method registered internally for that family. This preserves family-defined defaults
without ever selecting `:mpl` implicitly; estimator registration and execution
remain internal.

The fitting method determines which feature of the sample identifies the
parameters. No method dominates in every family and sample size.

- `:mle` — **Maximum likelihood** over `U`. Recommended when a stable density and a good reparameterization exist.
- `:mpl` — **Maximum pseudo-likelihood**. With `pseudo_values=false`, raw
  observations are converted by `pseudos` before the copula likelihood is
  maximized. This method is available whenever `:mle` is available, but is
  never selected by default.
- `:itau` — **Kendall inverse**: matches theoretical `tau(C)` to empirical `tau(U)`. Ideal for single-parameter families with a monotone inverse. For the elliptical families it is closed form in every dimension: each entry of the correlation matrix is `sinpi(τ̂/2)` of the corresponding pairwise sample coefficient, repaired to the nearest positive-definite correlation matrix when the pairwise entries are not jointly consistent. For `TCopula` the degrees of freedom are then the maximizer of the likelihood with that correlation held fixed.
- `:irho` — **Spearman inverse**: analogous to `rho`; can use scalar or matrix objectives. For `GaussianCopula` it is the closed form `2 sinpi(ρ̂_S/6)` entrywise.
- `:ibeta` — **Blomqvist inverse**: scalar; only valid for families with **≤ 1** free parameter.
- `:itau_irho` — **joint Kendall/Spearman matching** for a bivariate
  `TCopula`: Kendall's tau determines the correlation parameter and Spearman's
  rho determines the degrees of freedom.

The two likelihood names are kept consistent with the preprocessing request.
Asking for `method=:mle, pseudo_values=false` silently records the effective
method as `:mpl`. Asking for `method=:mpl, pseudo_values=true` records `:mle`
and warns, because no rank transformation occurs.

::: remark Why there is no full Sklar MLE yet

A generic joint optimizer would need a smooth unconstrained parameterization
for every requested marginal family. `Distributions.jl` does not expose enough
information to derive such mappings from `params` and constructors: marginal
parameters may be positive, bounded, ordered, matrix-valued, mutually
constrained, or may alter the support. Guessing those constraints would make a
nominally generic method unreliable. A future API extension can add full Sklar
MLE once marginal families can explicitly provide this optimization protocol;
the continuous, discrete and mixed-margin likelihood cases must also be
distinguished.

There is a second, statistical limitation: `Distributions.fit` is a common
entry point, not a universal promise that every marginal family uses maximum
likelihood. Its estimator is chosen by the individual distribution
implementation and is not exposed to Copulas.jl through a stable protocol; for
some families it may use another fitting principle altogether. Consequently,
independently calling `fit` on every margin neither identifies a joint MLE nor
even guarantees that every marginal block was estimated by marginal MLE.

:::

::: property Identifiability of inversion estimators

Rank-based methods cannot identify more free parameters than the matched
coefficients contain independent information. In particular, a single scalar
coefficient cannot generally identify a multi-parameter family; `:ibeta`
enforces this restriction explicitly.

:::

For **extreme-value** copulas, `:mle` / `:iupper` use the documented Pickands
representation when supported by the family.

### Weighted observations

Both likelihood estimators accept one weight per observation:

```julia
w = exp.(-0.01 .* (n:-1:1))            # exponential decay, any positive scale
M = fit(CopulaModel, ClaytonCopula, U; method=:mle, weights=w)
C = fit(GaussianCopula, X; pseudo_values=false, weights=w)
```

The fit maximizes the weighted pseudo-likelihood `∑ᵢ wᵢ log c(uᵢ)`. The
weights are normalized once so that they sum to the number of observations
`n`, which fixes their scale without changing the maximizer: a weight reads as
"how many observations this column counts for", uniform weights reproduce the
unweighted fit exactly, a zero weight removes its observation, and integer
weights summing to `n` give the fit of the sample in which each observation is
repeated that many times. The stored log-likelihood, and with it `aic`, `bic`
and `deviance`, are the weighted ones; `nobs` stays `n`. With
`pseudo_values=false` the rank transformation is the weighted one of
`pseudos(X; weights)`, which ranks each margin by weighted mass under the same
tie conventions.

Weights are an estimation choice, not an inference one. A rank-inversion
method refuses them, the Sklar route refuses them because its margins are
fitted unweighted, and `infer` and the composite goodness-of-fit tests refuse
a weighted model: the covariance of a weighted pseudo-likelihood estimator
depends on whether the weights are frequencies, importance ratios or a decay
schedule, and no such estimator is implemented.

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
you can call `coef`, `aic`, `bic`, `deviance`, and `residuals`, and obtain a
full `CopulaModel` with the same documented model interface. Use
`fitted_distribution(M)` for distribution operations such as CDF evaluation or
simulation.

The multivariate `EmpiricalEVCopula` projection may contain singular spectral
components and therefore has no global Lebesgue density. Its quick `fit`
interface, CDF, and sampler remain available, but likelihood-based summaries
are not applicable.

The only difference is that empirical models are **parameter-free** (`dof(M) = 0`),
so finite-dimensional parameter inference is unavailable and information
criteria reduce to AIC = BIC = −2 · loglikelihood.
Otherwise, all features—including the computation of dependence measures and the REPL summary—behave exactly the same.
