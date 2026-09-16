```@meta
CurrentModule = Copulas
```

# Information criteria

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

## Statistical interpretation

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

## Examples

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
