# Mixture models with ExpectationMaximization.jl

[ExpectationMaximization.jl](https://github.com/dmetivie/ExpectationMaximization.jl)
can fit mixture models whose components are distributions from Copulas.jl. The
integration is provided by a package extension, so ExpectationMaximization.jl
remains an optional dependency of Copulas.jl.

Install and load both packages to activate it:

```julia
using Pkg
Pkg.add(["Copulas", "ExpectationMaximization"])
```

The examples below use the convention followed throughout Copulas.jl: one
observation per column of a `d × n` data matrix.

```@example em
using Copulas
using Distributions
using ExpectationMaximization
using Random

rng = MersenneTwister(27)
n = 80
nothing # hide
```

## A mixture as a marginal distribution

The initial `MixtureModel` contains information that its type alone cannot
represent: its components, initial parameters, and weights. Consequently,
`fit_mle` accepts an initialized `SklarDist`. It first fits each margin, uses
the fitted marginal CDFs to construct pseudo-observations, and then fits the
copula. This is the inference-functions-for-margins (IFM) procedure; it is not
a joint maximum-likelihood fit of every parameter in the `SklarDist`.

```@example em
mixture_margin = MixtureModel(
    [Normal(-1.0, 0.7), Normal(1.5, 0.9)],
    [0.4, 0.6],
)

initial_sklar = SklarDist(
    ClaytonCopula(2, 1.0),
    (mixture_margin, LogNormal(0.2, 0.5)),
)

data = rand(rng, initial_sklar, n)
fitted_sklar = fit_mle(
    initial_sklar,
    data;
    copula_kwargs=(; vcov=false, derived_measures=false),
)

typeof(fitted_sklar.m[1])
```

The result remains a single `SklarDist`: the marginal mixture is fitted as a
marginal distribution. It is not expanded into a mixture of `SklarDist`
objects, which would generally define a different dependence model.

## A mixture of copulas

A copula can also be an EM component. During each M-step, the extension fits
the component by weighted maximum likelihood.

```@example em
initial_copula_mixture = MixtureModel(
    [ClaytonCopula(2, 0.5), ClaytonCopula(2, 2.5)],
    [0.5, 0.5],
)

uniform_data = rand(rng, initial_copula_mixture, n)
fitted_copula_mixture = fit_mle(
    initial_copula_mixture,
    uniform_data;
    maxiter=3,
)

components(fitted_copula_mixture)
```

The components may also belong to different copula families, provided each
family supports the generic Copulas.jl MLE parameterization.

## A mixture of complete multivariate models

The same mechanism applies when every EM component is a complete
`SklarDist`. The posterior weights computed by EM are passed first to the
marginal fits and then to the copula fit on the resulting pseudo-observations.
Each component update is therefore weighted IFM, rather than an exact joint
M-step for the complete `SklarDist` likelihood.

```@example em
component1 = SklarDist(
    ClaytonCopula(2, 0.7),
    (Normal(-1.0, 0.8), Normal(0.0, 0.4)),
)
component2 = SklarDist(
    ClaytonCopula(2, 2.2),
    (Normal(1.2, 0.6), Normal(0.8, 0.3)),
)

initial_sklar_mixture = MixtureModel(
    [component1, component2],
    [0.45, 0.55],
)

sklar_data = rand(rng, initial_sklar_mixture, n)
fitted_sklar_mixture = fit_mle(
    initial_sklar_mixture,
    sklar_data;
    maxiter=2,
)

components(fitted_sklar_mixture)
```

Mixture margins can themselves appear inside these `SklarDist` components;
the nested EM fits are dispatched recursively.

!!! note "Weighted fitting"
    ExpectationMaximization.jl supplies posterior observation weights during
    its M-step. A bare copula component is fitted by weighted maximum
    likelihood. A `SklarDist` component is fitted by weighted IFM: its margins
    and its copula each receive the weights, but they are fitted sequentially.
    The method is named `fit_mle` because that is the component-fitting hook
    expected by ExpectationMaximization.jl; the name does not make the
    `SklarDist` update a joint MLE. Consequently, EM mixtures of `SklarDist`
    components do not have the usual monotonic-likelihood guarantee of an
    exact M-step. Other Copulas.jl fitting methods such as `:itau` or `:irho`
    are intentionally not given an implicit weighted interpretation.

## Supported and discrete margins

The extension contains no list of supported marginal families and no
family-specific parameter transformations. It delegates each fit to
`fit_mle(initial_margin, observations[, weights])`. A continuous margin is
therefore supported when the loaded packages provide that method; in
particular, ExpectationMaximization.jl provides it recursively for initialized
`MixtureModel` margins.

Discrete margins are rejected by this IFM entry point. Evaluating their exact
copula likelihood requires copula-CDF differences over rectangles, rather
than the continuous density formula evaluated at marginal CDFs. Treating the
ties as ordinary continuous pseudo-observations would silently solve a
different statistical problem.
