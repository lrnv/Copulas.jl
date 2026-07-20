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
`fit_mle` accepts an initialized `SklarDist` and jointly optimizes the copula,
the marginal parameters, and the mixture probabilities against the complete
`SklarDist` likelihood. The instance supplies both the model structure and the
starting point of this numerical optimization.

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
fitted_sklar = fit_mle(initial_sklar, data)

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
`SklarDist`. Its margins and copula are jointly updated using the posterior
weights computed by EM. Thus the component update maximizes the same weighted
joint likelihood that appears in the M-step; it is not a separate marginal fit
followed by an IFM copula fit.

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
their components and probabilities are parameterized recursively and included
in the same joint optimization.

!!! note "Weighted fitting"
    ExpectationMaximization.jl supplies posterior observation weights during
    its M-step. The extension implements this component update using maximum
    likelihood. Other Copulas.jl fitting methods such as `:itau` or `:irho`
    are intentionally not given an implicit weighted interpretation.

## Supported margins and discrete data

Joint likelihood optimization requires an unconstrained parameterization from
which every candidate distribution can be reconstructed. The extension
currently provides one for `Normal`, `LogNormal`, `LogitNormal`, `Cauchy`,
`Gumbel`, `Laplace`, `Logistic`, `Beta`, `BetaPrime`, `FDist`, `Gamma`,
`InverseGaussian`, `Pareto`, `Weibull`, `Chisq`, `Exponential`, `Rayleigh`,
`TDist`, and `Uniform` margins. Continuous `MixtureModel` margins composed of
these distributions are supported recursively. Their initial probabilities
must be strictly positive. An informative `ArgumentError` is thrown for a
continuous margin whose parameterization is not yet implemented.

Discrete margins are intentionally rejected. For continuous margins, the
joint density factors into the copula density evaluated at marginal CDFs and
the product of the marginal densities. That formula is not valid when a margin
has atoms: the corresponding probability mass requires copula-CDF differences
over rectangles. `SklarDist` and its current `logpdf` implementation model the
continuous case, so silently applying the continuous formula to a Poisson,
categorical, or discrete-mixture margin would not be a valid likelihood.

The optimizer starts from the supplied instance and may reach a local rather
than global maximum, as is usual for mixture likelihoods. Marginal CDF values
that round numerically to zero or one are moved just inside the unit interval
before evaluating the copula density.
