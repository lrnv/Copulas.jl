````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Copulas.jl
  text:
  tagline: A Distributions.jl-compliant copula package.
  image:
    src: logo.svg
    alt: Copulas.jl
  actions:
    - theme: brand
      text: Getting started
      link: /manual/intro
    - theme: alt
      text: View on Github
      link: https://github.com/lrnv/Copulas.jl
    - theme: alt
      text: Bestiary
      link: /bestiary/elliptical
---
````

<!-- This file is generated from README.md by docs/sync_homepage.jl. -->

# Welcome to Copulas.jl!

The [Copulas.jl](https://github.com/lrnv/Copulas.jl) package provides a large collection of models for dependence structures of real random vectors, known as [copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)), with a wide selection of features:
- random number generation
- evaluation of (log)density and distribution functions
- copula-based multivariate distributions via Sklar's theorem
- fitting procedures, model diagnostics, and automatic family selection
- dependence metrics and tail coefficients
- marginalization, conditioning, and Rosenblatt transforms
- resampling-based hypothesis tests for dependence assumptions and goodness of fit

Since copulas are distribution functions, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API. This compliance allows direct interoperability with other packages based on this API, such as [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

Usually, users who work with copulas turn to the `R` package [`copula`](https://cran.r-project.org/web/packages/copula/copula.pdf). While still well-maintained and regularly updated, the `R` package `copula` is a complicated code base in terms of readability, extensibility, reliability, and maintenance.
This package aims to provide a lightweight, fast, reliable, and maintainable copula implementation in native Julia. Among other benefits, a notable feature of such a native implementation is floating point type agnosticism, i.e., compatibility with `BigFloat`, [`DoubleFloats`](https://github.com/JuliaMath/DoubleFloats.jl), [`MultiFloats`](https://github.com/dzhang314/MultiFloats.jl), and other numeric types.


The package revolves around two main types:

- `Copula`, the abstract supertype of all copulas
- `SklarDist`, the type for multivariate compound distributions via [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem)

## Getting started

The package is registered in Julia's General registry so you may simply install the package by running :

```julia
] add Copulas
```

The API contains random number generation, cdf and pdf evaluation, and the `fit` function from `Distributions.jl`. A typical use case might look like this:

```@example home_getting_started
using Copulas, Distributions, Random, Plots
X₁ = Gamma(2,3)
X₂ = Beta(1,4)
X₃ = Normal()
C = ClaytonCopula(3,5.2) # A 3-variate Clayton copula with θ = 5.2
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # Generate a dataset
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,Normal}}, simu) # estimate a model
plot(D̂) # plot the result
```

The list of available copula models is *very* large; browse the [Bestiary](https://lrnv.github.io/Copulas.jl/stable/bestiary/elliptical) for definitions, parameterizations, constructors, and model-specific caveats.
The general implementation philosophy is for the code to follow the mathematical boundaries of the implemented concepts. For example, this is the only implementation we know (in any language) that allows for **all** Archimedean copulas to be sampled: we use the Williamson transformation for non-standard generators, including user-provided black-box ones.

## Feature comparison


Other Julia packages cover related use cases. [`BivariateCopulas.jl`](https://github.com/AnderGray/BivariateCopulas.jl) focuses on a compact set of bivariate copulas, joint distributions, conditioning, and visualization. [`DatagenCopulaBased.jl`](https://github.com/iitis/DatagenCopulaBased.jl) focuses on data generation, including selected nested and chained constructions. The comparison below summarizes capabilities documented by each project; “Not documented” means that the linked public README does not advertise the feature, not that no implementation can exist.

| Capability | `Copulas.jl` | `DatagenCopulaBased.jl` | `BivariateCopulas.jl` |
|:--|:--|:--|:--|
| Sampling | ✅ Vector and matrix `rand` interface | ✅ Primary interface | ⚠️ Bivariate |
| `Distributions.jl` distribution API | ✅ | ❌ Not documented | ⚠️ Bivariate |
| CDF and density | ✅ When defined by the model | ❌ Not documented as a common interface | ⚠️ Bivariate |
| Copula plus arbitrary margins | ✅ `SklarDist`, any supported dimension | ⚠️ Marginal-transformation utilities | ⚠️ Bivariate joint distributions |
| Parameter fitting | ✅ Quick fits and full `CopulaModel` results | ❌ Not documented | ❌ Not documented |
| Automatic family selection | ✅ Explicit candidate sets with AIC, BIC, AICc, or HQC | ❌ Not documented | ❌ Not documented |
| Statistical-model diagnostics | ✅ Covariance, confidence intervals, residuals, prediction, information criteria | ❌ Not documented | ❌ Not documented |
| Dependence measures | ✅ Scalar and pairwise rank and tail measures | ⚠️ Empirical Kendall-correlation examples | ❌ Not documented as a common interface |
| Subsetting and conditioning | ✅ Copulas and `SklarDist`; univariate or multivariate results | ❌ Not documented | ⚠️ Bivariate conditional CDFs |
| Rosenblatt and inverse Rosenblatt transforms | ✅ | ❌ Not documented | ❌ Not documented |
| Hypothesis tests | ✅ Independence, exchangeability, radial symmetry, extreme-value dependence, goodness of fit | ❌ Not documented | ❌ Not documented |
| Plot recipes | ✅ Pairwise samples, margins, CDF/PDF contours and surfaces | ❌ Not documented | ⚠️ Bivariate scatter, CDF, density and contour plots |
| Archimedean models | ✅ Clayton, Frank, Gumbel, Joe, AMH, inverse Gaussian, BB1--BB10, and custom/empirical generators | ⚠️ Selected families, same-family nesting, and chains | ⚠️ Clayton and Frank |
| Structured multivariate models | ✅ Liouville and nested Archimedean copulas | ✅ Same-family nested copulas and bivariate chains | ❌ Not documented |
| Elliptical models | ✅ Gaussian and Student, multivariate | ✅ Gaussian and Student | ⚠️ Gaussian, bivariate |
| Extreme-value models | ✅ Logistic, Galambos, Hüsler--Reiss, extremal-``t``, Tawn, asymmetric and spectral families; bivariate and multivariate | ⚠️ Marshall--Olkin | ❌ Not documented |
| Nonparametric copulas | ✅ Empirical, beta, Bernstein, checkerboard, empirical EV | ❌ Not documented | ❌ Not documented |
| Archimax models | ⚠️ Generic bivariate construction, BB4 and BB5 | ❌ Not documented | ❌ Not documented |

The table compares public scope rather than runtime performance; algorithmic cost depends strongly on the family, dimension, and requested operation.

## Quick API Tour

Here is a practical tour of the main public workflows. For precise behavioral guarantees see the [Public API](https://lrnv.github.io/Copulas.jl/stable/api/public); for theory and model-specific guidance see the Manual and Bestiary.

### Copulas and Sklar distributions

You can construct a copula object with their respective constructors. They behave like multivariate distributions from `Distributions.jl` and respect their API:

```@example 1
using Copulas, Distributions, Random, StatsBase
# A 3-variate Clayton copula
C = ClaytonCopula(3, 2.0)
U = rand(C, 5)
Distributions.loglikelihood(C, U)
```

To build multivariate distributions, you can compose a copula with marginals via Sklar’s theorem:

```@example 1
X₁, X₂, X₃ = Gamma(2,3), Beta(1,5), LogNormal(0,1)
C2 = GumbelCopula(3, 1.7)
D  = SklarDist(C2, (X₁, X₂, X₃))
rand(D, 3)
pdf(D, rand(3))
```

### Dependence metrics

You can get scalar dependence metrics at copula level:

```@example 1
(
    kendall_tau = Copulas.τ(C),
    spearm_rho = Copulas.ρ(C),
    blomqvist_beta = Copulas.β(C),
    gini_gamma = Copulas.γ(C),
    entropy_iota = Copulas.ι(C),
    lower_tail_dep = Copulas.λₗ(C),
    upper_tail_dep = Copulas.λᵤ(C)
)
```

Pairwise matrices of bivariate versions are available through `StatsBase.corkendall(C)`, `StatsBase.corspearman(C)`, `Copulas.corblomqvist(C)`, `Copulas.corgini(C)`, `Copulas.corentropy(C)`, `Copulas.corlowertail(C)`, and `Copulas.coruppertail(C)`.

Same functions work passing a dataset instead of the copula for their empirical counterpart.

### Measure and transforms

The `measure` function measures hypercubes under the distribution of the copula. You can access the Rosenblatt transformation of a copula (or a Sklar distribution) through the `rosenblatt` and `inverse_rosenblatt` functions:

```@example 1
Copulas.measure(C, (0.1,0.2,0.3), (0.9,0.8,0.7))
x = rand(D, 100)
u = rosenblatt(D, x)
x2 = inverse_rosenblatt(D, u)
maximum(abs.(x2 .- x))
```

### Subsetting and conditioning

You can subset the dimensions of a model through `subsetdims()`, and you can condition a model on some of its marginals with `condition()`:

```@example 1
S23 = subsetdims(C2, (2,3))
StatsBase.corkendall(S23)
Dj  = condition(C2, 2, 0.3)  # Distributions of (U₁, U₃) | U₂ = 0.3 (d=2)
Distributions.cdf(Dj, [0.95, 0.80])
Dc  = condition(D, (2,3), (0.3, 0.2))
rand(Dc, 2)
```

### Fitting and automatic family selection

Fit both marginals and copula from raw data (Sklar):

```@example 1
X = rand(D, 150)
M = fit(CopulaModel, SklarDist{GumbelCopula, Tuple{Gamma,Beta,LogNormal}}, X; copula_method=:mle)
```

Directly fit a copula from pseudo-observations U:

```@example 1
U = pseudos(X)
Ĉ = fit(GumbelCopula, U; method=:itau)
```

Notes
- Direct copula fits default to `method=:mle`. Use `method=:mpl,
  pseudo_values=false` for maximum pseudo-likelihood from raw observations.
- Sklar fits default to sequential `sklar_method=:ifm`; `:ecdf` is the
  rank-based alternative. Neither route is joint full maximum likelihood.
- Their copula step defaults to `copula_method=:mle`, replaceable by another
  family-supported method such as `:itau` or `:irho`.

Use `CopulaModel` when diagnostics and inference matter. If the family is not
known in advance, fit an explicit, scientifically appropriate candidate set and
rank successful fits by an information criterion:

```@example 1
Msel = fit(
    CopulaModel,
    Copulas.Copula,
    U;
    candidates=(ClaytonCopula, GumbelCopula, FrankCopula),
    criterion=:bic,
    vcov=false,
)
selectiontable(Msel)
```

Selection is deliberately explicit: Copulas.jl does not treat every available
family as a sensible candidate for every dimension or scientific question. See
the [fitting interface](https://lrnv.github.io/Copulas.jl/stable/manual/fitting_interface) for covariance estimation,
confidence intervals, residuals, prediction, and selection caveats.

### Hypothesis testing

The test constructors share the `StatsAPI.HypothesisTest` interface and return a
`CopulaTest` queried with `teststatistic`, `pvalue`, and `nobs`. Available
procedures assess mutual independence, exchangeability, radial symmetry, the
extreme-value property, and goodness of fit to a specified fitted family.

```@example 1
test = IndependenceCopulaTest(U; N=19, rng=Xoshiro(42))
(statistic=teststatistic(test), pvalue=pvalue(test), observations=nobs(test))
```

These are resampling-based procedures. Set an RNG for reproducibility, use a
larger `N` for scientific work, and check the assumptions—especially continuity
and absence of ties—on the [hypothesis-testing page](https://lrnv.github.io/Copulas.jl/stable/manual/hypothesis_testing).


## Contributions are welcome

If you want to contribute to the package, ask a question, found a bug or simply want to chat, do not hesitate to open an issue on [the Copulas.jl repository](https://github.com/lrnv/Copulas.jl)


## Citation

Do not hesitate to star this repository to show support. If you use this package in your researches, please cite it with the following bibtex code:

```bibtex
@article{LavernyJimenez2024,
    author = {Oskar Laverny and Santiago Jimenez},
    title = {Copulas.jl: A fully Distributions.jl-compliant copula package},
    journal = {Journal of Open Source Software},
    doi = {10.21105/joss.06189},
    url = {https://doi.org/10.21105/joss.06189},
    year = {2024},
    publisher = {The Open Journal},
    volume = {9},
    number = {94},
    pages = {6189}
}
```
