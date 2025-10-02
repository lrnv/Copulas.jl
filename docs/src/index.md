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

# Welcome to Copulas.jl!

The [Copulas.jl](https://github.com/lrnv/Copulas.jl) package provides a large collection of models for dependence structures of real random vectors, known as [copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)), with a wide selection of features:
- random number generation
- evaluation of (log)density and distribution functions
- copula-based multivariate distributions via Sklar's theorem
- fitting procedures, including marginal models
- evaluation of dependence metrics

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

```@example
using Copulas, Distributions, Random, Plots
X₁ = Gamma(2,3)
X₂ = Beta(1,4)
X₃ = Normal()
C = ClaytonCopula(3,5.2) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # Generate a dataset
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,Normal}}, simu) # estimate a model
plot(D̂) # plot the result
```

The list of availiable copula models is *very* large, check it out on our [documentation](https://lrnv.github.io/Copulas.jl/stable) ! 
The general implementation philosophy is for the code to follow the mathematical boundaries of the implemented concepts. For example, this is the only implementation we know (in any language) that allows for **all** Archimedean copulas to be sampled: we use the Williamson transformation for non-standard generators, including user-provided black-box ones.

## Feature comparison


There are competing packages in Julia, such as [`BivariateCopulas.jl`](https://github.com/AnderGray/BivariateCopulas.jl) which only deals with a few models in bivariate settings but has very nice graphs, or [`DatagenCopulaBased.jl`](https://github.com/iitis/DatagenCopulaBased.jl), which only provides sampling and does not have exactly the same models as `Copulas.jl`. Since recently, we cover both of these packages’ functionalities completely, while still bringing, as a key feature, compliance with the broader ecosystem. The following table provides a feature comparison between the three: 

|                          | `Copulas.jl`            | `DatagenCopulaBased.jl` | `BivariateCopulas.jl` |
|--------------------------|-------------------------|-------------------------|-----------------------|
| `Distributions.jl`'s API | ✔️                      | ❌                     | ✔️                    |
| Fitting                  | ✔️                      | ❌                     | ❌                    |
| Plotting                 | ✔️                      | ❌                     | ✔️                    |
| Conditioning             | ✔️                      | ❌                     | ⚠️ Bivariate Only     |
| Available copulas        |                          |                        |                       |
| - Classic Bivariate      | ✔️                      | ✔️                     | ✔️                    |
| - Obscure Bivariate      | ✔️                      | ❌                     | ❌                    |
| - Classic Multivariate   | ✔️                      | ✔️                     | ❌                    |
| - Archimedeans           | ✔️ All of them          | ⚠️ Selected ones       | ⚠️Selected ones       |
| - Extreme Value Copulas  | ⚠️ Bivariate only       | ❌                     | ❌                    |
| - Archimax               | ⚠️ Bivariate only       | ❌                     | ❌                    |
| - Archimedean Chains     | ❌                      | ✔️                     | ❌                    |

Since our primary target is maintainability and readability of the implementation, we did not consider the efficiency and the performance of the code yet. Proper benchmarks will come in the near future. 

## Quick API Tour 

Here is a quick, practical tour of the public API. It shows how to construct copulas, build Sklar distributions, compute dependence metrics, subset and condition models, use Rosenblatt transforms, and fit models. For background, theory, details, and model descriptions, see the Manual.

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
Dj  = condition(C2, 2, 0.3)  # distortion of U₁ | U₂ = 0.3 (d=2)
Distributions.cdf(Dj, 0.95)
Dc  = condition(D, (2,3), (0.3, 0.2))
rand(Dc, 2)
```

### Fitting

Fit both marginals and copula from raw data (Sklar):

```@example 1
X = rand(D, 500)
M = fit(CopulaModel, SklarDist{GumbelCopula, Tuple{Gamma,Beta,LogNormal}}, X; copula_method=:mle)
```

Directly fit a copula from pseudo-observations U:

```@example 1
U = pseudos(X)
Ĉ = fit(GumbelCopula, U; method=:itau)
```

Notes
- `fit` chooses a reasonable default per family; pass `method`/`copula_method` to control it.
- Common methods: copulas `:mle`, `:itau`, `:irho`, `:ibeta`; Sklar `:ifm` (parametric CDFs) and `:ecdf` (pseudo-observations).


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
