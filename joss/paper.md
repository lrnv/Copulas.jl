---
title: 'Copulas.jl: A fully Distributions.jl-compliant copula package'
tags:
  - julia
  - copula dependence statistics
authors:
  - name: Oskar Laverny
    orcid: 0000-0002-7508-999X
    corresponding: true
    affiliation: 1
  - name: Santiago Jimenez
    orcid: 0000-0002-8198-3656
    affiliation: 2 # must use "" if you have more than one.
affiliations:
 - name: Aix Marseille Univ, Inserm, IRD, SESSTIM, Sciences Economiques & Sociales de la Santé & Traitement de l’Information Médicale, ISSPAM, Marseille, France.
   index: 1
 - name: Federal University of Pernambuco
   index: 2
date: 16 November 2023
bibliography: paper.bib
header-includes:
- |
  ```{=latex}
  \setmonofont[Path=./]{JuliaMono-Regular.ttf}
  ```
---
<!-- LTeX: langage=en -->
# Summary

Copulas are functions that describe dependence structures of random vectors, without describing their univariate marginals. In statistics, the separation is sometimes useful, the quality and/or quantity of available information on these two objects might differ. This separation can be formally stated through Sklar's theorem: 

**Theorem: existence and uniqueness of the copula [@sklar1959fonctions]:** For a given $d$-variate absolutely continuous random vector $\mathbf X$ with marginals $X_1,...X_d$, there exists a unique function $C$, the copula, such that $$F(\mathbf x) = C(F_1(x_1),...,F_d(x_d)),$$ where $F, F_1,...F_d$ are respectively the distributions functions of $\mathbf X, X_1,...X_d$.

Copulas are standard tools in probability and statistics, with a wide range of applications from biostatistics, finance or medicine, to fuzzy logic, global sensitivity and broader analysis. A few standard theoretical references on the matter are [@joe1997], [@nelsen2006], [@joe2014], and [@durantePrinciplesCopulaTheory2015].

The Julia package `Copulas.jl` brings most standard copula-related features into native Julia: random number generation, density and distribution function evaluations, fitting, construction of multivariate models through Sklar's theorem, and many more related functionalities. Copulas being fundamentally distributions of random vectors, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API [@djl1; @djl2], the Julian standard for implementation of random variables and random vectors. This compliance allows interoperability with other packages based on this API such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl) [@turing] and several others. 

# Statement of need

The R package `copula` [@r_copula_citation1; @r_copula_citation2; @r_copula_citation3; @r_copula_citation4] is the gold standard when it comes to sampling, estimating, or simply working around dependence structures. However, in other languages, the available tools are not as developed and/or not as recognized. We bridge the gap in the Julian ecosystem with this Julia-native implementation. Due to the very flexible type system in Julia, our code expressiveness and tidiness will increase its usability and maintainability in the long-run. Type-stability allows sampling in arbitrary precision without requiring more code, and Julia's multiple dispatch yields most of the below-described applications.

There are competing packages in Julia, such as [`BivariateCopulas.jl`](https://github.com/AnderGray/BivariateCopulas.jl) [@BivariateCopulas] which only deals with a few models in bivariate settings but has very nice graphs, or [`DatagenCopulaBased.jl`](https://github.com/iitis/DatagenCopulaBased.jl) [@DatagenCopulaBased_1; @DatagenCopulaBased_2; @DatagenCopulaBased_3; @DatagenCopulaBased_4], which only provides sampling and does not have exactly the same models as `Copulas.jl`. While not fully covering out both of these package's functionality (mostly because the three projects chose different copulas to implement), `Copulas.jl` is clearly the most fully featured, and brings, as a key feature, the compliance with the broader ecosystem.
# Comparison of other packages with copulas.jl
According to the package documentation, it is possible to summarize some of the most important functionalities of each package and those most needed by the community.
## Functionality
The following table shows some characteristics that differentiate each package.
| Characteristic                                | Copulas.jl         | DatagenCopulaBased.jl         | BivariateCopulas.jl          |
|-----------------------------------------------|--------------------|--------------------|--------------------|
| Every Archimedean Copula sampling     | Yes                 | No                 | No                 |
| Multivariate Copula sampling | Yes                | No                 | Yes                 |
| Nested Copula Sampling             | No                 | No                 | Yes    |
| Fitting Copula | Yes | only bivariate case | No |
##  Efficiency
To perform an efficiency test we use the "BenchmarkTools" package with the objective of comparing the execution time and the amount of memory necessary to generate copula samples with each package. We generate 10^6 samples for Clayton copula of dimensions 2, 5, 10 with parameter 0.8
| Package                           | Dimension | Execution Time (seconds) | Memory Usage (bytes) |
|-----------------------------------|-----------|--------------------------------------|-------------------------|
| Copulas.Clayton                   | 2         | 1.1495578e9                                  | 408973296                     |
| Copulas.Clayton                   | 5         | 1.3448951e9                                  | 386723344                     |
| Copulas.Clayton                   | 10        |       1.8044065e9                            |    464100752                  |
| BivariateCopulas.Clayton          | 2         | ...                                  | ...                     |
 | DatagenCopulaBased.Clayton        | 2         |   1.9868345e9                                |  1178800464                    |
| DatagenCopulaBased.Clayton        | 5         |       2.4276321e9                            | 1314855488                     |
| DatagenCopulaBased.Clayton        | 10        |   2.8009263e9                                |   1627164656                   |

# Examples

## `SklarDist`: sampling and fitting examples

The `Distributions.jl`'s API provides a `fit` function. You may use it to simply fit a compound model to some dataset as follows: 

```julia
using Copulas, Distributions, Random

# Define the marginals and the copula, then use Sklar's theorem:
X₁ = Gamma(2,3)
X₂ = Pareto(0.5)
X₃ = Binomial(10,0.8)
C = ClaytonCopula(3,0.7)
X = SklarDist(C,(X₁,X₂,X₃))

# Sample from the model: 
x = rand(D,1000)

# You may estimate the model as follows: 
D̂ = fit(SklarDist{FrankCopula,Tuple{Gamma,Normal,Binomial}}, x)
# Although you'll probbaly get a bad fit !
```

The API does not fix the fitting procedure, and only loosely specifies it, thus the implemented default might vary on the copula. If you want more control, you may turn to Bayesian estimation using `Turing.jl`:  

```julia
using Turing
@model function model(dataset)
  # Priors
  θ ~ TruncatedNormal(1.0, 1.0, 0, Inf)
  γ ~ TruncatedNormal(1.0, 1.0, 0.25, Inf)
  η ~ Beta(1,1)
  δ ~ Exponential(1)

  # Define the model through Sklar's theorem: 
  X₁ = Gamma(2,θ)
  X₂ = Pareto(γ)
  X₃ = Binomial(10,η)
  C = ClaytonCopula(3,δ)
  X = SklarDist(C,(X₁,X₂,X₃))

  # Add the loglikelyhood to the model : 
  Turing.Turing.@addlogprob! loglikelihood(D, dataset)
end
```

## The Archimedean interface

Archimedean copulas are a huge family of copulas that has seen a lot of theoretical work. Among others, you may take a look at [@mcneilMultivariateArchimedeanCopulas2009b]. We use [`WilliamsonTransforms.jl`](https://github.com/lrnv/WilliamsonTransforms.jl/)'s implementation of the Williamson $d$-transfrom to sample from any archimedean copula, including for example the `ClaytonCopula` with negative dependence parameter in any dimension, which is a first to our knowledge.

To construct an archimedean copula, you first need to reference its generator through the following API: 

```julia
struct MyGenerator{T} <: Generator
    θ::T
end
ϕ(G::MyGenerator,t) = exp(-G.θ * t) # can you recognise this one ?
max_monotony(G::MyGenerator) = Inf
C = ArchimedeanCopula(d,MyGenerator())
```

The obtained model automatically gets all copula functionalities (pdf, cdf, sampling, dependence measures, etc...). We nevertheless have specific implementation for a (large) list of known generators, and you may implement some other methods if you know closed form formulas for more performance. The use of the (inverse) Williamson d-transform allows the technical boundaries of our Archimedean implementation to *match* the necessary and sufficient conditions for a generator to produce a genuine Archimedean copula.

## Broader ecosystem

The package is starting to get used in several other places of the ecosystem. Among others, we noted: 

- The package [`GlobalSensitivity.jl`](https://github.com/SciML/GlobalSensitivity.jl) exploits `Copulas.jl` to provide Shapley effects implementation, see [this documentation](https://docs.sciml.ai/GlobalSensitivity/stable/tutorials/shapley/). 
- [`EconomicScenarioGenerators.jl`](https://github.com/JuliaActuary/EconomicScenarioGenerators.jl) uses `Copulas.jl`'s dependence structures to construct multivariate financial assets. 

# References
