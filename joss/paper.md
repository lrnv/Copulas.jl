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

The Julia package `Copulas.jl` brings most standard copula-related features into native Julia: random number generation, density and distribution function evaluations, fitting, construction of multivariate models through Sklar's theorem, and many more related functionalities. Since copulas can combine arbitrary univariate distributions to form distributions of multivariate random vectors, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API [@djl1; @djl2], the Julian standard for implementation of random variables and random vectors. This compliance allows interoperability with other packages based on this API such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl) [@turing] and several others. 

# Statement of need

The R package `copula` [@r_copula_citation1; @r_copula_citation2; @r_copula_citation3; @r_copula_citation4] is the gold standard when it comes to sampling, estimating, or simply working around dependence structures. However, in other languages, the available tools are not as developed and/or not as recognized. We bridge the gap in the Julian ecosystem with this Julia-native implementation. Due to the very flexible type system in Julia, our code's expressiveness and tidiness will increase its usability and maintainability in the long-run. Type-stability allows sampling in arbitrary precision without requiring more code, and Julia's multiple dispatch yields most of the below-described applications.

There are competing packages in Julia, such as [`BivariateCopulas.jl`](https://github.com/AnderGray/BivariateCopulas.jl) [@BivariateCopulas] which only deals with a few models in bivariate settings but has very nice graphs, or [`DatagenCopulaBased.jl`](https://github.com/iitis/DatagenCopulaBased.jl) [@DatagenCopulaBased_1; @DatagenCopulaBased_2; @DatagenCopulaBased_3; @DatagenCopulaBased_4], which only provides sampling and does not have exactly the same models as `Copulas.jl`. While not fully covering out both of these package's functionality (mostly because the three projects chose different implementation paths), `Copulas.jl` brings, as a key feature, the compliance with the broader ecosystem. The following table provides a feature comparison between the three: 

|                          | `Copulas.jl` | `DatagenCopulaBased.jl` | `BivariateCopulas.jl` |
|--------------------------|--------------|-------------------------|-----------------------|
| `Distributions.jl`'s API | Yes | No | Yes |
| Fitting                  | Yes | No | No |
| Plotting                 | No | No | Yes |
| Available copulas        |     |     |    |
| - Classic Bivariate      | Yes | Yes | Yes |
| - Classic Multivariate   | Yes | Yes | No |
| - Archimedeans           | All of them | Selected ones | Selected ones |
| - Obscure Bivariate      | Yes | No | No |
| - Archimedean Chains     | No | Yes | No |

Since our primary target is maintainability and readability of the implementation, we have not considered the efficiency and the performance of the code yet. However, a (limited in scope) benchmark on Clayton's `pdf` shows competitive behavior of our implementation w.r.t `DatagenCopulaBased.jl` (but not `BivariateCopulas.jl`). To perform this test we use the [`BenchmarkTools.jl`](https://github.com/JuliaCI/BenchmarkTools.jl) [@BenchmarkTools] package and generate 10^6 samples for Clayton copulas of dimensions 2, 5, 10 with parameter 0.8. The execution times (in seconds) are given below: 

|                              | 2         | 5         | 10        |
|------------------------------|-----------|-----------|-----------|
| `Copulas.Clayton`            | 1.1495578 | 1.3448951 | 1.8044065 |
| `BivariateCopulas.Clayton`   | 0.1331608 |         X |         X |
| `DatagenCopulaBased.Clayton` | 1.9868345 | 2.4276321 | 2.8009263 |

Code for these benchmarks in available in the repository.

# Examples

## `SklarDist`: sampling and fitting examples

The `Distributions.jl`'s API provides a `fit` function. You may use it to simply fit a compound model to some dataset as follows: 

```julia
using Copulas, Distributions, Random

# Define the marginals and the copula, then use Sklar's theorem:
X₁ = Gamma(2,3)
X₂ = Pareto(0.5)
X₃ = Normal(10,0.8)
C = ClaytonCopula(3,0.7)
D = SklarDist(C,(X₁,X₂,X₃))

# Sample as follows: 
x = rand(D,1000)

# You may estimate the model as follows: 
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Pareto, Normal}}, x)
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
  D = SklarDist(C,(X₁,X₂,X₃))

  # Add the loglikelihood to the model : 
  Turing.Turing.@addlogprob! loglikelihood(D, dataset)
end
```
Outside the common context we can use the API to work on different aspects, suppose you want to generate random samples of a copula whose parameter depends on a covariate $X$ [@abegaz2012], for example, Consider the Frank copula whose parameter depends on a covariate $X$ that follows a truncated normal distribution with mean zero and variance 9. Also, consider the parameter $\theta(x), x \in [-2,2]$.
```julia
using Distributions, Copulas, DataFrames

x= truncated(Normal(0,3),-2,2) 

function model_1(x)
   return (10-1.5x^2)
end

data = rand(x,10^5)

#Frank Copula whose parameter depends on a covariate
function rFrank(x, C_function, dimension)
  data_list = DataFrame(u1=Float64[], u2=Float64[])
  
  for i in x
    param = C_function(i)
    copula_instance = FrankCopula(dimension, param)
    value_copula = rand(copula_instance, 1)
    push!(data_list, (value_copula[1], value_copula[2]))
  end

  return data_list
end

New_frank = rFrank(data, model_1, 2)
```
In this way we easily obtain random samples of a new bivariate Frank copula whose parameter can depend on any covariate.

## The Archimedean interface

Archimedean copulas form a large class of copulas that has seen a lot of theoretical work. Among others, you may take a look at [@mcneilMultivariateArchimedeanCopulas2009b]. We use [`WilliamsonTransforms.jl`](https://github.com/lrnv/WilliamsonTransforms.jl/)'s implementation of the Williamson $d$-transfrom to sample from any archimedean copula, including for example the `ClaytonCopula` with negative dependence parameter in any dimension, which is a first to our knowledge.

To construct an archimedean copula, you first need to reference its generator through the following API: 

```julia
struct MyGenerator{T} <: Copulas.Generator
    θ::T
end
ϕ(G::MyGenerator,t) = exp(-G.θ * t) # can you recognise this one ?
Copulas.max_monotony(G::MyGenerator) = Inf
C = ArchimedeanCopula(4,MyGenerator(1.3)) # 4-dimensional copula
```

The obtained model automatically gets all copula functionalities (pdf, cdf, sampling, dependence measures, etc...). We nevertheless have specific implementation for a (large) list of known generators, and you may implement some other methods if you know closed form formulas for more performance. The use of the (inverse) Williamson d-transform allows the technical boundaries of our Archimedean implementation to *match* the necessary and sufficient conditions for a generator to produce a genuine Archimedean copula.

## Broader ecosystem

The package is starting to get used in several other places of the ecosystem. Among others, we noted: 

- The package [`GlobalSensitivity.jl`](https://github.com/SciML/GlobalSensitivity.jl) exploits `Copulas.jl` to provide Shapley effects implementation, see [this documentation](https://docs.sciml.ai/GlobalSensitivity/stable/tutorials/shapley/). 
- [`EconomicScenarioGenerators.jl`](https://github.com/JuliaActuary/EconomicScenarioGenerators.jl) uses `Copulas.jl`'s dependence structures to construct multivariate financial assets. 

# Acknowledgement

Santiago Jiménez Ramos thanks FACEPE for the full financing of his postgraduate studies.

# References
