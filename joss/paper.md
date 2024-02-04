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

---
<!-- LTeX: language=en -->
# Summary

Copulas are functions that describe dependence structures of random vectors, without describing their univariate marginals. In statistics, the separatiopn is sometimes usefull, the quality and/or quantity of available information on these two objects might differ. This separation can be formally stated through Sklar's theorem: 

**Theorem: existance and uniqueness of the copula [@sklar1959fonctions]:** For a given $d$-variate absolutely continuous random vector $\mathbf X$ with marginals $X_1,...X_d$, there exists a unique function $C$, the copula, such that $$F(\mathbf x) = C(F_1(x_1),...,F_d(x_d)),$$ where $F, F_1,...F_d$ are respectively the distributions functions of $\mathbf X, X_1,...X_d$.

Copulas are standard tools in probability and statistics, with a wide range of applications from biostatistics, finance or medecine, to fuzzy logic, global sensitivity and broader analysis. A few standard theoretical references on the matter are [@joe1997], [@nelsen2006], [@joe2014], and [@durantePrinciplesCopulaTheory2015].

The Julia package `Copulas.jl` brings most standard copula-related features into native Julia: random number generation, density and distribution function evaluations, fitting, construction of multivariate models through Sklar's theorem, and many more related functionalities. Copulas being fundamentally distributions of random vectors, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API [@djl1; @djl2], the Julian standard for implementation of random variables and random vectors. This complience allows interoperability with other packages based on this API such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl) [@turing] and several others. 

# Statement of need

The R package `copula` [@r_copula_citation1; @r_copula_citation2; @r_copula_citation3; @r_copula_citation4] is the gold standard when it comes to sampling, estimating, or simply working around dependence structures. However, in other languages, the available tools are not as developped and/or not as recognised. We bridge the gap in the Julian ecosystem with this Julia-native implementation. Due to the very flexible type system in Julia, our code expressiveness and tidyness will increase its usability and maintenability in the long-run. Type-stability allows sampling in arbitrary precision without requiering more code, and Julia's multiple dispatch yields most of the below-described applications.

There are competing packages in Julia, such as [`BivariateCopulas.jl`](https://github.com/AnderGray/BivariateCopulas.jl) [@BivariateCopulas] which only deals with a few models in bivariate settings but has very nice graphs, or [`DatagenCopulaBased.jl`](https://github.com/iitis/DatagenCopulaBased.jl) [@DatagenCopulaBased_1; @DatagenCopulaBased_2; @DatagenCopulaBased_3; @DatagenCopulaBased_4], which only provides sampling and does not have exactly the same models as `Copulas.jl`. While not fully covering out both of these package's functionality (mostly because the three projects chose different copulas to implement), `Copulas.jl` is clearly the must fully featured, and brings, as a key feature, the complience with the broader ecosystem.

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

The API does not fix the fitting procedure, and only loosely specify it, thus the implemented default might vary on the copula. If you want more control, you may turn to bayesian estimation using `Turing.jl` [@turing]:  

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

## The Archimedean API

Archimedean copulas are a huge family of copulas that has seen a lot of theoretical work. Among others, you may take a look at [@mcneilMultivariateArchimedeanCopulas2009b]. We use [`WilliamsonTransformations.jl`](https://github.com/lrnv/WilliamsonTransforms.jl/)'s implementation of the Williamson $d$-transfrom to sample from any archimedean copula, including for example the `ClaytonCopula` with negative dependence parameter in any dimension, which is a first to our knowledge.

The API is consisting of the folloiwng functions: 

```julia
ϕ(C::MyArchimedean, t) # Generator
williamson_dist(C::MyArchimedean) # Williamson d-transform
```

So that implementing your own archimedean copula only requires to subset the `ArchimedeanCopula` type and provide your generator as follows: 
```julia
struct MyUnknownArchimedean{d,T} <: ArchimedeanCopula{d}
    θ::T
end
ϕ(C::MyUnknownArchimedean,t) = exp(-t*C.θ)
```

The obtained model can be used as follows: 
```julia
C = MyUnknownCopula{2,Float64}(3.0)
spl = rand(C,1000)   # sampling
cdf(C,spl)           # cdf
pdf(C,spl)           # pdf
loglikelihood(C,spl) # llh
```

The following functions have defaults but can be overridden for performance: 

```julia
ϕ⁻¹(C::MyArchimedean, t) # Inverse of ϕ
ϕ⁽¹⁾(C::MyArchimedean, t) # first defrivative of ϕ
ϕ⁽ᵈ⁾(C::MyArchimedean,t) # dth defrivative of ϕ
τ(C::MyArchimedean) # Kendall tau
τ⁻¹(::Type{MyArchimedean},τ) = # Inverse kendall tau
fit(::Type{MyArchimedean},data) # fitting.
```

## Broader ecosystem

The package is starting to get used in several other places of the ecosystem. Among others, we noted: 

- The package [`GlobalSensitivity.jl`](https://github.com/SciML/GlobalSensitivity.jl) exploit `Copulas.jl` to provide Shapley effects implementation, see [this documentation](https://docs.sciml.ai/GlobalSensitivity/stable/tutorials/shapley/). 
- [`EconomicScenarioGenerators.jl`](https://github.com/JuliaActuary/EconomicScenarioGenerators.jl) uses depndence structures between financial assets. 


# Acknowledgments

<!-- If you have to Acknowledge some fundings that might be here. I dont think I do.  -->


# References
