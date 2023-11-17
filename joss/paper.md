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
  - name: YOUR NAME
    orcid: 0000-0000-0000-0000
    affiliation: "2, 3" # must use "" if you have more than one.
affiliations:
 - name: SESSTIM, Aix Marseille University, Marseille, France
   index: 1
 - name: YOUR FIRST AFFILIATION
   index: 2
 - name: YOUR SECOND AFFILIATION
   index: 3
date: 16 November 2023
bibliography: paper.bib

---
<!-- LTeX: language=en -->
# Summary

Copulas are functions that describe dependence structures of random vectors, without describing their univariate marginals. In statistics, it is sometimes useful to separate the two, the quality and/or quantity of available information on these two objects might differ. This separation can be formally stated through Sklar's theorem: 

**Theorem: existance and uniqueness of the copula [@sklar1959fonctions]:** For a given $d$-variate absolutely continuous random vector $\mathbf X$ with marginals $X_1,...X_d$, there exists a unique function $C$, the copula, such that $$F(\mathbf x) = C(F_1(x_1),...,F_d(x_d)),$$ where $F, F_1,...F_d$ are respectively the distributions functions of $\mathbf X, X_1,...X_d$.

Copulas are broadly used in probability and statistics, with a wide range of applications from biostatistics, finance or medecine, to fuzzy logic, global sensitivity and broader analysis. A few standard theoretical references on the matter are [@joe1997], [@nelsen2006], [@joe2014], and [@durantePrinciplesCopulaTheory2015].

The Julia package `Copulas.jl` brings most standard [copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)) features into native Julia: random number generation, density and distribution function evaluations, fitting, construction of multivariate models through Sklar's theorem, and many more related functionalities. Copulas being fundamentally random vectors, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API [@djl1; @djl2], the Julian standard for implementation of random variables and random vectors. This complience allows interoperability with other packages based on this API such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl) [@turing] and several others. 

# Statement of need

The R package `copula` [@r_copula_citation1; @r_copula_citation2; @r_copula_citation3; @r_copula_citation4] is the gold standard when it comes to sampling, estimating, or simply working around dependence structures. However, in other languages the situation is not as good. We bridge the gap in the Julian ecosystem with this package, that provides a Julia-native implementation of several computational routines. Due to the very flexible type system in Julia, our code as a never-seen-before expressiveness, which increase its usability and maintenability. type-stability also allows to sample in arbitrary precision without requiering more code. The several applications proposed below are mostly arising from the multiple dispatch principles of Julia.

There are competing packages in Julia, such as [`BivariateCopulas.jl`](https://github.com/AnderGray/BivariateCopulas.jl) which only deals with a few models in bivariate settings but has very nice graphs, or [`DatagenCopulaBased.jl`](https://github.com/iitis/DatagenCopulaBased.jl), which only provides sampling and does not have exactly the same models as `Copulas.jl`. While not fully covering out both of these package's functionality, `Copulas.jl` is clearly the must fully featured, and brings the complience with the broader ecosystem, a key feature.

# Examples

## `SklarDist`, sampling and fitting

We can exploit the `fit` function, which is part of the `Distributions.jl`'s API, to simply fit a compound model to some dataset as follows: 

```julia
using Copulas, Distributions, Random

# Define the marginals : 
X₁ = Gamma(2,3)
X₂ = Pareto(0.5)
X₃ = Binomial(10,0.8)

# Define the copula: 
C = ClaytonCopula(3,0.7)

# Use Sklar's theorem: 
X = SklarDist(C,(X₁,X₂,X₃))

# Sample from the model: 
x = rand(D,1000)

# You may even estimate the model all at once as follows: 
D̂ = fit(SklarDist{FrankCopula,Tuple{Gamma,Normal,Binomial}}, x)
# Although you'll probbaly get a bad fit !
```

The default fitting precedure, as the API requests, is not very specific and might vary from a copula to the other. If you want more control, you may turn to bayesian estimation using `Turing.jl` [@turing]:  

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

Archimedean copulas are a huge family of copulas that has seen a lot of theoretical work. Among others, you may take a look at [@mcneilMultivariateArchimedeanCopulas2009b]. The package implements the Williamson-d-transfrom sampling scheme for all archimedean copulas, including for example the `ClaytonCopula` with negative dependence parameter in any dimension, which is a first to our knowledge. 

The implementation of the clayton copula is actualy only a few lines of code: 

```julia
struct ClaytonCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
    function ClaytonCopula(d,θ)
        if θ < -1/(d-1)
            throw(ArgumentError("Theta must be greater than -1/(d-1)"))
        elseif θ == -1/(d-1)
            return WCopula(d)
        elseif θ == 0
            return IndependentCopula(d)
        elseif θ == Inf
            return MCopula(d)
        else
            return new{d,typeof(θ)}(θ)
        end
    end
end
ϕ(  C::ClaytonCopula,      t) = max(1+C.θ*t,zero(t))^(-1/C.θ)
ϕ⁻¹(C::ClaytonCopula,      t) = (t^(-C.θ)-1)/C.θ
τ(C::ClaytonCopula) = ifelse(isfinite(C.θ), C.θ/(C.θ+2), 1)
τ⁻¹(::Type{ClaytonCopula},τ) = ifelse(τ == 1,Inf,2τ/(1-τ))
williamson_dist(C::ClaytonCopula{d,T}) where {d,T} = C.θ >= 0 ? WilliamsonFromFrailty(Distributions.Gamma(1/C.θ,C.θ),d) : ClaytonWilliamsonDistribution(C.θ,d)
```

But if you want to sample from your own archimedean copula, not all the methods are necessary: 

```julia
# Simply subset ArchimedeanCopula and provide the generator
struct MyUnknownArchimedean{d,T} <: ArchimedeanCopula{d}
    θ::T
end
ϕ(C::MyUnknownArchimedean,t) = exp(-t*C.θ)

# You can now sample
C = MyUnknownCopula{2,Float64}(3.0)
spl = rand(C,1000)

# evaluate the cdf: 
cdf(C,[0.5,0.5])

# or even the loglikelyhood : 
loglikelyhood(C,spl)
```

The automatic sampling uses [`WilliamsonTransformations.jl`](https://github.com/lrnv/WilliamsonTransforms.jl/) to compute the radial part automatically from an unknown generator. 

## Broader ecosystem

The package is starting to get used in several other places of the ecosystem. Among others, we noted: 

- The package [`GlobalSensitivity.jl`](https://github.com/SciML/GlobalSensitivity.jl) exploit `Copulas.jl` to provide Shapley effects implementation, see [this documentation](https://docs.sciml.ai/GlobalSensitivity/stable/tutorials/shapley/). 
- [`EconomicScenarioGenerators.jl`](https://github.com/JuliaActuary/EconomicScenarioGenerators.jl) uses depndence structures between financial assets. 


# Acknowledgments

<!-- If you have to Acknowledge some fundings that might be here. I dont think I do.  -->


# References