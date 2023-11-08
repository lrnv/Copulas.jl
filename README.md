# Copulas

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/Copulas.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lrnv.github.io/Copulas.jl/dev)
[![Build Status](https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lrnv/Copulas.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lrnv/Copulas.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![DOI](https://zenodo.org/badge/456485213.svg)](https://zenodo.org/badge/latestdoi/456485213)

# What is this package ? 

**Warning: This is fairly untested and experimental work and the API might change without notice.**

This package brings most standard [copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)) features into native Julia: random number generation, pdf and cdf, fitting, copula-based multivariate distributions through Sklar's theorem, etc., while fully complying with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API (after all, copulas are distributions functions) in order to provide interoperability with other packages based on this API such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

Usually, people that use and work with copulas turn to R, because of the amazing `R` package [`copula`](https://cran.r-project.org/web/packages/copula/copula.pdf).
While it is still well maintained and regularly updated, the `R` package `copula` is a mixture of obscure, heavily optimized `C` code and more standard `R` code, which makes it a complicated code base for readability, extensibility, reliability and maintenance.

This is an attempt to provide a very light, fast, reliable and maintainable copula implementation in native Julia (which means, in particular, floating point type agnostic, i.e. compatibility with `BigFloat`, [`DoubleFloats`](https://github.com/JuliaMath/DoubleFloats.jl), [`MultiFloats`](https://github.com/dzhang314/MultiFloats.jl) and other kind of numbers). The two most important exported types are: 

- `Copula`: an abstract mother type for all the copulas in the package. 
- `SklarDist`:  allows construction of a multivariate distribution by specifying the copula and the marginals through [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem). 

# What is already implemented

The API contains random number generation, cdf and pdf evaluation, and the `fit` function from `Distributions.jl`. A typical use case might look like this: 

```julia
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

# This generates a (3,1000)-sized dataset from the multivariate distribution D
simu = rand(D,1000)

# While the following estimates the parameters of the model from a dataset: 
D̂ = fit(SklarDist{FrankCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
# Increase the number of observations to get a beter fit (or not?)  
```

Available copula families are:
- `EllipticalCopulas`: `GaussianCopula` and `TCopula`
- `ArchimedeanCopula`: `WilliamsonCopula` (for any generator), but also `ClaytonCopula`,`FrankCopula`, `AMHCopula`, `JoeCopula`, `GumbelCopula`, supporting the full ranges in every dimensions (e.g. ClaytonCopula can be sampled with negative dependence in any dimension, not just d=2). 
- `WCopula`, `IndependentCopula` and `MCopula`, which are [Fréchet-Hoeffding bounds](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds),
- `PlackettCopula`, see ref?
- `EmpiricalCopula` to follow closely a given dataset.

The next ones to be implemented will probably be: 
- Extreme values copulas. 
- Nested archimedeans (for any generators, with automatic nesting conditions checking). 
- Bernstein copula and more general Beta copula as smoothing of the Empirical copula. 
- `CheckerboardCopula` (and more generally `PatchworkCopula`)

Adding a new `ArchimedeanCopula` is very easy. The `Clayton` implementation is as short as: 

```julia
struct ClaytonCopula{d,T} <: Copulas.ArchimedeanCopula{d}
    θ::T
end
ClaytonCopula(d,θ)            = ClaytonCopula{d,typeof(θ)}(θ)     # Constructor
ϕ(C::ClaytonCopula, t)        = (1+sign(C.θ)*t)^(-1/C.θ)          # Generator
ϕ⁻¹(C::ClaytonCopula,t)       = sign(C.θ)*(t^(-C.θ)-1)            # Inverse Generator
τ(C::ClaytonCopula)           = C.θ/(C.θ+2)                       # θ -> τ
τ⁻¹(::Type{ClaytonCopula},τ)  = 2τ/(1-τ)                          # τ -> θ
williamson_dist(C::ClaytonCopula{d,T}) where {d,T} = WilliamsonFromFrailty(Distributions.Gamma(1/C.θ,1),d) # Radial distribution
```
The Archimedean API is modular: 

- To sample an archimedean, only `ϕ` is required. Indeed, the `williamson_dist` has a generic fallback that uses [WilliamsonTransforms.jl](https://www.github.com/lrnv/WilliamsonTransforms.jl) for any generator. Note however that providing the `williamson_dist` yourself if you know it will allow sampling to be an order of magnitude faster.
- To evaluate the cdf and (log-)density in any dimension, only `ϕ` and `ϕ⁻¹` are needed.
- Currently, to fit the copula `τ⁻¹` is needed as we use the inverse tau moment method. But we plan on also implementing inverse rho and MLE (density needed). 
- Note that the generator `ϕ` follows the convention `ϕ(0)=1`, while others (e.g., https://en.wikipedia.org/wiki/Copula_(probability_theory)#Archimedean_copulas) use `ϕ⁻¹` as the generator.

# Dev Roadmap

## Next

- [ ] More documentation and tests for the current implementation. 
- [ ] Docs: show how to use the WilliamsonCopula to implement generic archimedeans.
- [ ] Give the user the choice of fitting method via `fit(dist,data; method="MLE")` or `fit(dist,data; method="itau")` or `fit(dist,data; method="irho")`.
- [ ] Fitting a generic archimedean with an empirically produced generarator
- [ ] Automatic checking of generator d-monotonicity ? Dunno if it is even possible. 

## Maybe later

- [ ] `NestedArchimedean`, with automatic checking of nesting conditions for generators. 
- [ ] `Vines`?
- [ ] `Archimax` ?
- [ ] `BernsteinCopula` and `BetaCopula` could also be implemented. 
- [ ] `PatchworkCopula` and `CheckerboardCopula`: could be nice things to have :)
- [ ] Goodness of fits tests ?

## Contributions are welcome

Do not hesitate to open an issue to discuss :)
