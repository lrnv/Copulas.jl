# Copulas

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/Copulas.jl/stable) -->
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
- `GaussianCopula`,
- `TCopula`,
- `ArchimedeanCopula` (for any generator),
- `ClaytonCopula`,`FrankCopula`, `AMHCopula`, `JoeCopula`, `GumbelCopula` as example of the `ArchimedeanCopula` abstract type, see below,
- `WCopula` and `MCopula`, which are [Fréchet-Hoeffding bounds](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Fr%C3%A9chet%E2%80%93Hoeffding_copula_bounds),
- `EmpiricalCopula` to follow closely a given dataset.

The next ones to be implemented will probably be: 
- Nested archimedeans (general, with the possibility to nest any family with any family, assuming it is possible, with parameter checks.)
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
frailty_dist(C::ClaytonCopula) = Distributions.Gamma(1/C.θ,1)      # Radial distribution
```
The Archimedean API is modular: 

- To sample an archimedean, only `frailty_dist` and `ϕ` are needed.
- To evaluate the cdf and (log-)density in any dimension, only `ϕ` and `ϕ⁻¹` are needed.
- Currently, to fit the copula `τ⁻¹` is needed as we use the inverse tau moment method. But we plan on also implementing inverse rho and MLE (density needed). 
- Note that the generator `ϕ` follows the convention `ϕ(0)=1`, while others (e.g., https://en.wikipedia.org/wiki/Copula_(probability_theory)#Archimedean_copulas) use `ϕ⁻¹` as the generator.
- We plan on implementing the Williamson transformations so that `radial-dist` can be automaticlaly deduced from `ϕ` and vice versa, if you dont know much about your archimedean family

# Dev Roadmap

## Urgent things

- [ ] Add tests and documentation

## Next

- [ ] Extensive documentation and tests for the current implementation. 
- [x] Implement archimedean density generally. 
- [ ] Docs: show how to implement another archimedean.  
- [ ] Give the user the choice of fitting method via `fit(dist,data; method="MLE")` or `fit(dist,data; method="itau")` or `fit(dist,data; method="irho")`.
- [ ] Fitting a generic archimedean : should provide an empirical generator
- [ ] Make `Archimedean` more generic : inputing only `frailty_dist` or only `phi` shoudl be enough to get `pdf, cdf, rand, tau, rho, itau, irho, fit, frailty_dist`, etc...  **Williamson d-transform and inverse d-transform should be implemented.** The checking of nesting possibility should be done automatically with some rules (is phi_inv \circ phi complementely monotonous ? with obviously shortcut for inter-family nestings.)   

## Maybe later

- [ ] `Vines`?
- [ ] `NestedArchimedean` and very easy implementation of new archimeean copulas via the radial dist or the phi/invphi + Williamson transform. 
- [ ] `BernsteinCopula` and `BetaCopula` could also be implemented. 
- [ ] `PatchworkCopula` and `CheckerboardCopula`: could be nice things to have :)
- [ ] Goodness of fits tests ?

## Contributions are welcome

Do not hesitate to open an issue to discuss :)
