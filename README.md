# Copulas

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/Copulas.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lrnv.github.io/Copulas.jl/dev)
[![Build Status](https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lrnv/Copulas.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lrnv/Copulas.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

# What is this package ? 

**Warning: This is fairly untested and experimental work and the API might change without notice.**

This package aims at bringing into native Julia most of the standard copula features: random number generation, fitting, construction of copula-based multivariate distributions through Sklar's theorem, etc. while fully complying with the `Distributions.jl` API (after all, copulas are distributions functions). 

Usually, people that use and work with copulas turn to R, because of the amazing `R` package `copula`.
While still perfectly maintained and updated today, the `R` package `copula` is full of obscured, heavily optimized, fast `C` code on one hand, and obscure, heavily optimized slow `R` code on the other hand.

This is an attempt to provide a very light, fast, reliable and maintainable copula implementation in native Julia (in particular, type-agnostic, so it'll work with arbitrary type of floats like `Float32` for speed, `BigFloats` or `DoubleFloats` or `MultiFloats` for precision), with correct SIMD'sation, etc. 

Two of the exported types are of most importance: 

- `Copula` : this is an abstract mother type for all our copulas. 
- `SklarDist` : Allows to construct a multivariate distribution by specifying the copula and the marginals, through Sklar's theorem. 

# What is already implemented

The API we implemented contains random number generations, cdf and pdf evaluations, and the `fit` function from `Distributions.jl`. Typical use case might look like this: 

```julia
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Franck Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

# This generates a (3,1000)-sized dataset from the multivariate distribution D
simu = rand(D,1000)

# While the following estimates the parameters of the model from a dataset : 
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
# Increase the number of observations to get a beter fit !  
```

Atop from the very neat `SklarDist` type, available copulas are :
- `EmpiricalCopula`
- `GaussianCopula`
- `TCopula`
- `ArchimedeanCopula` (general, for any generator)
- `ClaytonCopula`,`FranckCopula`, `AMHCopula`, `JoeCopula`, `GumbelCopula` as exemple instantiations of the `ArchimedeanCopula` abstract type, see after

Next ones to be implemented will probably be : 
- More archimedean copulas ? Some with non-completely cmonotone generators ? 
- Nested archimedeans (general, with the possibility to nest any family with any family, assuming it is possible, with parameter checks.)
- Bernstein copula and more general Beta copula as smoothing of the Empirical copula. 
- `CheckerboardCopula` (and more generally `PatchworkCopula`)

Adding a new `ArchimedeanCopula` is very easy. The `Clayton` implementation is as short as : 

```julia
struct ClaytonCopula{d,T} <: ArchimedeanCopula{d}
    θ::T
end
ClaytonCopula(d,θ)            = ClaytonCopula{d,typeof(θ)}(θ)     # Constructor
ϕ(C::ClaytonCopula, t)        = (1+sign(C.θ)*t)^(-1/C.θ)          # Generator
ϕ⁻¹(C::ClaytonCopula,t)       = sign(C.θ)*(t^(-C.θ)-1)            # Inverse Generator
τ(C::ClaytonCopula)           = C.θ/(C.θ+2)                       # θ -> τ
τ⁻¹(::Type{ClaytonCopula},τ)  = 2τ/(1-τ)                          # τ -> θ
radial_dist(C::ClaytonCopula) = Distributions.Gamma(1/C.θ,1)      # Radial distribution
```
The Archimedean API is modular : 

- To sample an archimedean, only `radial_dist` and `ϕ` are needed.
- To evaluate the cdf, only `ϕ` and `ϕ⁻¹` are needed
- Currently to fit the copula, `τ⁻¹` is needed as we use inversed tau moment method. But we plan on also implementing inverse rho and MLE (density needed). 

# Dev Roadmap

## First step (current)

The following should be enough for the first public release: 

- [x] Restrict to only `GaussianCopula` and `StudentCopula` at first
- [x] Make the `Copula` and `SklarDist` objets work with `pdf`,`cdf`,`rand!`, with full compatiblity with `Distribution.jl`
- [x] Implement `fit` with a marginal-first scheme that relies on `Distribution.jl`, and fits the multivariate normal or multivariate student from the pseudo-observations pushed in gausian or student space (easy scheme). 
- [x] Implement some archimedean copulas
- [ ] Add tests and documentation !!!! May take a while. 

## Second step

- [ ] Extensive documentation and tests for the current implementation. 
- [ ] Make `Archimedean` more generic : inputing only `radial_dist` or only `phi` shoudl be enough to get `pdf, cdf, rand, tau, rho, itau, irho, fit, radial_dist`, etc...  **Williamson d-transform and inverse d-transform should be implemented.**
- [ ] Provide archimedean density. 
- [ ] Show in the docs how easy it is to implement your own archimedean, with pointers to methods to overload.  
- [ ] Implement tau and rho more generally and itau/irho methods to fit the copulas, or MLE, with the choice given to the user via `fit(dist,data; method="MLE")` or `fit(dist,data; method="itau")` or `fit(dist,data; method="irho")` or others... 
- [ ] Fitting a generic archimedean : should provide an empirical generator 

## Next steps

- [ ] `NestedArchimedean` and very easy implementation of new archimeean copulas via the radial dist or the phi/invphi + Williamson transform. The checking of nesting possibility should be done automatically with some rules (is phi_inv \circ phi complementely monotonous ? with obviously shortcut for inter-family nestings.)  
- [ ] `BernsteinCopula` and `BetaCopula` could also be implemented. 
- [ ] `PatchworkCopula` and `CheckerboardCopula`: could be nice things to have :)
- [ ] `Vines` ?
- [ ] Goodness of fits tests ?

# Detailed Developemment plans

In term of methods, there should be for each `Copula` : 

- The `cdf`, `_logpdf` and `rand`/`rand!` methods for the `Distributions.jl` API.
- A `tau` and `rho` method, with `itau` and `irho` potentials ? 
- A `fit` method that uses maximum likelyhood ? or itau ? or irho ? or something else ? 

And for the `SklarDist`: 
- An easy construction sheme: input the copula and the marginals; 
- Random genreration, `cdf` and `pdf` methods that comply with the `Distributions.jl` standards, and other methods to comply with this standard. 
- A `fit` method that would use MLE or itau/irho/other + fitting margins through Distributions.jl

Complying with `Distributions.jl` standards by considering that a copula is, after all, a multivariate distribution (continuous in most of the cases), allows to use seemlessly the methods from `StatsPlots.jl` to plot things out, and everyhting else that is already implemented for the Distributions.jl stuff: maybe some other packages like MonteCarloMeasurements could use it directly ? Stan ? I do not know if it is based on Distributions.jl...