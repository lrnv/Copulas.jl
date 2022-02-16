# Copulas

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://lrnv.github.io/Copulas.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://lrnv.github.io/Copulas.jl/dev)
[![Build Status](https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/lrnv/Copulas.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/lrnv/Copulas.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

# What is this package ? 

This packages aims at bringing into native Julia most of the standard copula code that is yet unavailable in native Julia. Usually, people that use and work with copulas turn to R and not Julia to work, because of the amaising package `copula` that is available there. 

However, I beleive that this package, while still maintained correclty today, is quite old and rotting, full of heavily optimized and obscure C code.. This is an attempt to provide a very fast, relaliable and maintainable copula implmentation in Julia, that would embrace the Julian way (and, in particular, the possiblity to work with arbitrary type of floats like `Float32` for speed, `BigFLoats` or `DoubleFloats` or `MultiFloats`), correct SIMD'sation, etc. 

Morover, we want and try to implement the `Copula` objects and the `SklarDist` objects to comply fully with the `Distributions.jl` API. Somehting like this should work : 

```julia
using Distributions, Random
X₁ = Gamma(2,3)
X₂ = Normal(1,3)
X₃ = LogNormal(0,1)
C = FranckCopula(0.7,3) # A 3-variate Franck Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # A (3,1000)-sized datast that correspond from the simulation
fit(D,simu) # Would use Distributions.fit() on the marginals and a MLE on the Franck copula. 
fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,LogNormal}) # Increase the number of observtions to get a beter fit !  



ϕ(t) = e^(-t) # An archimedean generator
C = ArchimedeanCopula(ϕ,4) # Do you know which 4-variate copula it is ? 
SC = flipmargins(C,dims=(1,2,3,4)) # This function would reverese marginal, for any copula, making a survival copula. 
fit(SC,rand((4,1000))) # would fit the SC copula onto random independant observations ! 

using StatsPlots
plot(X₁) # plots the random variable
plot(C) # equivalent plot of the theoretical copula
plot(D) # plot of the multivariate distribution obtain via SklarDist. 

# .... Many other things ! 
```

# Developemment plans

The package should have two main types : 

- The `Copula` abstract type. 
- The `SklarDist` abstract type, representing a multivariate distribution contructed from a `Copula` and many `Distributions.UnivariateDistributions` representing the margins. 

The `Copula` abstract type is then subtyped:

- `ElipticalCopula` which is then subtyped `GaussianCopula` and `StudentCopula`
- `ArchimedeanCopula` which is then subtyped `Franck`, `Clayton`, `Gumbel`, etc. : it should be easy to provide another one : just add a few methods implementing the interface such as phi, phi_inv, radial_dist, tau, rho, itau, irho, cdf, pdf, etc... with of course defaults methods implemented : phi and phi_inv should suffice, the reste could be defined from it by defautl, but more efficient if you implement it yourself.
- `NestedArchimedeanCopula` with a generic nesting algorithm. The checking of nesting possibility should be done automatically with some rules (is phi_inv \circ phi complementely monotonous or something like that ? maybe some rules can help check that.) 
- `EmpiricalCopula` : just produce the empirical copula from a dataset; 
- `BernsteinCopula` and `BetaCopula` could also be implemented. 
- `PatchworkCopula` and `CheckerboardCopula`: could be nice things to have :)


In term of methods, there should be for each `Copula` : 

- A `cdf` method, and `pdf` method, a `Distributions.rand` implementation to comply with the `Distributions.jl` standard
- A `tau` and `rho` method
- A `fit` method that uses maximum likelyhood or itau or irho or something like that to fit the copula to pseudo-data. 

And for the `SklarDist`: 

- An easy construction sheme: input the copula and the marginals; 
- Random genreration, `cdf` and `pdf` methods that comply with the `Distributions.jl` standards, and other methods to comply with this standard. 
- A `fit` method that would use MLE or itau/irho/other + fitting margins through Distributions.jl

Complying with `Distributions.jl` standards by considering that a copula is, after all, a multivariate distribution (continuous in most of the cases), allows to use seemlessly the methods from `StatsPlots.jl` to plot things out, and everyhting else that is already implemented for the Distributions.jl stuff: maybe some other packages like MonteCarloMeasurements could use it directly ? Stan ? I do not know if it is based on Distributions.jl...


For archimedean copulas, 

- Automatic transformation to get the radial_dist from phi and/or phi from the radial dist ? Maybe the archimedean copula implemntation could be centered on this `radial dist` concept: give the radial distribution and we compute everything else ourselves. 

Some goodness of fit tests ? Anything els that is not at parity with the `copula` R package ?

A Vine implementation that beats the C++ equivalents ? 

Else ? 

## First step

The following should be enough for the first public release: 
- Restrict to only `GaussianCopula` and `StudentCopula` at first
- Make the `Copula` and `SklarDist` objets work with `pdf`,`cdf`,`rand!`, with full compatiblity with `Distribution.jl`
- Implement `fit` with a marginal-first scheme that relies on `Distribution.jl`, and fits the multivariate normal or multivariate student from the pseudo-observations pushed in gausian or student space (easy scheme). 
- Add one archimedean just to say so.

## Second step

- Add `Archimedean` with a fully generic implementation (input via radil dist or via phi ?), `pdf, cdf, rand, tau, rho, itau, irho, fit, radial_dist`, Williamson d-transform and inverse d-transform, etc.. 
- Provide `Franck`, `Gumbel` and `Clayton` as examples with nice parametric overloads.
- Show in the docs how easy it is to implement your own archimedean, with pointers to methods to overload.  
- For the `fit`, looks like `itau` and `irho` are easier to implement.

## Next steps

- `NestedArchimedean` and very easy implementation of new archimeean copulas via the radial dist or the phi/invphi + Williamson transform. 
- Implement tau and rho more generally and itau/irho methods to fit the copulas, or MLE, with the choice given to the user via `fit(dist,data; method="MLE")` or `fit(dist,data; method="itau")` or `fit(dist,data; method="irho")` or others... 
- `EmpiricalCopula`, `BernsteinCopula`, `BetaCopula`, `PatchworkCopula` and `CheckerboardCopula`: could be nice things to have :)
- Generic Archimedean fitting ?  Fully generic fitting of Archimedean copula with a given radial_dist : the archimedean copula with a `Gamma()` riadial dist could be fitted by looking at the parameter of the gamma that makes the distribution match, no ? Throuhg Williamson transformaiton and everything... 
- `Vines` ? Maybe later. 
- Others ? 