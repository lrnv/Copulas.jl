```@meta
CurrentModule = Copulas
```

## Introduction

The [Copulas.jl](https://github.com/lrnv/Copulas.jl) package provides a large collection of models for dependence structures between random variables, so-called [copulas](https://en.wikipedia.org/wiki/Copula_(probability_theory)), with a good selection of features: 
- random number generation
- evaluation of (log)density and distribution functions
- copula-based multivariate distributions through Sklar's theorem
- fitting procedures, including marginal models or not

Since copulas are distribution functions, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API. This compliance allows direct interoperability with other packages based on this API such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

Usually, people that use and work with copulas turn to the `R` package [`copula`](https://cran.r-project.org/web/packages/copula/copula.pdf). While still well-maintained and regularly updated, the `R` package `copula` is a complicated code base for readability, extensibility, reliability, and maintenance.
This is an attempt to provide a very light, fast, reliable and maintainable copula implementation in native Julia. Among others, one of the notable benefits of such a native implementation is the floating point type agnosticism, i.e., compatibility with `BigFloat`, [`DoubleFloats`](https://github.com/JuliaMath/DoubleFloats.jl), [`MultiFloats`](https://github.com/dzhang314/MultiFloats.jl) and other kind of numbers.


The package revolves around two main types: 

- `Copula`, the abstract mother type of all copulas
- `SklarDist`, the type for multivariate compound distribution through [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem). 

## Installation

The package is available on Julia's general registry, and can be installed as follows: 

```julia
] add Copulas
```

## Contributing

**Contributions are welcomed !** If you want to contribute to the package, ask a question, found a bug or simply want to chat, do not hesitate to open an issue on this repo. General guidelines on collaborative practices (colprac) are available at https://github.com/SciML/ColPrac.

## Cite this work

Please use the following BibTeX if you want to cite this work: 

```bibtex
@software{oskar_laverny_2023_10084669,
  author       = {Oskar Laverny},
  title        = {Copulas.jl: A fully `Distributions.jl`-compliant copula package},
  year         = 2022+,
  doi          = {10.5281/zenodo.6652672},
  url          = {https://doi.org/10.5281/zenodo.6652672}
}
```