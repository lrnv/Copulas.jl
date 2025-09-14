```@meta
CurrentModule = Copulas
```

# Home

## Introduction

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

## Installation

The package is available in Julia's General Registry and can be installed as follows:

```julia
] add Copulas
```

## Contributing

!!! tip "Contributions are welcome!"
    If you want to contribute to the package, ask a question, found a bug, or simply want to chat, do not hesitate to open an issue on this repo. General guidelines on collaborative practices (colprac) are available at https://github.com/SciML/ColPrac.

## Cite this work

Please use the following BibTeX if you want to cite this work: 

```bibtex
@article{Laverny2024, 
    doi = {10.21105/joss.06189}, 
    url = {https://doi.org/10.21105/joss.06189}, 
    year = {2024}, 
    publisher = {The Open Journal}, 
    volume = {9}, 
    number = {94}, 
    pages = {6189}, 
    author = {Oskar Laverny and Santiago Jimenez}, 
    title = {Copulas.jl: A fully Distributions.jl-compliant copula package}, journal = {Journal of Open Source Software}
}
```
