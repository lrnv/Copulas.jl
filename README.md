<h1 align=center><img src="https://cdn.rawgit.com/lrnv/Copulas.jl/main/docs/src/assets/logo.svg" width="30px" height="30px"/> Copulas.jl</h1>
<p align=center><i>A fully `Distributions.jl`-compliant copula package</i></p>

<p align=center>
    <a href="https://lrnv.github.io/Copulas.jl/stable"><img src="https://img.shields.io/badge/docs-stable-blue.svg" alt="Stable" /></a>
    <a href="https://lrnv.github.io/Copulas.jl/dev"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Dev" /></a>
    <a href="https://joss.theoj.org/papers/98fa5d88d0d8f27038af2da00f210d45"><img src="https://joss.theoj.org/papers/98fa5d88d0d8f27038af2da00f210d45/status.svg"></a>
<!--     <a href="https://zenodo.org/badge/latestdoi/456485213"><img src="https://zenodo.org/badge/456485213.svg" alt="DOI" /></a> -->
<br />
    <a href="https://www.repostatus.org/#active"><img src="https://www.repostatus.org/badges/latest/active.svg" alt="Project Status: Active – The project has reached a stable, usable state and is being actively developed." /></a>
     <a href="https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml?query=branch%3Amain"><img src="https://github.com/lrnv/Copulas.jl/actions/workflows/CI.yml/badge.svg?branch=main" alt="Build Status" /></a>
     <a href="https://codecov.io/gh/lrnv/Copulas.jl"><img src="https://codecov.io/gh/lrnv/Copulas.jl/branch/main/graph/badge.svg"/></a>
     <a href="https://github.com/JuliaTesting/Aqua.jl"><img src="https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg" alt="Aqua QA" /></a>
    <!-- <a href="https://benchmark.tansongchen.com/TaylorDiff.jl"><img src="https://img.shields.io/buildkite/2c801728055463e7c8baeeb3cc187b964587235a49b3ed39ab/main.svg?label=benchmark" alt="Benchmark Status" /></a> -->
<br />
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT" /></a>
    <a href="https://github.com/SciML/ColPrac"><img src="https://img.shields.io/badge/contributor's%20guide-ColPrac-blueviolet" alt="ColPrac: Contributor's Guide on Collaborative Practices for Community Packages" /></a>
    <a href="https://github.com/invenia/BlueStyle"><img src="https://img.shields.io/badge/code%20style-blue-4495d1.svg" alt="Code Style: Blue" /></a>
</p>

<!-- <p align=center>
  <a href="https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=563952901&machine=standardLinux32gb&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=EastUshttps://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=563952901&machine=standardLinux32gb&devcontainer_path=.devcontainer%2Fdevcontainer.json&location=EastUs"><img src="https://github.com/codespaces/badge.svg" alt="Open in GitHub Codespaces" /></a>
</p> -->

`Copulas.jl` brings most standard [copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)) features into native Julia: random number generation, pdf and cdf, fitting, copula-based multivariate distributions through Sklar's theorem, etc. Since copulas are distribution functions, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API. This allows interoperability with the broader ecosystem, based on this API, such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

Usually, people that use and work with copulas turn to R, because of the amazing package [`R::copula`](https://cran.r-project.org/web/packages/copula/copula.pdf). While well-maintained and regularly updated, `R::copula` is a mixture of obscure, heavily optimized `C` code and more standard `R` code, which makes it a complicated code base for readability, extensibility, reliability and maintenance.

This is an attempt to provide a very light, fast, reliable and maintainable copula implementation in native Julia. One of the notable benefits of such a native implementation (among others) is the floating point type agnosticity, i.e. compatibility with `BigFloat`, [`DoubleFloats`](https://github.com/JuliaMath/DoubleFloats.jl), [`MultiFloats`](https://github.com/dzhang314/MultiFloats.jl), etc.

The package revolves around two main types: 

- `Copula`, an abstract mother type for all the copulas in the package
- `SklarDist`, a distribution type that allows construction of a multivariate distribution by specifying the copula and the marginals through [Sklar's theorem](https://en.wikipedia.org/wiki/Copula_(probability_theory)#Sklar's_theorem). 

**Warning: This is fairly experimental work, use with caution.**

## Getting started

The package is registered in Julia's General registry so you may simply install the package by running : 

```julia
] add Copulas
```

The API contains random number generation, cdf and pdf evaluation, and the `fit` function from `Distributions.jl`. A typical use case might look like this: 

```julia
using Copulas, Distributions, Random
X₁ = Gamma(2,3)
X₂ = Pareto()
X₃ = LogNormal(0,1)
C = ClaytonCopula(3,0.7) # A 3-variate Clayton Copula with θ = 0.7
D = SklarDist(C,(X₁,X₂,X₃)) # The final distribution

simu = rand(D,1000) # Generate a dataset

# You may estimate a copula using the `fit` function:
D̂ = fit(SklarDist{ClaytonCopula,Tuple{Gamma,Normal,LogNormal}}, simu)
```

The list of availiable copula models is *very* large, check it out on our [documentation](https://lrnv.github.io/Copulas.jl/stable) ! 

The general implementation philosophy is for the code to follow the mathematical boundaries of the implemented concepts. For example, this is the only implementation we know (in any language) that allows for **all** Archimedean copulas to be sampled: we use the Williamson transformation for non-standard generators, including user-provided black-box ones.

## Feature comparison


There are competing packages in Julia, such as [`BivariateCopulas.jl`](https://github.com/AnderGray/BivariateCopulas.jl) which only deals with a few models in bivariate settings but has very nice graphs, or [`DatagenCopulaBased.jl`](https://github.com/iitis/DatagenCopulaBased.jl), which only provides sampling and does not have exactly the same models as `Copulas.jl`. While not fully covering out both of these package's functionality (mostly because the three projects chose different implementation paths), `Copulas.jl` brings, as a key feature, the compliance with the broader ecosystem. The following table provides a feature comparison between the three: 

|  | `Copulas.jl` | `DatagenCopulaBased.jl` | `BivariateCopulas.jl` |
|----------------|--------------|-------------------------|-----------------------|
| `Distributions.jl`'s API | ✔️ | ❌ | ✔️ |
| Fitting                  | ✔️ | ❌ | ❌ |
| Plotting                 | ❌ | ❌ | ✔️ |
| Available copulas        |     |     |    |
| - Classic Bivariate      | ✔️ | ✔️ | ✔️ |
| - Classic Multivariate   | ✔️ | ✔️ | ❌ |
| - Archimedeans           | ✔️ (All of them) | ⚠️ Selected ones | ⚠️Selected ones |
| - Bivariate Extreme Value| ✔️ | ❌ | ❌ |
| - Obscure Bivariate      | ✔️ | ❌ | ❌ |
| - Archimedean Chains     | ❌ | ✔️ | ❌ |

Since our primary target is maintainability and readability of the implementation, we did not consider the efficiency and the performance of the code yet. Proper benchmarks will come in the near future. 

## Contributions are welcome

If you want to contribute to the package, ask a question, found a bug or simply want to chat, do not hesitate to open an issue on this repo. General guidelines on collaborative practices (colprac) are available at https://github.com/SciML/ColPrac.
## Citation 

Do not hesitate to star this repository to show support. If you use this package in your researches, please cite it with the following bibtex code:

```bibtex
@article{LavernyJimenez2024,
    author = {Oskar Laverny and Santiago Jimenez}, 
    title = {Copulas.jl: A fully Distributions.jl-compliant copula package},
    journal = {Journal of Open Source Software},
    doi = {10.21105/joss.06189}, 
    url = {https://doi.org/10.21105/joss.06189}, 
    year = {2024}, 
    publisher = {The Open Journal}, 
    volume = {9}, 
    number = {94}, 
    pages = {6189}
}
```
