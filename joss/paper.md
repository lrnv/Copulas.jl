---
title: 'Copulas.jl: A fully Distributions.jl-compliant copula package'
tags:
  - julia
  - copula dependence statistics
authors:
  - name: Oskar Laverny
    orcid: 0000-0002-7508-999X
    corresponding: true
    affiliation: "1, 2"
  - name: YOUR NAME
    orcid: 0000-0000-0000-0000 YOUR ORCID
    affiliation: "3, 4"
affiliations:
 - name: Aix-Marseille University
   index: 1
 - name: SESSTIM
   index: 2
 - name: YOUR FIRST AFFILIATION
   index: 3
 - name: YOUR SECOND AFFILIATION
   index: 4
date: 16 November 2023
bibliography: paper.bib

---

# Summary

`Copulas.jl` brings most standard [copula](https://en.wikipedia.org/wiki/Copula_(probability_theory)) features into native Julia: random number generation, pdf and cdf, fitting, copula-based multivariate distributions through Sklar's theorem, etc. Since copulas are distribution functions, we fully comply with the [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) API. This complience allows interoperability with other packages based on this API such as, e.g., [`Turing.jl`](https://github.com/TuringLang/Turing.jl).

# Statement of need

A little longer, maybe 30 to 40 lines, describing the need for a propper copula package in julia. 

## Examples

A few exemples, link to the documentation, etc. Of course references must be included. This can be half a page or a bit more. 


```julia
using Copulas

# We may include some julia code..
```

# Acknowledgments

If you have to Acknowledge some fundings that might be here. I dont think I do. 


# References