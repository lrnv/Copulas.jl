```@meta
CurrentModule = Copulas
```

# Development roadmap

We plan to implement additional copula models in this package. The next ones to be implemented will likely be:
- Liouville
- Nested Archimedeans (for any generators, with automatic nesting condition checks)
- Bernstein copula and more general Beta copula as smoothings of the Empirical copula
- `CheckerboardCopula` (and more generally `PatchworkCopula`)
- Allow the user to choose the fitting method via `fit(dist, data; method="MLE")`, `fit(dist, data; method="itau")`, or `fit(dist, data; method="irho")`
- Fitting a generic archimedean with an empirically produced generator

**Possible future additions:**
- `NestedArchimedean`, with automatic checking of nesting conditions for generators
- `Vines`
- `Archimax`
- `BernsteinCopula` and `BetaCopula`
- `PatchworkCopula` and `CheckerboardCopula`
- Goodness-of-fit tests

