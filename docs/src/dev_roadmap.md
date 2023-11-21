```@meta
CurrentModule = Copulas
```

# Development roadmap

We hope to implement a few more copula models into this package. The next ones to be implemented will probably be: 
- Extreme values copulas. 
- Nested archimedeans (for any generators, with automatic nesting conditions checking). 
- Bernstein copula and more general Beta copula as smoothing of the Empirical copula. 
- `CheckerboardCopula` (and more generally `PatchworkCopula`)

More precisely, the following directions could be pursued:

**Next:**
- More documentation and tests for the current implementation. 
- Docs: show how to use the `WilliamsonCopula` to implement generic archimedeans.
- Give the user the choice of fitting method via `fit(dist,data; method="MLE")` or `fit(dist,data; method="itau")` or `fit(dist,data; method="irho")`.
- Fitting a generic archimedean with an empirically produced generator
- Automatic checking of generator d-monotony ? Dunno if it is even possible. 

**Maybe later:**
- `NestedArchimedean`, with automatic checking of nesting conditions for generators. 
- `Vines`?
- `Archimax` ?
- `BernsteinCopula` and `BetaCopula` could also be implemented. 
- `PatchworkCopula` and `CheckerboardCopula`: could be nice things to have :)
- Goodness of fits tests ?

