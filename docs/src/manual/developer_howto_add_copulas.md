```@meta
CurrentModule = Copulas
```

# How to add new copulas

This guide shows how to extend Copulas.jl with new models. Start with the minimal contracts, then see family-specific notes.

## Minimal contracts

- Type: subtype `Copula{d}` or reuse existing wrappers (e.g., `SurvivalCopula`, `SubsetCopula`).
- Evaluation: implement `_cdf(C::YourCopula, u)`; optionally `Distributions._logpdf` for performance.
- Parameters: define `Distributions.params(C)::NamedTuple` and a constructor `YourCopula(d, ...)`.
- Fitting (optional):
  - `_available_fitting_methods(::Type{YourCopula}) = (:mle, :itau, :irho, :ibeta)` (or a subset, or others)
  - `_fit(::Type{YourCopula}, U, ::Val{:mymethod}; kwargs...) = (Ĉ, (; vcov=?, converged=?, ...))`
  - For generic bindings, also implement `_unbound_params`, `_rebound_params`, and `_example`.

Family-specific guidance:

## Archimedean

Prefer implementing a `Generator` (Williamson-transformable). Provide `ϕ`, `ϕ⁻¹`, and `williamson_dist`. See the Archimedean manual.

## Extreme value

Implement the Pickands dependence function `A(C::ExtremeValueCopula)`. Sampling/logpdf fallbacks exist; specialized fast-paths are welcome.

## Archimax

Compose an Archimedean generator with an extreme-value tail as in `ArchimaxCopula`. Provide the pair and leverage existing building blocks.

## Elliptical

Reuse `GaussianCopula` or `TCopula` patterns: parameterize with correlation (and ν for t). Ensure `Distributions.params` and fit hooks.

## Miscellaneous/others

For specialized bivariate families (Plackett, FGMC, Raftery...), follow existing files as templates. Implement `_cdf` and, when easy, `Distributions._logpdf` and conditioning distortions.

## Extras and tooling

- Conditioning/subsetting: add specialized `Distortion`/`ConditionalCopula` only when it yields clear speedups.
- Show: extend pretty-printing in `src/show.jl` when adding notable types.
- Tests and docs: add a short example to the bestiary and a smoke test to `test/GenericTests.jl`.
