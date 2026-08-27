# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

## Mathematical correctness

- Inventory the mathematical properties checked by the former generic suite and
  classify each as a universal invariant, a mechanism-level check, or a
  family-specific regression.
- Complete representative coherence tests for CDF/PDF integration and
  differentiation, rectangle probabilities, conditional distributions,
  Rosenblatt transforms, dependence measures, generators, extreme-value tails,
  Archimax constructions, and radial/Kendall representations.
- Cover singular and mixed copulas with their mathematically appropriate
  properties instead of applying continuous-density or bijection assumptions.

## Shared components

- Add focused contracts for distortions, radial and other internal univariate
  distributions, spectral representations, and shared samplers where testing
  only through complete copulas would obscure failures or duplicate compilation.
- Check boundary behavior, generalized quantiles, atoms, inverse identities,
  derivatives, support, and numerical fallbacks wherever relevant.

## Dispatch and numerical paths

- Build an explicit, compact registry containing one representative for every
  generic fallback, closed form, sampler, conditioning implementation,
  quadrature path, and relevant numeric type/dimension path.
- Verify that specialized paths agree with their generic references where this
  can be done cheaply, without recreating a cartesian copula-by-operation matrix.
- Audit the registry against the implementation so that no public mechanism or
  fast path is exercised only accidentally.

## Family and extension regressions

- Migrate the useful family-specific coverage still living in `test/old/` into
  focused family or extension tests: published reference values, limiting cases,
  constructor validation, numerical corner cases, and previously fixed bugs.
- Remove each legacy assertion in the same commit that introduces its classified
  replacement; do not retain generic API checks in family files.

## Runtime and completion

- Record compilation and execution timings by test group and compare them with
  the historical baseline.
- Remove redundant model/operation combinations and excessive numerical work
  while preserving the functional and mathematical coverage above.
- Remove `test/old/` once its last useful test has been migrated.
- Delete this file in the commit that completes the migration.
