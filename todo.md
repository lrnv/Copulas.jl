# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

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
- For each specialization, compare its result with the generic reference using
  `invoke` where dispatch permits it, or a narrowly named internal generic helper
  where it does not. Cover CDF/PDF, sampling laws, conditioning, Rosenblatt,
  generators, tails, transforms and measure inverses as applicable.
- Record paths that have no meaningful generic equivalent (notably atoms and
  some spectral samplers) and validate them directly with category-appropriate
  mathematical identities instead of forcing a continuous comparison.

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
