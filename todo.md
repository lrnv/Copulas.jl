# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

## Mathematical reference implementations

The correctness argument should form an explicit chain rather than a matrix of
families and operations:

1. validate each generic implementation against an independent mathematical
   oracle;
2. compare every specialized dispatch path with the corresponding generic
   implementation on safe interior inputs;
3. apply the public API contract to every concrete public family;
4. retain family tests only for published values, parameter limits, atoms and
   regressions which cannot be inferred from the generic implementation.

- Introduce the smallest possible test-only reference types rather than using
  production families whose closed forms may bypass the code under test:
  - a smooth copula with independently known CDF, density, rectangle masses,
    conditionals and Rosenblatt transform;
  - a generator defining only its core function, so generic inversion and AD
    derivatives are exercised;
  - a tail defining only its STDF, so generic Pickands, partial derivative,
    extreme-value density and conditioning machinery are exercised;
  - a simple radial distribution for the generic Williamson transform and its
    real-order inverse.
- Use one reference type per genuinely different mathematical category. A
  continuous oracle must not be used to justify discrete, singular or mixed
  behavior; those require mass and generalized-quantile identities.
- For every reference type, check all applicable independent identities:
  Frechet bounds and uniform margins, CDF/PDF integration and differentiation,
  inclusion-exclusion and additivity of rectangle masses, normalized
  conditional derivatives and densities, Rosenblatt factorization and inverse,
  dependence-measure definitions, generator inverse/derivative/monotonicity
  identities, tail bounds/homogeneity/convexity/max-stability, and Williamson
  transform identities.
- Avoid circular tests: a generic CDF defined as the integral of a density needs
  an analytic CDF oracle; a derivative fallback needs an independently known
  derivative; two paths sharing the same helper are not independent evidence.
- Keep comparisons away from parameter and support boundaries unless boundary
  behavior is itself the property under test. Specialized numerical stabilization
  may legitimately differ from a generic reference at those boundaries.
- Replace the current production-family representatives in
  `paths/mathematical_coherence.jl` as each generic oracle becomes available;
  do not keep both versions without a distinct coverage reason.
- Consider this layer complete only when every generic mathematical fallback is
  mapped to an independent oracle, with exceptions explicitly documented.

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
