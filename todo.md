# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

## Proof-obligation completeness

- Add a mechanical proof registry for every deterministic operation, analogous
  to the existing Rosenblatt route assertion, so that every route selected by
  the public fixtures is linked to either a generic oracle, an equivalence test,
  or an explicit independent identity.
- Complete density proofs: bivariate specialized `logpdf` routes need numerical
  CDF-derivative comparisons, while multivariate Archimedean, extreme-value,
  elliptical, Liouville, nested, and composed routes need one independent
  density identity per implementation mechanism.
- Complete conditioning proofs for distinct multivariate specialization routes;
  the current exhaustive comparison only covers continuous bivariate
  distortions.
- Account separately for every specialized `inverse_rosenblatt` route instead
  of inferring its coverage from the forward-transform registry.
- Complete dependence-measure specialization proofs for singular and mixed
  families and explicitly account for `gamma` and entropy routes, which are
  currently excluded from the generic-equivalence loop.
- Complete fitting correctness beyond availability and parameter round trips:
  retain defining-statistic tests for inverse estimators and add inexpensive
  recovery/optimality oracles for each distinct MLE and empirical-estimator
  mechanism.
- Add specialization-proof registries for generator and tail primitives. In
  particular, validate distinct analytic `ellpartial` routes against numerical
  derivatives wherever smooth, with explicit atom/non-differentiability
  identities for spectral tails.
- Record the marker-only contract of `IndependentGenerator`, `MGenerator`, and
  `WGenerator`, and the reduced numerical contract of `EmpiricalGenerator`, so
  the public generator registry does not imply unsupported smooth primitives.

## Runtime and completion

- Record compilation and execution timings by test group and compare them with
  the historical baseline.
- Remove redundant model/operation combinations and excessive numerical work
  while preserving the functional and mathematical coverage above.
- Delete this file in the commit that completes the migration.
