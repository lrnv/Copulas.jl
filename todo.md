# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

## Proof-obligation completeness

- Add a mechanical proof registry for every deterministic operation, analogous
  to the existing Rosenblatt route assertion, so that every route selected by
  the public fixtures is linked to either a generic oracle, an equivalence test,
  or an explicit independent identity.
- Complete multivariate density proofs with one independent identity for each
  Archimedean, extreme-value, elliptical, Liouville, nested, and composed
  implementation mechanism.
- Complete dependence-measure specialization proofs for singular and mixed
  families and explicitly account for `gamma` and entropy routes, which are
  currently excluded from the generic-equivalence loop.
- Complete fitting correctness for empirical-estimator mechanisms; inverse
  estimators have defining-statistic tests and MLE routes now have an
  in-sample likelihood optimality check.
- Add specialization-proof registries for generator and tail primitives. In
  particular, add explicit atom/non-differentiability identities for spectral
  `ellpartial` routes; smooth routes are checked against numerical derivatives.

## Runtime and completion

- Record compilation and execution timings by test group and compare them with
  the historical baseline.
- Remove redundant model/operation combinations and excessive numerical work
  while preserving the functional and mathematical coverage above.
- Delete this file in the commit that completes the migration.
