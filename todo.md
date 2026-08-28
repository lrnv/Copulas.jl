# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

## Proof-obligation completeness

- Add a mechanical proof registry for every deterministic operation, analogous
  to the existing Rosenblatt route assertion, so that every route selected by
  the public fixtures is linked to either a generic oracle, an equivalence test,
  or an explicit independent identity.

## Runtime and completion

- Record compilation and execution timings by test group and compare them with
  the historical baseline.
- Remove redundant model/operation combinations and excessive numerical work
  while preserving the functional and mathematical coverage above.
- Delete this file in the commit that completes the migration.
