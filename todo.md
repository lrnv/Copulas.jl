# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

## Runtime and completion

- Record compilation and execution timings by test group and compare them with
  the historical baseline.
- Remove redundant model/operation combinations and excessive numerical work
  while preserving the functional and mathematical coverage above.
- Delete this file in the commit that completes the migration.
