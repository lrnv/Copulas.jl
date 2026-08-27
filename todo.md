# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

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
