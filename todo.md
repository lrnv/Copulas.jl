# Remaining test-architecture work

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.
- Migrate the useful family-specific and regression coverage still living in
  `test/old/`, removing each legacy assertion when its replacement is committed.
- Record timings by test group, compare them with the historical baseline, and
  remove `test/old/` once its last useful test has been migrated.
- Delete this file in the commit that completes the migration.
