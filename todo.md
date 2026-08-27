# Remaining test-architecture work

## Public API contract

- Make the new public-contract suite pass in CI and resolve every behavioral
  discrepancy it exposes.

## Complete the four proof obligations

- Inventory every deterministic specialization of CDF/density, subsetting,
  conditioning, Rosenblatt transforms, dependence measures, fitting, generator
  primitives, and tail primitives.
- Compare each specialization with its generic fallback at one interior point,
  or with an independent identity when the fallback is not applicable.
- Make the dispatch registries prove that every public family reaches a checked
  generic or specialized path for each public behaviour.
- Add a focused contract for the Plots extension and remove the remaining
  untested advertised fitting-path exceptions.

## Runtime and completion

- Record compilation and execution timings by test group and compare them with
  the historical baseline.
- Remove redundant model/operation combinations and excessive numerical work
  while preserving the functional and mathematical coverage above.
- Delete this file in the commit that completes the migration.
