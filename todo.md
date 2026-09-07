# Documentation and API scope — #431, #445, #447

Only unfinished work belongs here; remove completed items as changes are committed.

- Verify the rendered placement of the revised public component docstrings and
  their qualified non-exported names with Documenter.
- Verify the new contributor examples in Documenter CI.
- Finish the audit of conditioning return values and document SurvivalCopula
  fitting configuration.
- Correct empirical-generator return semantics, dependence terminology and
  input-shape/pseudos typos.
- Review the public operation table against actual supported cases, including the
  newly added hypothesis tests and family selection; link authoritative docstrings.
- Relabel/reorganize internal primitive tests currently presented as public API
  contracts without deleting assertions or weakening coverage.
- Verify public/internal docstring placement and links with Documenter in CI;
  review the final diff for remaining contradictory promises before release.
