# Documentation and API scope — #431, #445, #447

Only unfinished work belongs here; remove completed items as changes are committed.

- Confirm the retained mathematical surface (`A`, `𝒲₋₁`, `IndependentGenerator`)
  and its relationship to internal Pickands/frailty protocols. Specify inverse
  Williamson results by distribution semantics, not concrete type or continuity.
- Audit docstrings of public components: distinguish constructors and mathematical
  meaning from fields, type parameters, caches and specialization hooks. Move
  implementation contracts to the developer/internal reference.
- Align the developer guide's compatibility definition with documented extensions
  of Distributions.jl and StatsBase.jl; do not promise a stable contributor protocol.
- Correct #445's fitting skeleton and distinguish custom fitting from generic
  engine opt-in; replace the inconsistent Nelsen2/boundary examples.
- Correct EV dimension validity, sampling and smoothness claims; update Tail and
  EllipticalCopula descriptions to the actual object-based architecture.
- Audit conditioning return values and qualify derivative formulas and Rosenblatt
  inverses for singular/atomic models; document SurvivalCopula fitting configuration.
- Correct generator monotonicity order, empirical-generator return semantics,
  Archimedean formula, dependence terminology and input-shape/pseudos typos.
- Review the public operation table against actual supported cases, including the
  newly added hypothesis tests and family selection; link authoritative docstrings.
- Relabel/reorganize internal primitive tests currently presented as public API
  contracts without deleting assertions or weakening coverage.
- Verify public/internal docstring placement and links with Documenter in CI;
  review the final diff for remaining contradictory promises before release.
